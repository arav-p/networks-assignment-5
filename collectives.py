"""
Custom implementations of collective communication algorithms using
the PyTorch torch.distributed gloo backend.

Algorithms implemented:
  AllGather: Ring, Recursive Doubling, Swing
  Broadcast: Binary Tree, Binomial Tree

All routines use only point-to-point primitives (dist.send / dist.recv /
dist.isend / dist.irecv). Built-in collectives are NOT used for the
algorithms under test; they are only optionally used as a correctness
reference.
"""

import math
import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# AllGather implementations
# ---------------------------------------------------------------------------

def allgather_ring(local_chunk: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
    """
    Ring AllGather.

    Each rank starts with a chunk of size C. After P-1 steps of
    send-to-(rank+1) / recv-from-(rank-1), every rank holds all P chunks.
    Total data moved per rank: (P-1)*C.
    """
    chunk_size = local_chunk.numel()
    output = torch.empty(world_size * chunk_size, dtype=local_chunk.dtype)
    output[rank * chunk_size : (rank + 1) * chunk_size].copy_(local_chunk)

    send_to = (rank + 1) % world_size
    recv_from = (rank - 1) % world_size

    for s in range(world_size - 1):
        send_slot = (rank - s) % world_size
        recv_slot = (rank - s - 1) % world_size

        send_buf = output[send_slot * chunk_size : (send_slot + 1) * chunk_size].contiguous()
        recv_buf = torch.empty(chunk_size, dtype=local_chunk.dtype)

        send_req = dist.isend(send_buf, dst=send_to)
        dist.recv(recv_buf, src=recv_from)
        send_req.wait()

        output[recv_slot * chunk_size : (recv_slot + 1) * chunk_size].copy_(recv_buf)

    return output


def allgather_recursive_doubling(local_chunk: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
    """
    Recursive Doubling AllGather (power-of-two world size).

    log2(P) rounds. At round i, rank r exchanges its currently-held block
    (size 2^i * chunk) with peer (r XOR 2^i); the block doubles each round.
    """
    assert (world_size & (world_size - 1)) == 0, \
        "recursive doubling requires power-of-two world size"

    chunk_size = local_chunk.numel()
    output = torch.empty(world_size * chunk_size, dtype=local_chunk.dtype)
    output[rank * chunk_size : (rank + 1) * chunk_size].copy_(local_chunk)

    num_steps = int(math.log2(world_size))

    for i in range(num_steps):
        mask = 1 << i
        peer = rank ^ mask

        # Block I currently hold is aligned on 2^i and has size 2^i.
        my_block_lo = rank & ~((1 << i) - 1) if i > 0 else rank
        # But after step 0, the block covers two indices aligned on 2^1, etc.
        # Precisely: after completing j steps, my block spans [r & ~((1<<j)-1),
        # r & ~((1<<j)-1) + 2^j). Before step i, j == i.
        block_size = 1 << i
        my_block_lo = (rank & ~(block_size - 1)) if block_size > 1 else rank
        my_block_hi = my_block_lo + block_size

        peer_block_lo = my_block_lo ^ mask
        peer_block_hi = peer_block_lo + block_size

        send_buf = output[my_block_lo * chunk_size : my_block_hi * chunk_size].contiguous()
        recv_buf = torch.empty(block_size * chunk_size, dtype=local_chunk.dtype)

        # Lower-ranked peer sends first to avoid head-of-line blocking.
        if rank < peer:
            s = dist.isend(send_buf, dst=peer)
            dist.recv(recv_buf, src=peer)
            s.wait()
        else:
            r = dist.irecv(recv_buf, src=peer)
            dist.send(send_buf, dst=peer)
            r.wait()

        output[peer_block_lo * chunk_size : peer_block_hi * chunk_size].copy_(recv_buf)

    return output


def _swing_distance(step: int) -> int:
    """
    Signed Swing distance for step `step` (0-indexed).

    We use d_i = (-2)^i, i.e. 1, -2, 4, -8, 16, -32, ...
    This is the "direction-alternating" Swing variant: at each round we
    double the hop and flip direction around the logical ring, which
    (i)  keeps |d_i| under P/2 for the last round  (balanced fan-out),
    (ii) guarantees that the set of partial subset-sums of
         {d_0, ..., d_{log2(P)-1}} covers every residue class mod P
         for P = 2^n, so an AllGather completes in exactly log2(P) rounds.

    The Jacobsthal-based Swing distances from Di Girolamo et al.
    ([+1, -1, +3, -5, +11, ...]) target AllReduce on torus topologies and
    do not, on their own, yield a spanning set for AllGather on every P.
    """
    return (-2) ** step


def _compute_swing_have_sets(world_size: int, num_steps: int):
    """
    Iteratively compute, for every rank r and every step boundary j
    (0..num_steps), the set of chunk-indices r owns at that boundary.

    Returns a list `have[j]` where `have[j][r]` is a frozenset of chunk ids.
    """
    have = [[frozenset({r}) for r in range(world_size)]]
    for i in range(num_steps):
        d = _swing_distance(i)
        next_level = []
        for r in range(world_size):
            src = (r - d) % world_size
            next_level.append(have[i][r] | have[i][src])
        have.append(next_level)
    return have


def allgather_swing(local_chunk: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
    """
    Swing AllGather (power-of-two world size).

    log2(P) rounds. At round i, rank r sends its accumulated chunks to
    peer (r + d_i) mod P and receives from peer (r - d_i) mod P, where
    d_i is the signed Swing distance. Message count per round is 2^i,
    matching recursive doubling in volume but with alternating-sign
    distances that shorten the longest logical hop on ring/torus topologies.
    """
    assert (world_size & (world_size - 1)) == 0, \
        "swing requires power-of-two world size"

    chunk_size = local_chunk.numel()
    output = torch.empty(world_size * chunk_size, dtype=local_chunk.dtype)
    output[rank * chunk_size : (rank + 1) * chunk_size].copy_(local_chunk)

    num_steps = int(math.log2(world_size))
    have_sets = _compute_swing_have_sets(world_size, num_steps)

    for i in range(num_steps):
        d = _swing_distance(i)
        peer_send = (rank + d) % world_size
        peer_recv = (rank - d) % world_size

        my_indices = sorted(have_sets[i][rank])
        incoming_indices = sorted(have_sets[i][peer_recv])

        send_buf = torch.cat([
            output[idx * chunk_size : (idx + 1) * chunk_size] for idx in my_indices
        ]).contiguous()
        recv_buf = torch.empty(len(incoming_indices) * chunk_size, dtype=local_chunk.dtype)

        # Post non-blocking ops first, then wait, to avoid deadlock when
        # peer_send == peer_recv (possible when d = P/2).
        s_req = dist.isend(send_buf, dst=peer_send)
        r_req = dist.irecv(recv_buf, src=peer_recv)
        s_req.wait()
        r_req.wait()

        for k, idx in enumerate(incoming_indices):
            output[idx * chunk_size : (idx + 1) * chunk_size].copy_(
                recv_buf[k * chunk_size : (k + 1) * chunk_size]
            )

    return output


# ---------------------------------------------------------------------------
# Broadcast implementations
# ---------------------------------------------------------------------------

def broadcast_binary_tree(tensor: torch.Tensor, root: int, world_size: int, rank: int) -> torch.Tensor:
    """
    Binary-tree Broadcast.

    Heap-shaped binary tree rooted at `root`: logical index i has children
    2i+1 and 2i+2, parent (i-1)//2. Logical index = (physical_rank - root) mod P.
    """
    logical = (rank - root) % world_size

    if logical != 0:
        parent_logical = (logical - 1) // 2
        parent_rank = (parent_logical + root) % world_size
        dist.recv(tensor, src=parent_rank)

    for child_logical in (2 * logical + 1, 2 * logical + 2):
        if child_logical < world_size:
            child_rank = (child_logical + root) % world_size
            dist.send(tensor, dst=child_rank)

    return tensor


def broadcast_binomial_tree(tensor: torch.Tensor, root: int, world_size: int, rank: int) -> torch.Tensor:
    """
    Binomial-tree Broadcast.

    ceil(log2(P)) rounds. At round i (mask = 2^i), every rank that already
    holds the data and whose logical index has bit i clear sends to the
    peer whose logical index has bit i set (if in range).
    """
    if world_size <= 1:
        return tensor

    logical = (rank - root) % world_size
    num_steps = int(math.ceil(math.log2(world_size)))

    # Invariant at the start of round i (mask = 2^i):
    #   - ranks with logical < mask already have the data
    #   - ranks with mask <= logical < 2*mask receive this round
    #   - ranks with logical >= 2*mask are not yet involved
    for i in range(num_steps):
        mask = 1 << i
        if logical < mask:
            peer_logical = logical + mask
            if peer_logical < world_size:
                peer_rank = (peer_logical + root) % world_size
                dist.send(tensor, dst=peer_rank)
        elif logical < (mask << 1):
            parent_logical = logical - mask
            parent_rank = (parent_logical + root) % world_size
            dist.recv(tensor, src=parent_rank)

    return tensor
