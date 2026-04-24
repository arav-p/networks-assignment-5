"""
Correctness tests: run each custom algorithm and compare its output
against the PyTorch built-in collective on the same input.
"""
import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from collectives import (
    allgather_ring,
    allgather_recursive_doubling,
    allgather_swing,
    broadcast_binary_tree,
    broadcast_binomial_tree,
)


def _setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        backend="gloo", rank=rank, world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}"
    )


def _worker(rank, world_size, port, result_file):
    try:
        _setup(rank, world_size, port)
        failures = []

        # AllGather tests
        chunk_elems = 17  # odd number to check no alignment assumption
        local = torch.arange(
            rank * chunk_elems, (rank + 1) * chunk_elems, dtype=torch.float32
        )
        expected = torch.arange(world_size * chunk_elems, dtype=torch.float32)

        for name, fn in [
            ("ring", allgather_ring),
            ("recursive_doubling", allgather_recursive_doubling),
            ("swing", allgather_swing),
        ]:
            try:
                out = fn(local.clone(), world_size, rank)
                if not torch.equal(out, expected):
                    failures.append(f"[rank {rank}] allgather {name}: mismatch. "
                                    f"got[:5]={out[:5].tolist()} expected[:5]={expected[:5].tolist()}")
            except Exception as e:
                failures.append(f"[rank {rank}] allgather {name}: {e}\n{traceback.format_exc()}")

        # Broadcast tests (root = 0, 1, 2 where possible)
        for root in range(min(3, world_size)):
            for name, fn in [
                ("binary_tree", broadcast_binary_tree),
                ("binomial_tree", broadcast_binomial_tree),
            ]:
                try:
                    if rank == root:
                        t = torch.arange(100, dtype=torch.float32) + 7.0
                    else:
                        t = torch.zeros(100, dtype=torch.float32)
                    fn(t, root, world_size, rank)
                    expected_t = torch.arange(100, dtype=torch.float32) + 7.0
                    if not torch.equal(t, expected_t):
                        failures.append(
                            f"[rank {rank}] broadcast {name} root={root}: mismatch. "
                            f"got[:5]={t[:5].tolist()}"
                        )
                except Exception as e:
                    failures.append(f"[rank {rank}] broadcast {name} root={root}: {e}\n{traceback.format_exc()}")

        dist.destroy_process_group()

        with open(f"{result_file}.{rank}", "w") as f:
            f.write("\n".join(failures))

    except Exception:
        with open(f"{result_file}.{rank}", "w") as f:
            f.write(f"FATAL: {traceback.format_exc()}")


def main():
    any_fail = False
    for ws in [2, 4, 8]:
        prefix = f"/tmp/assn5_test_{ws}"
        for r in range(ws):
            p = f"{prefix}.{r}"
            if os.path.exists(p): os.remove(p)
        print(f"== world_size = {ws} ==", flush=True)
        mp.spawn(_worker, args=(ws, 30000 + ws, prefix), nprocs=ws, join=True)
        for r in range(ws):
            p = f"{prefix}.{r}"
            if os.path.exists(p):
                content = open(p).read().strip()
                if content:
                    print(content)
                    any_fail = True
                os.remove(p)
    if any_fail:
        print("FAILED"); sys.exit(1)
    print("ALL CORRECTNESS TESTS PASSED")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
