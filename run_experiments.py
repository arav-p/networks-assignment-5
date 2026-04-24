"""
Experiment driver for Assignment 5.

Launches N processes with the gloo backend, runs each custom collective
for a range of message sizes (fixed P sweep) and a range of process counts
(fixed-size sweep), times them, and writes the results to results.json.

Timing method:
  - dist.barrier() synchronises every rank before start.
  - Each rank records t0 = time.perf_counter(), runs the collective,
    records t1 = time.perf_counter(), and reports (t1 - t0).
  - The completion time reported for a trial is max over ranks, because a
    collective is only "done" when the last rank finishes.
  - Each (algorithm, size, P) cell is measured over N_WARMUP warm-up iters
    + N_TRIALS timed iters; reported value = median of timed iters.
"""

import argparse
import json
import os
import sys
import time
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

# Map algorithm name -> function, kind ("allgather" or "broadcast")
ALG_TABLE = {
    "ring":              (allgather_ring,              "allgather"),
    "recursive_doubling":(allgather_recursive_doubling,"allgather"),
    "swing":             (allgather_swing,             "allgather"),
    "binary_tree":       (broadcast_binary_tree,       "broadcast"),
    "binomial_tree":     (broadcast_binomial_tree,     "broadcast"),
}

N_WARMUP = 1
N_TRIALS = 3


def _setup(rank: int, world_size: int, master_port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    # Keep the gloo connect/handshake timeouts generous to avoid flakes
    # during cold starts.
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{master_port}",
    )


def _teardown():
    if dist.is_initialized():
        dist.destroy_process_group()


def _time_one(fn_kind, fn, world_size, rank, total_elems, dtype) -> float:
    """Single timed invocation. Returns elapsed seconds on this rank."""
    if fn_kind == "allgather":
        # Total message is `total_elems`, chunk size = total_elems / P.
        assert total_elems % world_size == 0
        chunk_elems = total_elems // world_size
        local = torch.full((chunk_elems,), float(rank), dtype=dtype)
        dist.barrier()
        t0 = time.perf_counter()
        _ = fn(local, world_size, rank)
        t1 = time.perf_counter()
        return t1 - t0
    else:  # broadcast
        tensor = torch.full((total_elems,), float(rank == 0), dtype=dtype)
        dist.barrier()
        t0 = time.perf_counter()
        _ = fn(tensor, 0, world_size, rank)
        t1 = time.perf_counter()
        return t1 - t0


def _worker(rank: int, world_size: int, master_port: int, job_spec, result_path: str):
    """
    job_spec: list of dicts
        {"alg": str, "bytes": int}
    Each rank runs every job, writes timings to result_path.{rank}.json
    """
    try:
        _setup(rank, world_size, master_port)
        dtype = torch.float32
        elem_bytes = torch.tensor([], dtype=dtype).element_size()

        local_results = []
        for job in job_spec:
            alg = job["alg"]
            nbytes = job["bytes"]
            fn, kind = ALG_TABLE[alg]

            total_elems = nbytes // elem_bytes
            # Round down to multiple of P for allgather
            if kind == "allgather":
                total_elems -= total_elems % world_size
                if total_elems == 0:
                    total_elems = world_size
            else:
                if total_elems == 0:
                    total_elems = 1

            # Warm-up
            for _ in range(N_WARMUP):
                _time_one(kind, fn, world_size, rank, total_elems, dtype)

            times = []
            for _ in range(N_TRIALS):
                t = _time_one(kind, fn, world_size, rank, total_elems, dtype)
                times.append(t)

            local_results.append({
                "alg": alg,
                "bytes": nbytes,
                "actual_elems": total_elems,
                "world_size": world_size,
                "rank": rank,
                "times": times,
            })

        _teardown()

        with open(f"{result_path}.{rank}.json", "w") as f:
            json.dump(local_results, f)

    except Exception as e:
        traceback.print_exc()
        with open(f"{result_path}.{rank}.err", "w") as f:
            f.write(traceback.format_exc())
        try:
            _teardown()
        except Exception:
            pass
        sys.exit(1)


def run_one_config(world_size: int, job_spec, master_port: int, tmp_prefix: str):
    """Spawns `world_size` processes, runs job_spec on each, aggregates timings."""
    # Clean any stale per-rank files
    for r in range(world_size):
        for ext in ("json", "err"):
            p = f"{tmp_prefix}.{r}.{ext}"
            if os.path.exists(p):
                os.remove(p)

    ctx = mp.spawn(
        _worker,
        args=(world_size, master_port, job_spec, tmp_prefix),
        nprocs=world_size,
        join=True,
        daemon=False,
    )

    # Read back and merge
    merged = {}  # (alg, bytes) -> list of per-rank max-time per trial
    for r in range(world_size):
        err_path = f"{tmp_prefix}.{r}.err"
        if os.path.exists(err_path):
            raise RuntimeError(f"Rank {r} failed. See {err_path}")
        path = f"{tmp_prefix}.{r}.json"
        with open(path) as f:
            data = json.load(f)
        for entry in data:
            key = (entry["alg"], entry["bytes"])
            merged.setdefault(key, []).append(entry["times"])

    # For each (alg, bytes), take per-trial max across ranks, then median.
    summary = []
    for (alg, nbytes), per_rank_trials in merged.items():
        n_trials = len(per_rank_trials[0])
        per_trial_max = []
        for t in range(n_trials):
            tr = max(per_rank_trials[r][t] for r in range(world_size))
            per_trial_max.append(tr)
        per_trial_max.sort()
        median = per_trial_max[len(per_trial_max) // 2]
        summary.append({
            "alg": alg,
            "bytes": nbytes,
            "world_size": world_size,
            "time_s": median,
            "trials_s": per_trial_max,
        })

    # Clean temp files
    for r in range(world_size):
        path = f"{tmp_prefix}.{r}.json"
        if os.path.exists(path):
            os.remove(path)

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--max-bytes", type=int, default=8 * 1024 * 1024,
                    help="Max message size for the size-sweep (default: 8 MiB).")
    ap.add_argument("--max-ranks", type=int, default=8,
                    help="Max process count for the rank-sweep (default: 8).")
    ap.add_argument("--fixed-size", type=int, default=1 * 1024 * 1024,
                    help="Message size used in the rank-sweep (default: 1 MiB).")
    ap.add_argument("--fixed-ranks", type=int, default=4,
                    help="World size used in the size-sweep (default: 4).")
    args = ap.parse_args()

    # Build message-size list: powers of two from 1 KiB up to --max-bytes
    sizes = []
    s = 1024
    while s <= args.max_bytes:
        sizes.append(s)
        s *= 4  # 1K, 4K, 16K, 64K, 256K, 1M, 4M, 16M, ...
    if sizes[-1] != args.max_bytes:
        # keep the terminal size just in case
        pass

    # Build rank list: powers of two 2..max_ranks
    ranks = []
    p = 2
    while p <= args.max_ranks:
        ranks.append(p)
        p *= 2

    allgather_algs = ["ring", "recursive_doubling", "swing"]
    broadcast_algs = ["binary_tree", "binomial_tree"]

    all_results = {
        "size_sweep": [],   # fixed ranks, varying size
        "rank_sweep": [],   # fixed size, varying ranks
        "meta": {
            "sizes_bytes": sizes,
            "ranks": ranks,
            "fixed_ranks": args.fixed_ranks,
            "fixed_size_bytes": args.fixed_size,
            "dtype": "float32",
            "n_warmup": N_WARMUP,
            "n_trials": N_TRIALS,
        },
    }

    base_port = 29600

    # ---- Size sweep at fixed world size ----
    print(f"[size-sweep] world_size={args.fixed_ranks}, sizes={sizes}", flush=True)
    jobs = []
    for alg in allgather_algs + broadcast_algs:
        for nb in sizes:
            jobs.append({"alg": alg, "bytes": nb})
    out = run_one_config(args.fixed_ranks, jobs, base_port, "/tmp/assn5_size")
    all_results["size_sweep"].extend(out)
    for row in sorted(out, key=lambda r: (r["alg"], r["bytes"])):
        print(f"  {row['alg']:20s} {row['bytes']:>10d}B  {row['time_s']*1e3:8.3f} ms", flush=True)

    # ---- Rank sweep at fixed size ----
    for ws in ranks:
        print(f"[rank-sweep] world_size={ws}, size={args.fixed_size}B", flush=True)
        jobs = [{"alg": alg, "bytes": args.fixed_size}
                for alg in allgather_algs + broadcast_algs]
        port = base_port + ws
        out = run_one_config(ws, jobs, port, f"/tmp/assn5_rank_{ws}")
        all_results["rank_sweep"].extend(out)
        for row in sorted(out, key=lambda r: r["alg"]):
            print(f"  {row['alg']:20s} P={row['world_size']:<3d}  {row['time_s']*1e3:8.3f} ms",
                  flush=True)

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
