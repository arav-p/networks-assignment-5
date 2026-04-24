"""Produce the four required plots from results.json."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALLGATHER_ALGS = ["ring", "recursive_doubling", "swing"]
BROADCAST_ALGS = ["binary_tree", "binomial_tree"]

LABELS = {
    "ring": "Ring",
    "recursive_doubling": "Recursive Doubling",
    "swing": "Swing",
    "binary_tree": "Binary Tree",
    "binomial_tree": "Binomial Tree",
}
MARKERS = {
    "ring": "o",
    "recursive_doubling": "s",
    "swing": "^",
    "binary_tree": "o",
    "binomial_tree": "s",
}


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.0f}TB"


def plot_size_sweep(results, outdir):
    fixed_P = results["meta"]["fixed_ranks"]
    rows = results["size_sweep"]

    by_alg = {}
    for r in rows:
        by_alg.setdefault(r["alg"], []).append(r)

    for kind, algs, title, fname in [
        ("allgather", ALLGATHER_ALGS,
         f"AllGather completion time vs. message size (P={fixed_P})",
         "allgather_vs_size.png"),
        ("broadcast", BROADCAST_ALGS,
         f"Broadcast completion time vs. message size (P={fixed_P})",
         "broadcast_vs_size.png"),
    ]:
        plt.figure(figsize=(8, 5))
        for alg in algs:
            if alg not in by_alg: continue
            pts = sorted(by_alg[alg], key=lambda r: r["bytes"])
            xs = [r["bytes"] for r in pts]
            ys = [r["time_s"] * 1e3 for r in pts]
            plt.plot(xs, ys, marker=MARKERS[alg], label=LABELS[alg], linewidth=1.8)
        plt.xscale("log", base=2)
        plt.yscale("log")
        plt.xlabel("Message size (bytes)")
        plt.ylabel("Completion time (ms)")
        plt.title(title)
        plt.grid(True, which="both", linestyle=":", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        path = os.path.join(outdir, fname)
        plt.savefig(path, dpi=140)
        plt.close()
        print(f"  wrote {path}")


def plot_rank_sweep(results, outdir):
    fixed_bytes = results["meta"]["fixed_size_bytes"]
    rows = results["rank_sweep"]

    by_alg = {}
    for r in rows:
        by_alg.setdefault(r["alg"], []).append(r)

    for kind, algs, title, fname in [
        ("allgather", ALLGATHER_ALGS,
         f"AllGather completion time vs. number of processes (message = {_human_bytes(fixed_bytes)})",
         "allgather_vs_ranks.png"),
        ("broadcast", BROADCAST_ALGS,
         f"Broadcast completion time vs. number of processes (message = {_human_bytes(fixed_bytes)})",
         "broadcast_vs_ranks.png"),
    ]:
        plt.figure(figsize=(8, 5))
        for alg in algs:
            if alg not in by_alg: continue
            pts = sorted(by_alg[alg], key=lambda r: r["world_size"])
            xs = [r["world_size"] for r in pts]
            ys = [r["time_s"] * 1e3 for r in pts]
            plt.plot(xs, ys, marker=MARKERS[alg], label=LABELS[alg], linewidth=1.8)
        plt.xlabel("Number of processes (P)")
        plt.ylabel("Completion time (ms)")
        plt.title(title)
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        path = os.path.join(outdir, fname)
        plt.savefig(path, dpi=140)
        plt.close()
        print(f"  wrote {path}")


def main():
    with open("results.json") as f:
        results = json.load(f)
    outdir = "plots"
    os.makedirs(outdir, exist_ok=True)
    plot_size_sweep(results, outdir)
    plot_rank_sweep(results, outdir)
    print("Done.")


if __name__ == "__main__":
    main()
