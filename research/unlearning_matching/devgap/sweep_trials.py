"""Dev metrics + dev AvgGap (vs paired Retrain) for every saved epoch of every
discovery trial.  Reads only stored epoch distribution files."""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import re

import numpy as np

import devgap


def one(job):
    trial, sched, path = job
    d = np.load(path)
    m = devgap.metrics(d)
    ep = int(re.search(r"epoch(\d+)", path.name).group(1))
    return dict(trial=trial, schedule=sched, epoch=ep, **m)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--opt-root", type=Path, required=True)
    p.add_argument("--retrain", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--procs", type=int, default=16)
    p.add_argument("--stem", default="seed42_split42")
    a = p.parse_args()
    jobs = []
    for trial in sorted(x for x in a.opt_root.iterdir() if x.is_dir()):
        run = trial / a.stem
        if not run.is_dir():
            continue
        for sched in sorted(x for x in run.iterdir() if x.is_dir() and x.name != "source_snapshot"):
            for f in sorted(sched.glob("epoch*_distributions.npz")):
                jobs.append((trial.name, sched.name, f))
    ref = devgap.metrics(np.load(a.retrain))
    with Pool(a.procs) as pool:
        rows = pool.map(one, jobs, chunksize=4)
    for r in rows:
        r.update({f"gap_{k}": v for k, v in devgap.gap(r, ref).items()})
    a.out.write_text(json.dumps(dict(retrain=ref, rows=rows)) + "\n")
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
