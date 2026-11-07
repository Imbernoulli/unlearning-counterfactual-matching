"""Official-test evaluation of frozen checkpoints against paired Retrain.

Calls the unchanged benchmark evaluate.evaluate_state (UA/RA/TA/MIA) on the
full CIFAR-10 arrays for each candidate and its paired Retrain, computes the
four absolute gaps per run, then mean and sample SD across runs.  Only run
after the method, hyperparameters and epoch are frozen.

Plan JSON: {"runs": [{"id": ..., "splits": path, "retrain": path,
                      "candidates": {"name": ckpt_path, ...}}]}
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "tasks/security-machine-unlearning/scripts"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    import sys
    sys.path.insert(0, str(TASK))
    common = load("common", TASK / "common.py")
    sys.modules["common"] = common
    ev = load("bench_evaluate", TASK / "evaluate.py")
    with np.load(a.data) as z:
        arrays = (z["x_train"], z["y_train"], z["x_test"], z["y_test"])
    if common.sha256_arrays(*arrays) != common.SHA256_DATA["cifar10"]:
        raise RuntimeError("dataset hash mismatch")
    plan = json.loads(a.plan.read_text())
    results = json.loads(a.out.read_text()) if a.out.exists() else {"runs": {}}
    # cache keyed by checkpoint sha256 + forget-set hash; aggregate only runs in this plan
    cache = results.setdefault("cache", {})
    results["runs"] = {}
    for run in plan["runs"]:
        forget = json.loads(Path(run["splits"]).read_text())["indices"]["forget"]
        fh = hashlib.sha256(json.dumps(sorted(forget)).encode()).hexdigest()[:16]
        rec = results["runs"].setdefault(run["id"], {})
        todo = {"retrain": run["retrain"], **run["candidates"]}
        for name, path in todo.items():
            key = common.sha256_file(path) + ":" + fh
            if key not in cache:
                state = torch.load(path, map_location="cpu", weights_only=True)
                m = ev.evaluate_state("cifar10", state, forget, arrays, .1)
                cache[key] = {k: m[k] for k in ("ua", "ra", "ta", "mia")}
                a.out.write_text(json.dumps(results, indent=1))
            rec[name] = dict(cache[key], checkpoint=str(path), key=key)
            print(run["id"], name, {k: round(v, 3) for k, v in cache[key].items()}, flush=True)
        ref = rec["retrain"]
        for name in run["candidates"]:
            g = {k: abs(rec[name][k] - ref[k]) for k in ("ua", "ra", "ta", "mia")}
            g["avg"] = sum(g.values()) / 4
            rec[name]["gap"] = g
    agg = {}
    names = sorted({n for r in results["runs"].values() for n in r if n != "retrain"})
    for name in names:
        rows = [r[name] for r in results["runs"].values() if name in r and "gap" in r[name]]
        if not rows:
            continue
        sd = (lambda v: statistics.stdev(v) if len(v) > 1 else None)
        agg[name] = {"n": len(rows),
                     **{k: statistics.mean(r[k] for r in rows) for k in ("ua", "ra", "ta", "mia")},
                     **{f"gap_{k}": statistics.mean(r["gap"][k] for r in rows) for k in ("ua", "ra", "ta", "mia", "avg")},
                     "gap_avg_sd": sd([r["gap"]["avg"] for r in rows])}
    rr = [r["retrain"] for r in results["runs"].values() if "retrain" in r]
    agg["retrain"] = {"n": len(rr), **{k: statistics.mean(r[k] for r in rr) for k in ("ua", "ra", "ta", "mia")}}
    # benchmark-style scoring: gap to the fixed Retrain reference means (holdout dgp.py) when given
    fixed = plan.get("fixed_reference")
    if fixed:
        for name in names:
            rows = [r[name] for r in results["runs"].values() if name in r]
            g = [sum(abs(r[k] - fixed[k]) for k in ("ua", "ra", "ta", "mia")) / 4 for r in rows]
            agg[name]["fixed_ref_gap_avg"] = statistics.mean(g)
            agg[name]["fixed_ref_gap_avg_sd"] = statistics.stdev(g) if len(g) > 1 else None
    results["aggregate"] = agg
    a.out.write_text(json.dumps(results, indent=1))
    for name, v in agg.items():
        print(name, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})


if __name__ == "__main__":
    main()
