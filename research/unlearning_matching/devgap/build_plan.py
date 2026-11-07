"""Build the final_eval plan for one frozen confirmation tag.

Runs: s43..s49 (fresh deletion splits, canonical Original), o142..o542
(independent Originals, split 42), and optionally s42 (development split,
reported separately).  Candidates per run: our frozen last.pt, SalUn and
Original from the same split, compared against that split's paired Retrain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

W = Path("/mnt/datablaze2/bohanlyu/unlearning-20261106")
S = W / "snapshot/out"
CANON = "/srv/home/bohanlyu/data_mlsbench/unlearn-lab/original/cifar10_resnet18_original.pt"
FIXED = {"ua": 5.4222, "ra": 99.9994, "ta": 94.29, "mia": 12.7028}  # holdout/security-machine-unlearning/dgp.py


def refs(name):
    if name == "s42":
        d = S / "eval-only/canonical/seed42_split42"
    elif name in ("s43", "s44"):
        d = S / f"eval-only/canonical/seed{name[1:]}_split{name[1:]}"
    elif name.startswith("s"):
        d = W / f"fresh_splits/out/seed{name[1:]}_split{name[1:]}"
    else:
        d = S / f"final_replicates/original_seed{name[1:]}/baselines/seed{name[1:]}_split42"
    orig = CANON if name.startswith("s") else str(S / f"independent_originals/seed{name[1:]}/original.pt")
    return d / "splits.json", d / "retrain.pt", d / "salun.pt", orig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--runs", default="s43,s44,s45,s46,s47,s48,s49,o142,o242,o342,o442,o542")
    p.add_argument("--checkpoint", default="last.pt")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    runs = []
    for name in a.runs.split(","):
        splits, retrain, salun, orig = refs(name)
        ours = W / f"devgap/confirm/{a.tag}/{name}/{a.checkpoint}"
        missing = [str(x) for x in (splits, retrain, salun, ours) if not Path(x).exists()]
        if missing:
            print("skip", name, "missing", missing)
            continue
        runs.append({"id": name, "splits": str(splits), "retrain": str(retrain),
                     "candidates": {"ours": str(ours), "salun": str(salun), "original": orig}})
    a.out.write_text(json.dumps({"tag": a.tag, "fixed_reference": FIXED, "runs": runs}, indent=1))
    print(len(runs), "runs ->", a.out)


if __name__ == "__main__":
    main()
