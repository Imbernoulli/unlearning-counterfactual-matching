"""Aggregate paired unlearning results without selecting or changing methods."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ORDER = ("original", "ga", "ft", "l1_sparse", "rl", "salun",
         "regun_mean_control", "w2_logodds", "w2_logodds_margin", "retrain")
LABELS = {"original": "Original", "ga": "Gradient ascent", "ft": "Fine-tuning",
          "l1_sparse": "L1-sparse FT", "rl": "Random labelling", "salun": "SalUn",
          "regun_mean_control": "Mean-target control", "w2_logodds": "Log-odds transport",
          "w2_logodds_margin": "Log-odds + margin", "retrain": "Paired Retrain",
          "global_quantile_iprojection": "Global quantile + I-projection",
          "global_quantile_logodds_mse": "Global quantile + log-odds MSE"}
FOLLOWUP = ("global_quantile_iprojection", "global_quantile_logodds_mse")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--followup-dir", type=Path)
    args = parser.parse_args()
    files = sorted(args.runs_dir.glob("seed*_split*/results.json"))
    if not files:
        raise SystemExit("No completed paired results found")
    runs, evidence = [], []
    for path in files:
        run = json.loads(path.read_text())
        if run.get("reference_epochs") != 182 or set(run["methods"]) != set(ORDER):
            raise ValueError(f"Incomplete or incompatible experiment: {path}")
        config = json.loads((path.parent / "run_config.json").read_text())
        evidence.append(dict(path=str(path.resolve()),
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest(), config=config))
        runs.append(run)
    if args.followup_dir:
        for run, source in zip(runs, files):
            path = args.followup_dir / source.parent.name / "results.json"
            followup = json.loads(path.read_text())
            if (followup["seed"], followup["split_seed"]) != (run["seed"], run["split_seed"]):
                raise ValueError("Follow-up has a different deletion split identity")
            if set(followup["methods"]) != {"retrain", *FOLLOWUP}:
                raise ValueError("Missing preregistered follow-up methods")
            for key in ("ua", "ra", "ta", "mia"):
                if abs(followup["methods"]["retrain"][key] - run["methods"]["retrain"][key]) > 1e-3:
                    raise ValueError(f"Recomputed paired reference differs on {key}")
            for name in FOLLOWUP:
                run["methods"][name] = followup["methods"][name]
            evidence[runs.index(run)]["followup"] = dict(
                path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                preregistration=json.loads((path.parent / "preregister.json").read_text()))
    for key in ("code_sha256", "candidate_epochs", "candidate_lr", "forget_weight",
                "anchor_weight", "reference_amp", "torch_version", "numpy_version",
                "sklearn_version", "planned_methods"):
        if len({json.dumps(r["config"][key], sort_keys=True) for r in evidence}) != 1:
            raise ValueError(f"Runs differ in {key}")
    if len({(r["seed"], r["split_seed"]) for r in runs}) != len(runs):
        raise ValueError("Duplicate experiment identities")
    rows = []
    order = (*ORDER[:-1], *FOLLOWUP, "retrain") if args.followup_dir else ORDER
    for name in order:
        values = [r["methods"][name] for r in runs]
        row = dict(method=name, label=LABELS[name], n=len(values))
        for key in ("ua", "ra", "ta", "mia", "avg_gap", "seconds"):
            arr = np.asarray([v[key] for v in values], dtype=float)
            if not np.isfinite(arr).all():
                raise ValueError(f"Nonfinite {name}/{key}")
            row[f"{key}_mean"] = float(arr.mean())
            row[f"{key}_sd"] = float(arr.std(ddof=1)) if len(arr) > 1 else None
        rows.append(row)
    comparisons = {}
    candidates = ("regun_mean_control", "w2_logodds", "w2_logodds_margin")
    if args.followup_dir:
        candidates += FOLLOWUP
    for candidate in candidates:
        comparisons[candidate] = {}
        for baseline in ("rl", "salun"):
            # Positive means lower gap than the baseline, preserving pairing.
            d = [r["methods"][baseline]["avg_gap"] - r["methods"][candidate]["avg_gap"] for r in runs]
            comparisons[candidate][baseline] = dict(
                baseline_gap_minus_candidate_gap=d, mean=float(np.mean(d)),
                sd=float(np.std(d, ddof=1)) if len(d) > 1 else None)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(n_runs=len(runs), seeds=[r["seed"] for r in runs],
                   note="One shared Original; paired deletion/retraining runs. Sample SD across runs. "
                        "Update-loop time excludes fixed target construction. "
                        "Empirical comparisons do not establish certified unlearning.",
                   methods=rows, paired_comparisons=comparisons, evidence=evidence)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| Method | UA | RA | TA | MIA | AvgGap | Update seconds |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        gap = f"{r['avg_gap_mean']:.3f}"
        if r["avg_gap_sd"] is not None:
            gap += f" ± {r['avg_gap_sd']:.3f}"
        lines.append(f"| {r['label']} | {r['ua_mean']:.2f} | {r['ra_mean']:.2f} | "
                     f"{r['ta_mean']:.2f} | {r['mia_mean']:.2f} | {gap} | {r['seconds_mean']:.1f} |")
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
