"""CPU analysis of saved forget/target/selection distributions; no inference.

Separate distribution-level fitting residual F--Q from fixed-target mismatch
Q--S, and compare frozen Original target T0 and selection S0 with current S.
Only exact named forget/target/selection arrays are loaded. No model weights,
audit, official-test arrays, Retrain predictions, or results.json are opened.
This is a diagnostic, not a causal or additive bias decomposition.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

HERE = Path(__file__).resolve().parent
FULL_LEVELS = np.linspace(0., 1., 1001)
TAIL_LEVELS = np.linspace(0., .1, 101)
READ_MANIFEST = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_named(path, keys):
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in keys}
    previous = READ_MANIFEST.get(str(path), {}).get("arrays_read", [])
    READ_MANIFEST[str(path)] = dict(sha256=digest(path), arrays_read=sorted(set(previous).union(keys)))
    return values


def read_distribution(path, groups):
    keys = [f"{group}_{key}" for group in groups for key in ("logodds", "confidence", "labels")]
    values = read_named(path, keys)
    return {group: {key: np.asarray(values[f"{group}_{key}"])
                    for key in ("logodds", "confidence", "labels")} for group in groups}


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    return np.exp(-np.logaddexp(0., -values))


def ks(left, right):
    left, right = np.sort(np.asarray(left, dtype=np.float64)), np.sort(np.asarray(right, dtype=np.float64))
    if len(left) == 0 or len(right) == 0:
        raise ValueError("empty empirical distribution")
    positions = np.sort(np.concatenate((left, right)))
    return float(np.max(np.abs(np.searchsorted(left, positions, side="right") / len(left)
                               - np.searchsorted(right, positions, side="right") / len(right))))


def quantile_rmse(left, right, levels):
    return float(np.sqrt(np.mean((np.quantile(left, levels) - np.quantile(right, levels)) ** 2)))


def pair_metrics(left, right):
    ls, rs = np.asarray(left["logodds"], dtype=np.float64), np.asarray(right["logodds"], dtype=np.float64)
    lc, rc = np.asarray(left["confidence"], dtype=np.float64), np.asarray(right["confidence"], dtype=np.float64)
    return dict(confidence_ks=ks(lc, rc), logodds_ks=ks(ls, rs),
        logodds_quantile_rmse=quantile_rmse(ls, rs, FULL_LEVELS),
        lower10pct_logodds_quantile_rmse=quantile_rmse(ls, rs, TAIL_LEVELS),
        logodds_median_shift=float(np.median(ls) - np.median(rs)),
        logodds_lower05_shift=float(np.quantile(ls, .05) - np.quantile(rs, .05)),
        confidence_below_05_gap_pp=float(100 * (np.mean(lc < .5) - np.mean(rc < .5))),
        confidence_below_09_gap_pp=float(100 * (np.mean(lc < .9) - np.mean(rc < .9))),
        confidence_below_099_gap_pp=float(100 * (np.mean(lc < .99) - np.mean(rc < .99))))


def reconstruct_quantiles(source, labels, reference, reference_labels):
    """Frozen algorithm: exact empirical quantile-bin average, stable source rank."""
    target = np.empty(len(source), dtype=np.float64)
    for cls in np.unique(labels):
        ids = np.flatnonzero(labels == cls)
        ids = ids[np.argsort(source[ids], kind="stable")]
        ref = np.sort(np.asarray(reference[reference_labels == cls], dtype=np.float64))
        n, m = len(ids), len(ref)
        if not m:
            raise ValueError(f"missing target class {cls}")
        left = np.maximum(np.arange(n)[:, None] / n, np.arange(m)[None, :] / m)
        right = np.minimum((np.arange(n)[:, None] + 1) / n, (np.arange(m)[None, :] + 1) / m)
        target[ids] = (n * np.maximum(right - left, 0)) @ ref
    # The frozen algorithm computes in double then stores in the source dtype.
    return target.astype(source.dtype)


def subset(distribution, cls):
    mask = np.ones(len(distribution["labels"]), dtype=bool) if cls == "all" else distribution["labels"] == cls
    return {key: value[mask] for key, value in distribution.items()}


def run_probe(run, schedule):
    seed = run / "seed42_split42"
    config_path = seed / "preregister.json"
    if not config_path.exists():
        return dict(run=str(run), status="missing", reason="no preregistration")
    config = json.loads(config_path.read_text())["configuration"]
    READ_MANIFEST[str(config_path)] = dict(sha256=digest(config_path), json_fields_read=["configuration"])
    files = sorted((seed / schedule).glob("epoch*_distributions.npz"))
    if not files:
        return dict(run=str(run), status="missing", reason="no saved epoch distribution NPZ", configuration=config)
    initial_path = seed / schedule / "epoch000_initial_distributions.npz"
    if not initial_path.exists():
        return dict(run=str(run), status="missing", reason="no initial Original distribution NPZ", configuration=config)
    original = read_distribution(initial_path, ("forget", "target", "selection"))
    q_path = seed / "calibration_targets.npz"
    if q_path.exists():
        q_scores = read_named(q_path, ["logodds"])["logodds"]
        target_provenance = dict(kind="saved_training_target", path=str(q_path), sha256=digest(q_path))
    else:
        if config.get("target_kind") != "global_logodds":
            return dict(run=str(run), status="missing", reason="non-global target cache missing; cannot reconstruct")
        source_path = HERE / "fixed_quantile.py"
        if digest(source_path) != config["fixed_target_source_sha256"]:
            raise ValueError("frozen target algorithm source hash differs from preregistration")
        q_scores = reconstruct_quantiles(original["forget"]["logodds"], original["forget"]["labels"],
            original["target"]["logodds"], original["target"]["labels"])
        target_provenance = dict(kind="deterministically_reconstructed_not_saved_target",
            original_cache=str(initial_path), frozen_source=str(source_path),
            frozen_source_sha256=digest(source_path), no_model_inference=True)
    if len(q_scores) != len(original["forget"]["labels"]):
        raise ValueError("training-target and forget lengths differ")
    q = dict(logodds=q_scores, confidence=sigmoid(q_scores).astype(np.float32), labels=original["forget"]["labels"])
    if q_path.exists() and config.get("target_kind") == "global_logodds":
        reconstructed = reconstruct_quantiles(original["forget"]["logodds"], original["forget"]["labels"],
            original["target"]["logodds"], original["target"]["labels"])
        if not np.allclose(q_scores, reconstructed, atol=2e-5, rtol=2e-6):
            raise ValueError(f"saved target incompatible with frozen reconstruction: {run}")
        target_provenance["reconstruction_max_absolute_difference"] = float(np.max(abs(q_scores - reconstructed)))
    best_path = seed / schedule / "best.json"
    best_epoch = json.loads(best_path.read_text())["epoch"] if best_path.exists() else None
    if best_path.exists():
        READ_MANIFEST[str(best_path)] = dict(sha256=digest(best_path), json_fields_read=["epoch"])
    rows = []
    for path in files:
        match = re.fullmatch(r"epoch(\d+)_(.+)_distributions\.npz", path.name)
        if match is None:
            continue
        epoch, stage = int(match.group(1)), match.group(2)
        current = read_distribution(path, ("forget", "selection"))
        for cls in ("all", *range(10)):
            groups = {"F": subset(current["forget"], cls), "Q": subset(q, cls),
                      "S": subset(current["selection"], cls), "T0": subset(original["target"], cls),
                      "S0": subset(original["selection"], cls)}
            for pair in ("F_Q", "Q_S", "F_S", "T0_S", "S0_S", "Q_T0", "T0_S0"):
                left, right = pair.split("_")
                rows.append(dict(epoch=epoch, stage=stage, cls=cls, pair=pair,
                    n_left=len(groups[left]["labels"]), n_right=len(groups[right]["labels"]),
                    **pair_metrics(groups[left], groups[right])))
            f, t = (np.asarray(groups[name]["logodds"], dtype=np.float64) for name in ("F", "Q"))
            rows.append(dict(epoch=epoch, stage=stage, cls=cls, pair="F_Q_pointwise",
                n_left=len(f), n_right=len(t), pointwise_logodds_rmse=float(np.sqrt(np.mean((f - t) ** 2)))))
    return dict(run=str(run), status="ok", configuration=config, target_provenance=target_provenance,
        selected_epoch_from_existing_rule=best_epoch,
        epochs_present=sorted({row["epoch"] for row in rows}),
        missing_epochs=sorted(set(range(config["epochs"] + 1)) - {row["epoch"] for row in rows}),
        confidence_note="saved FP32 softmax confidence for F/S/T0/S0; Q from FP32 sigmoid(logodds)",
        rows=rows)


def self_test():
    assert np.isclose(ks([0., 0., 1.], [0., 1., 1.]), 1 / 3)
    assert quantile_rmse([0., 1.], [0., 1.], FULL_LEVELS) == 0
    source = np.array([3., 1., 2.], dtype=np.float32)
    labels = np.zeros(3, dtype=np.int64)
    out = reconstruct_quantiles(source, labels, source, labels)
    assert np.array_equal(source, out)
    small = reconstruct_quantiles(np.array([2., 1.]), np.zeros(2, dtype=int),
                                   np.array([0., 3., 6.]), np.zeros(3, dtype=int))
    assert np.allclose(small, [5., 1.])
    print("target-bias CPU checks passed: exact KS/ties, quantile identity, unequal-size reconstruction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--runs", nargs="+", default=["r1_C_frozen_lr003_a01", "r2_C_rank_kl_lr003_a1",
        "r2_C_rank_kl_lr003_a10", "r3_layer4_fc_C_lr003_a01", "r3_layer4_fc_C_lr01_a01"])
    parser.add_argument("--schedule", default="C")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.root is None:
        if args.self_test:
            return
        parser.error("--root is required")
    if args.output_dir is None:
        parser.error("--output-dir is required")
    result = dict(schema=1, source_sha256=digest(Path(__file__)), device="CPU/NumPy; no inference",
        metric_definition=dict(confidence_ks="exact two-sample KS on saved FP32 confidence",
            logodds_quantile_rmse="RMSE of numpy linear quantiles at 1001 uniformly spaced levels [0,1]",
            lower10pct_logodds_quantile_rmse="RMSE of linear quantiles at 101 levels [0,.1]",
            interpretation="pairwise distances; not additive/causal decomposition and not certification"),
        runs=[run_probe(args.root / name, args.schedule) for name in args.runs],
        input_manifest=READ_MANIFEST)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "target_bias.json").write_text(json.dumps(result, indent=2) + "\n")
    flat = [{"run": Path(run["run"]).name, **row} for run in result["runs"] if run["status"] == "ok" for row in run["rows"]]
    fields = sorted({key for row in flat for key in row})
    with (args.output_dir / "target_bias.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    for run in result["runs"]:
        print(json.dumps({key: run.get(key) for key in ("run", "status", "reason", "epochs_present", "selected_epoch_from_existing_rule", "target_provenance")}))


if __name__ == "__main__":
    main()
