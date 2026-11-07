"""One-way evaluation of a frozen, selection-only candidate manifest.

Freeze resolves artifact paths and SHA256 values without loading a model.
Audit reads only verified training arrays and never loads a Retrain model.
Test requires the completed audit of the identical frozen manifest, then
reports paired-Retrain metrics using the benchmark's original evaluator.
Neither phase selects a checkpoint, changes a manifest, or overwrites outputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TASK = ROOT / "tasks/security-machine-unlearning/scripts"
METRICS = ("ua", "ra", "ta", "mia")
TIMING_KEYS = ("training_loop_seconds", "target_generation_seconds",
               "diagnostic_seconds", "total_seconds")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    """Exclusive creation: an existing audit, test, or frozen file is immutable."""
    path = Path(path)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def artifact(value, relative_to, verify=True):
    item = dict(path=value) if isinstance(value, str) else dict(value)
    path = Path(item["path"]).expanduser()
    path = (relative_to / path if not path.is_absolute() else path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"artifact is not a regular file: {path}")
    actual = sha256(path) if verify else None
    expected = item.get("sha256")
    if expected is not None and (not isinstance(expected, str)
                                 or not re.fullmatch(r"[0-9a-f]{64}", expected)):
        raise ValueError(f"invalid SHA256: {path}")
    if verify and expected is not None and actual != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    if not verify and expected is None:
        raise ValueError(f"frozen artifact has no SHA256: {path}")
    return dict(path=str(path), sha256=actual if verify else expected)


def source_artifacts():
    return {name: dict(path=str(path), sha256=sha256(path)) for name, path in {
        "evaluate_selected": Path(__file__).resolve(),
        "common": TASK / "common.py", "benchmark_evaluate": TASK / "evaluate.py",
    }.items()}


def freeze(source, destination):
    source, destination = source.resolve(strict=True), destination.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to replace frozen manifest: {destination}")
    manifest = json.loads(source.read_text())
    if manifest.get("frozen"):
        raise ValueError("input is already frozen")
    entries = manifest.get("candidates", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest needs a nonempty candidates list")
    seen, resolved = set(), []
    for entry in entries:
        entry = dict(entry)
        ident = entry.get("id", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", ident) or ident in seen:
            raise ValueError(f"invalid or duplicate candidate id: {ident!r}")
        seen.add(ident)
        if not isinstance(entry.get("method"), str) or not entry["method"]:
            raise ValueError(f"candidate {ident}: method is required")
        for key in ("seed", "split_seed", "epoch"):
            if type(entry.get(key)) is not int or (key == "epoch" and entry[key] < 0):
                raise ValueError(f"candidate {ident}: {key} must be an integer")
        if not isinstance(entry.get("config"), dict):
            raise ValueError(f"candidate {ident}: config must be an object")
        timing = entry.get("timing", {})
        if not isinstance(timing, dict):
            raise ValueError(f"candidate {ident}: timing must be an object")
        timing = {**dict.fromkeys(TIMING_KEYS), **timing}
        for key, value in timing.items():
            if value is not None and (type(value) not in (float, int)
                                      or not math.isfinite(value) or value < 0):
                raise ValueError(f"candidate {ident}: invalid timing value {key}")
        entry["timing"] = timing
        for key in ("checkpoint", "split_manifest", "selection_audit_split"):
            entry[key] = artifact(entry[key], source.parent)
        for key in ("training_metadata", "selection_metadata", "preregistration"):
            if entry.get(key) is not None:
                entry[key] = artifact(entry[key], source.parent)
        if entry.get("paired_retrain") is not None:
            entry["paired_retrain"] = artifact(entry["paired_retrain"], source.parent)
        if entry.get("paired_retrain_metadata") is not None:
            entry["paired_retrain_metadata"] = artifact(entry["paired_retrain_metadata"], source.parent)
        resolved.append(entry)
    frozen = dict(schema=1, frozen=True, created_utc=now(),
                  source_manifest=dict(path=str(source), sha256=sha256(source)),
                  sources=source_artifacts(), candidates=resolved,
                  policy="selection was completed before freeze; audit cannot reselect; test follows audit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_new(destination, frozen)
    print(json.dumps(dict(event="manifest_frozen", path=str(destination), sha256=sha256(destination),
                          n_candidates=len(resolved))), flush=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime(args):
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.gpu == "cpu" else args.gpu
    global np, torch, common
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    common = load_module("selected_eval_common", TASK / "common.py")
    common.set_determinism(0)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_manifest(path, phase):
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != 1 or manifest.get("frozen") is not True:
        raise ValueError("evaluation requires a --freeze output")
    current = source_artifacts()
    for name, item in manifest["sources"].items():
        if name not in current or current[name]["sha256"] != item["sha256"]:
            raise ValueError(f"evaluation source changed after freeze: {name}")
    for entry in manifest["candidates"]:
        for key in ("checkpoint", "split_manifest", "selection_audit_split"):
            artifact(entry[key], path.parent)
        for key in ("training_metadata", "selection_metadata", "preregistration"):
            if entry.get(key) is not None:
                artifact(entry[key], path.parent)
        # Audit deliberately does not open the Retrain checkpoint.
        if phase == "test":
            if not entry.get("paired_retrain"):
                raise ValueError(f"candidate {entry['id']} has no frozen paired_retrain")
            artifact(entry["paired_retrain"], path.parent)
            if entry.get("paired_retrain_metadata") is not None:
                item = artifact(entry["paired_retrain_metadata"], path.parent)
                metadata = json.loads(Path(item["path"]).read_text())
                if (any(metadata.get(key) != entry[key] for key in ("seed", "split_seed"))
                        or metadata.get("epochs") != 182
                        or metadata.get("test_used_during_training") is not False):
                    raise ValueError(f"candidate {entry['id']}: paired Retrain training provenance differs")
    return manifest


def load_arrays(data_dir, full):
    # np.load is lazy; audit never accesses either official-test member.
    with np.load(data_dir / "cifar10.npz", allow_pickle=False) as archive:
        x, y = archive["x_train"], archive["y_train"].astype(np.int64)
        arrays = (x, y)
        if full:
            if "x_test" in archive.files:
                arrays += (archive["x_test"], archive["y_test"].astype(np.int64))
            else:
                with np.load(data_dir / "withheld/cifar10_test.npz", allow_pickle=False) as test:
                    arrays += (test["x_test"], test["y_test"].astype(np.int64))
    expected = (common.SHA256_DATA if full else common.SHA256_TRAIN)["cifar10"]
    actual = common.sha256_arrays(*arrays)
    if actual != expected:
        raise ValueError("dataset ordering/content does not match the pinned SHA256")
    return arrays, actual


def validate_splits(entry, labels):
    """Reconstruct every recorded split, including the fixed 5000 retain subset."""
    saved = json.loads(Path(entry["split_manifest"]["path"]).read_text())
    if any(saved.get(k) != entry[k] for k in ("seed", "split_seed")):
        raise ValueError(f"candidate {entry['id']}: seed/split manifest mismatch")
    train, val, forget, retain = common.split_indices(labels, 10, entry["split_seed"], .1)
    rng, target, calibration = np.random.RandomState(20261106), [], []
    for cls in range(10):
        order = rng.permutation(val[labels[val] == cls])
        target.extend(order[:len(order) // 2])
        calibration.extend(order[len(order) // 2:])
    target, calibration = np.sort(target), np.sort(calibration)
    expected = dict(train=train, forget=forget, retain=retain, target=target, calibration=calibration)
    if saved.get("validation_seed") != common.SPLIT_SEED or saved.get("target_split_seed") != 20261106:
        raise ValueError("validation/target split seed mismatch")
    if set(saved.get("indices", {})) != set(expected):
        raise ValueError("split index names changed")
    for key, indices in expected.items():
        if not np.array_equal(saved["indices"][key], indices):
            raise ValueError(f"candidate {entry['id']}: {key} indices differ from the declared split")
    rng, selection, audit = np.random.RandomState(20261107), [], []
    for cls in range(10):
        order = rng.permutation(calibration[labels[calibration] == cls])
        selection.extend(order[:125])
        audit.extend(order[125:])
    selection, audit = np.sort(selection), np.sort(audit)
    saved_audit = json.loads(Path(entry["selection_audit_split"]["path"]).read_text())
    if (saved_audit.get("seed") != 20261107 or saved_audit.get("audit_predicted") is not False
            or not np.array_equal(saved_audit.get("selection"), selection)
            or not np.array_equal(saved_audit.get("audit"), audit)
            or len(selection) != 1250 or len(audit) != 1250):
        raise ValueError("selection/audit manifest differs from the declared 1250/1250 split")
    positions = np.sort(np.random.RandomState(7312026).choice(len(retain), 5000, replace=False))
    return dict(forget=forget, retain_subset=retain[positions], audit=audit, selection=selection)


def load_state(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    reference = common.build_model("cifar10").state_dict()
    if not isinstance(state, dict) or set(state) != set(reference):
        raise ValueError(f"checkpoint has incorrect state_dict keys: {path}")
    for name, tensor in state.items():
        expected = reference[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != expected.shape
                or tensor.dtype != expected.dtype or not torch.isfinite(tensor).all()):
            raise ValueError(f"checkpoint has invalid tensor {name}: {path}")
    return state


def confidence_ks(x, y):
    x, y = np.sort(np.asarray(x, dtype=np.float64)), np.sort(np.asarray(y, dtype=np.float64))
    if not len(x) or not len(y):
        raise ValueError("KS requires nonempty samples")
    points = np.sort(np.concatenate((x, y)))
    return float(np.max(np.abs(np.searchsorted(x, points, side="right") / len(x)
                               - np.searchsorted(y, points, side="right") / len(y))))


def predict(state, arrays, indices, device):
    model = common.build_model("cifar10").to(device).eval()
    model.load_state_dict(state, strict=True)
    x, y = arrays[:2]
    result = {}
    with torch.inference_mode():
        for name, ids in indices.items():
            confidence, error = [], []
            for start in range(0, len(ids), 256):
                positions = ids[start:start + 256]
                images = common.to_gpu_u8(x[positions], device).float().div_(255.)
                labels = torch.from_numpy(y[positions]).to(device)
                logits = model(images)
                probs = logits.softmax(1)
                if not torch.isfinite(probs).all():
                    raise ValueError("nonfinite prediction")
                confidence.append(probs.gather(1, labels[:, None]).squeeze(1).double().cpu().numpy())
                error.append(logits.argmax(1).ne(labels).cpu().numpy())
            result[name] = dict(confidence=np.concatenate(confidence), error=np.concatenate(error),
                                labels=y[ids], indices=ids)
    del model
    return result


def summarize(distributions):
    return {name: dict(n=len(values["error"]), error_pct=float(100 * values["error"].mean()),
                       accuracy_pct=float(100 * (1 - values["error"].mean())),
                       confidence_mean=float(values["confidence"].mean()))
            for name, values in distributions.items()}


def audit_metrics(candidate, original):
    summary, baseline = summarize(candidate), summarize(original)
    ff, aa = candidate["forget"], candidate["audit"]
    gap = summary["forget"]["error_pct"] - summary["audit"]["error_pct"]
    ks = confidence_ks(ff["confidence"], aa["confidence"])
    classwise = {str(cls): confidence_ks(ff["confidence"][ff["labels"] == cls],
                                       aa["confidence"][aa["labels"] == cls]) for cls in range(10)}
    return dict(audit_J=max(abs(gap) / 100., ks), forget_audit_error_gap_pp=gap,
                absolute_forget_audit_error_gap_pp=abs(gap), forget_audit_confidence_ks=ks,
                forget_original_audit_confidence_ks=confidence_ks(ff["confidence"], original["audit"]["confidence"]),
                classwise_confidence_ks=classwise,
                mean_classwise_confidence_ks=statistics.mean(classwise.values()),
                worst_classwise_confidence_ks=max(classwise.values()),
                retain_subset_accuracy_drift_pp=summary["retain_subset"]["accuracy_pct"] - baseline["retain_subset"]["accuracy_pct"],
                audit_accuracy_drift_pp=summary["audit"]["accuracy_pct"] - baseline["audit"]["accuracy_pct"],
                candidate_summary=summary, original_summary=baseline,
                interpretation="independent audit only; no re-selection or claim that J bounds paired-Retrain AvgGap")


def aggregate(records):
    groups = {}
    for record in records:
        entry = record["candidate"]
        config_hash = hashlib.sha256(json.dumps(entry["config"], sort_keys=True).encode()).hexdigest()
        key = (entry["method"], config_hash)
        groups.setdefault(key, []).append(record)
    summaries = []
    for (method, config_hash), rows in groups.items():
        def moments(samples):
            return dict(n=len(samples), mean=statistics.mean(samples),
                        sample_sd=statistics.stdev(samples) if len(samples) > 1 else None)
        values = {section: {name: moments([row[section][name] for row in rows]) for name in METRICS}
                  for section in ("candidate_metrics", "paired_retrain_metrics", "absolute_gaps")}
        values["avg_gap"] = moments([row["avg_gap"] for row in rows])
        values["timing"] = {name: (moments(samples) if samples else None)
                            for name in TIMING_KEYS
                            for samples in [[row["candidate"]["timing"][name] for row in rows
                                             if row["candidate"]["timing"][name] is not None]]}
        summaries.append(dict(method=method, config=rows[0]["candidate"]["config"], config_sha256=config_hash,
                              n=len(rows), seeds=[row["candidate"]["seed"] for row in rows],
                              split_seeds=[row["candidate"]["split_seed"] for row in rows], metrics=values))
    return summaries


def evaluate(args):
    path = args.candidate_manifest.resolve(strict=True)
    manifest_sha = sha256(path)
    manifest = read_manifest(path, args.phase)
    out = args.out_dir.resolve()
    phase_dir = out / args.phase
    if phase_dir.exists():
        raise FileExistsError(f"refusing to overwrite or repeat {args.phase}: {phase_dir}")
    if args.phase == "test":
        audit_path = out / "audit/receipt.json"
        audit = json.loads(audit_path.read_text())
        if not audit.get("complete") or audit.get("manifest_sha256") != manifest_sha:
            raise ValueError("test needs a completed audit of the identical frozen manifest")
        if audit.get("candidate_ids") != [entry["id"] for entry in manifest["candidates"]]:
            raise ValueError("audited candidates differ")
        for item in audit["outputs"]:
            artifact(item, out / "audit")
    device = runtime(args)
    arrays, dataset_sha = load_arrays(args.data_dir, args.phase == "test")
    if sha256(args.original) != common.SHA256_ORIGINAL["cifar10"]:
        raise ValueError("Original checkpoint SHA256 mismatch")
    original = load_state(args.original)
    splits = [validate_splits(entry, arrays[1]) for entry in manifest["candidates"]]
    out.mkdir(parents=True, exist_ok=True)
    phase_dir.mkdir()  # exclusive phase reservation, including incomplete runs
    write_new(phase_dir / "started.json", dict(manifest_sha256=manifest_sha, started_utc=now(),
               phase=args.phase, policy="one-way evaluation; incomplete outputs are not overwritten"))
    records, outputs = [], []
    provenance = dict(manifest_path=str(path), manifest_sha256=manifest_sha,
                      dataset_sha256=dataset_sha, original=dict(path=str(args.original.resolve()), sha256=sha256(args.original)),
                      sources=source_artifacts(), torch_version=torch.__version__, numpy_version=np.__version__,
                      sklearn_version=importlib.metadata.version("scikit-learn"), device=str(device), threads=args.threads)
    evaluator = load_module("selected_benchmark_evaluator", TASK / "evaluate.py") if args.phase == "test" else None
    for entry, indices in zip(manifest["candidates"], splits):
        start = time.monotonic()
        state = load_state(entry["checkpoint"]["path"])
        cosine = common.descent_cosine(state, original)
        if cosine < common.MIN_DESCENT_COSINE:
            raise ValueError(f"candidate {entry['id']} is not an update of Original")
        if args.phase == "audit":
            if not args.include_selection:
                indices = {name: values for name, values in indices.items() if name != "selection"}
            before, after = predict(original, arrays, indices, device), predict(state, arrays, indices, device)
            record = dict(candidate=entry, phase="audit", metrics=audit_metrics(after, before),
                          selected_checkpoint_is_final=True, reselected=False)
            distribution_path = phase_dir / f"{entry['id']}_distributions.npz"
            with distribution_path.open("xb") as stream:
                np.savez_compressed(stream, **{f"{prefix}_{name}_{key}": value
                    for prefix, distributions in (("original", before), ("candidate", after))
                    for name, values in distributions.items() for key, value in values.items()})
            distribution_path.chmod(0o444)
            outputs.append(dict(path=distribution_path.name, sha256=sha256(distribution_path)))
        else:
            retrain = load_state(entry["paired_retrain"]["path"])
            candidate_metrics = evaluator.evaluate_state("cifar10", state, indices["forget"], arrays, .1)
            retrain_metrics = evaluator.evaluate_state("cifar10", retrain, indices["forget"], arrays, .1)
            gaps = {name: abs(candidate_metrics[name] - retrain_metrics[name]) for name in METRICS}
            record = dict(candidate=entry, phase="test", candidate_metrics=candidate_metrics,
                          paired_retrain_metrics=retrain_metrics, absolute_gaps=gaps,
                          avg_gap=statistics.mean(gaps.values()))
        record.update(descent_cosine=cosine, evaluation_seconds=time.monotonic() - start,
                      created_utc=now(), provenance=provenance)
        target = phase_dir / f"{entry['id']}.json"
        write_new(target, record)
        outputs.append(dict(path=target.name, sha256=sha256(target)))
        records.append(record)
        print(json.dumps(dict(event=f"{args.phase}_complete", candidate=entry["id"],
                              **({"avg_gap": record["avg_gap"]} if args.phase == "test" else {"audit_J": record["metrics"]["audit_J"]}))), flush=True)
    if args.phase == "test":
        target = phase_dir / "summary.json"
        write_new(target, dict(groups=aggregate(records), provenance=provenance,
                               note="per-run absolute gaps precede aggregation; sample SD is undefined for n=1"))
        outputs.append(dict(path=target.name, sha256=sha256(target)))
    # The manifest and all referenced artifacts are rechecked after predictions.
    if sha256(path) != manifest_sha:
        raise ValueError("frozen manifest changed during evaluation")
    read_manifest(path, args.phase)
    write_new(phase_dir / "receipt.json", dict(complete=True, phase=args.phase, completed_utc=now(),
              manifest_sha256=manifest_sha, candidate_ids=[entry["id"] for entry in manifest["candidates"]],
              outputs=outputs, provenance=provenance, reselected=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, help="write an immutable resolved manifest, without model evaluation")
    parser.add_argument("--phase", choices=("audit", "test"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--gpu", help="physical GPU id, inherited visibility when omitted, or cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--include-selection", action="store_true", help="also report selection diagnostics; never reselect")
    args = parser.parse_args()
    if args.freeze is not None:
        if args.phase is not None:
            parser.error("--freeze and --phase are separate invocations")
        freeze(args.candidate_manifest, args.freeze)
    else:
        if not args.phase or any(getattr(args, name) is None for name in ("data_dir", "original", "out_dir")):
            parser.error("--phase needs --data-dir, --original and --out-dir")
        if args.threads < 1:
            parser.error("--threads must be positive")
        evaluate(args)


if __name__ == "__main__":
    main()
