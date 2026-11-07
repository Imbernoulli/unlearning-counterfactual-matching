"""Build a readable final-evaluation manifest from an explicit confirmation plan.

This standard-library-only tool checks paths, hashes and training metadata. It
never discovers candidates, compares scores, loads a model, or reads prediction
archives/history/audit/test results. The caller supplies every checkpoint kind,
epoch and method. The unchanged evaluator performs the subsequent freeze.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
COMMON = HERE.parents[1] / "tasks/security-machine-unlearning/scripts/common.py"
TIMING_KEYS = ("training_loop_seconds", "target_generation_seconds", "diagnostic_seconds", "total_seconds")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON number in {path}: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def pin(value, root):
    value = dict(path=value) if isinstance(value, str) else dict(value)
    path = Path(value["path"]).expanduser()
    path = (root / path if not path.is_absolute() else path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"not a regular file: {path}")
    digest = sha256(path)
    if value.get("sha256") is not None and value["sha256"] != digest:
        raise ValueError(f"artifact SHA256 mismatch: {path}")
    return dict(path=str(path), sha256=digest)


def literal_pins():
    names, values = {"SPLIT_SEED", "SHA256_TRAIN", "SHA256_ORIGINAL"}, {}
    for node in ast.parse(COMMON.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    values[target.id] = ast.literal_eval(node.value)
    if set(values) != names:
        raise ValueError("could not read canonical data/Original pins")
    return values


def integer(value, label, minimum=None):
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"{label} must be an integer" + (f" >= {minimum}" if minimum is not None else ""))
    return value


def equal(actual, expected, label):
    # Distinguish JSON true from integer 1 when checking provenance.
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError(f"{label} mismatch: expected {expected!r}, found {actual!r}")


def parameter(config, key):
    locations = [source[key] for source in (config, config.get("arguments", {})) if key in source]
    if not locations:
        raise ValueError(f"preregistration has no parameter {key}")
    for value in locations[1:]:
        equal(value, locations[0], f"ambiguous preregistration parameter {key}")
    return locations[0]


def pointer(config, path):
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("preregister_checks keys must be JSON pointers starting with /")
    value = config
    for part in path[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def split_checks(saved, partition, seed, split_seed, pins):
    for key, expected in dict(seed=seed, split_seed=split_seed,
                              validation_seed=pins["SPLIT_SEED"], target_split_seed=20261106).items():
        equal(saved.get(key), expected, f"split {key}")
    sizes = dict(train=45000, forget=4500, retain=40500, target=2500, calibration=2500)
    indices = saved.get("indices", {})
    equal(sorted(indices), sorted(sizes), "split index names")
    sets = {}
    for key, size in sizes.items():
        values = indices[key]
        if (not isinstance(values, list) or len(values) != size
                or any(type(v) is not int or v < 0 or v >= 50000 for v in values)
                or len(set(values)) != size):
            raise ValueError(f"invalid {key} membership")
        sets[key] = set(values)
    if (sets["forget"] & sets["retain"] or sets["train"] != sets["forget"] | sets["retain"]
            or sets["target"] & sets["calibration"]
            or sets["train"] & (sets["target"] | sets["calibration"])
            or len(sets["train"] | sets["target"] | sets["calibration"]) != 50000):
        raise ValueError("training/forget/retain/target/calibration are not the declared partitions")
    equal(partition.get("seed"), 20261107, "selection/audit seed")
    equal(partition.get("audit_predicted"), False, "audit prediction prohibition")
    parts = []
    for name in ("selection", "audit"):
        values = partition.get(name)
        if (not isinstance(values, list) or len(values) != 1250
                or any(type(v) is not int for v in values) or len(set(values)) != 1250):
            raise ValueError(f"invalid {name} membership")
        parts.append(set(values))
    if parts[0] & parts[1] or parts[0] | parts[1] != sets["calibration"]:
        raise ValueError("selection/audit do not partition calibration")


def seconds(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {label}")
    return value


def original_check(item, root, config, pins):
    if item.get("original_manifest") is None:
        equal(config.get("original_sha256"), pins["SHA256_ORIGINAL"]["cifar10"], "canonical Original")
        return None
    # The existing wrapper verifies the 182-epoch, train-only Original source,
    # membership/data pins and checkpoint without importing Torch.
    spec = importlib.util.spec_from_file_location("confirmation_independent_verifier", HERE / "evaluate_independent.py")
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    reference = pin(item["original_manifest"], root)
    _, manifest, _, _ = wrapper.verify_original(Path(reference["path"]))
    equal(manifest["seed"], item["seed"], "independent Original training seed")
    equal(config.get("original_sha256"), manifest["original_sha256"], "independent Original")
    return reference


def candidate(item, group, root, pins):
    ident = item["id"]
    if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", ident):
        raise ValueError(f"invalid candidate id {ident!r}")
    method = group["method"]
    if not isinstance(method, str) or not method:
        raise ValueError("group method must be a nonempty string")
    if not isinstance(group.get("config", {}), dict):
        raise ValueError("group config must be an object")
    kind = item["checkpoint_kind"]
    if kind not in ("best", "last"):
        raise ValueError("checkpoint_kind must explicitly be best or last")
    seed, split_seed, epoch, horizon = [integer(item[key], key, 0 if key in ("epoch", "preregistered_epochs") else None)
                                       for key in ("seed", "split_seed", "epoch", "preregistered_epochs")]
    directory = Path(item["schedule_dir"]).expanduser()
    directory = (root / directory if not directory.is_absolute() else directory).resolve(strict=True)
    schedule = directory.name
    if schedule not in ("A", "B", "C", "D"):
        raise ValueError("schedule_dir must name the actual A/B/C/D schedule directory")
    run = directory.parent
    preregistration = pin(str(run / "preregister.json"), root)
    config = load_json(preregistration["path"])["configuration"]
    for key, expected in dict(seed=seed, split_seed=split_seed, epochs=horizon).items():
        equal(parameter(config, key), expected, f"preregistered {key}")
    if epoch > horizon or (kind == "last" and epoch != horizon):
        raise ValueError("last must complete the registered horizon; best cannot exceed it")
    if schedule not in parameter(config, "schedules"):
        raise ValueError("schedule was not preregistered")
    equal(config.get("train_sha256"), pins["SHA256_TRAIN"]["cifar10"], "training data SHA256")
    equal(config.get("audit_predictions_computed"), False, "preregistered audit prohibition")
    checks = group["preregister_checks"]
    if not isinstance(checks, dict) or not checks:
        raise ValueError("group needs explicit nonempty preregister_checks")
    for path, expected in checks.items():
        equal(pointer(config, path), expected, f"preregistered {path}")
    original = original_check(item, root, config, pins)
    split = pin(str(run / "splits.json"), root)
    partition = pin(str(run / "selection_audit_split.json"), root)
    split_checks(load_json(split["path"]), load_json(partition["path"]), seed, split_seed, pins)
    metadata_pin = pin(str(directory / f"{kind}.json"), root)
    metadata = load_json(metadata_pin["path"])
    for key, expected in dict(epoch=epoch, checkpoint=f"{kind}.pt", schedule=schedule).items():
        equal(metadata.get(key), expected, f"checkpoint metadata {key}")
    for key, expected in dict(seed=seed, split_seed=split_seed).items():
        if key in metadata or kind == "last":
            equal(metadata.get(key), expected, f"checkpoint metadata {key}")
    for key in ("lr", "forget_weight", "target_kind", "loss", "bn_mode"):
        if key in metadata:
            equal(metadata[key], parameter(config, key), f"checkpoint metadata {key}")
    checkpoint = pin(dict(path=str(directory / f"{kind}.pt"), sha256=item.get("checkpoint_sha256")), root)
    if metadata.get("is_original_fallback"):
        equal(epoch, 0, "Original fallback epoch")
        # torch.save of an unchanged state need not reproduce the Original's
        # serialized bytes. This metadata-only builder cannot prove tensor
        # equality and must not mistake different serialization for training.
    retrain = pin(item["paired_retrain"], root)
    retrain_meta = pin(item["paired_retrain_metadata"], root)
    saved_retrain = load_json(retrain_meta["path"])
    for key, expected in dict(seed=seed, split_seed=split_seed, epochs=182, test_used_during_training=False).items():
        equal(saved_retrain.get(key), expected, f"paired Retrain {key}")
    timing = dict.fromkeys(TIMING_KEYS)
    notes = ["Missing timing is unknown, not zero; no total is inferred from schedule wall time."]
    if kind == "last":
        equal(metadata.get("calibration_only"), True, "train-only checkpoint provenance")
        timing["training_loop_seconds"] = seconds(metadata["train_seconds"], "train_seconds")
    else:
        notes.append("best.json has no exact cumulative training-loop timer; last.json time is not substituted.")
    if metadata.get("target_generation_seconds") is not None:
        timing["target_generation_seconds"] = seconds(metadata["target_generation_seconds"], "target_generation_seconds")
    grouping = dict(group.get("config", {}))
    if "preregister_checks" in grouping or "schedule" in grouping:
        raise ValueError("group config reserves preregister_checks and schedule")
    grouping.update(preregister_checks=checks, schedule=schedule)
    entry = dict(id=ident, method=method, seed=seed, split_seed=split_seed, epoch=epoch,
        config=grouping, checkpoint=checkpoint, split_manifest=split,
        selection_audit_split=partition, preregistration=preregistration,
        paired_retrain=retrain, paired_retrain_metadata=retrain_meta,
        timing=timing, timing_notes=notes,
        confirmation=dict(checkpoint_kind=kind, preregistered_epochs=horizon,
                          original_sha256=config["original_sha256"], original_manifest=original,
                          schedule_wall_seconds=(seconds(metadata["wall_seconds"], "wall_seconds")
                                                 if metadata.get("wall_seconds") is not None else None)))
    entry["selection_metadata" if kind == "best" else "training_metadata"] = metadata_pin
    return entry


def build(plan_path, output):
    plan_path, output = plan_path.expanduser().resolve(strict=True), output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {output}")
    plan = load_json(plan_path)
    equal(plan.get("schema"), 1, "plan schema")
    if not isinstance(plan.get("groups"), dict) or not isinstance(plan.get("candidates"), list) or not plan["candidates"]:
        raise ValueError("plan needs groups and explicit nonempty candidates")
    pins, entries, seen = literal_pins(), [], set()
    for item in plan["candidates"]:
        entry = candidate(item, plan["groups"][item["group"]], plan_path.parent, pins)
        if entry["id"] in seen:
            raise ValueError(f"duplicate candidate id {entry['id']}")
        seen.add(entry["id"])
        entries.append(entry)
    # One evaluator invocation has one Original; independent Originals need
    # separate readable/frozen manifests and separate audit/test output roots.
    identities = {json.dumps(entry["confirmation"]["original_manifest"], sort_keys=True) for entry in entries}
    original_hashes = {entry["confirmation"]["original_sha256"] for entry in entries}
    if len(identities) != 1 or len(original_hashes) != 1:
        raise ValueError("use a separate manifest per Original")
    value = dict(schema=1, created_utc=datetime.now(timezone.utc).isoformat(),
        builder=dict(path=str(Path(__file__).resolve()), sha256=sha256(__file__),
                     plan=pin(str(plan_path), plan_path.parent)), candidates=entries)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    output.chmod(0o444)
    print(json.dumps(dict(event="confirmation_readable_manifest_built", path=str(output),
                          sha256=sha256(output), n_candidates=len(entries))), flush=True)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    build(args.plan, args.out)


if __name__ == "__main__":
    main()
