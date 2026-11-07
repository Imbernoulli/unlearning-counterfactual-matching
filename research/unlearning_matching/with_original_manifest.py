"""Run an unchanged study with an explicitly hash-pinned independent Original.

The pin is changed only on an imported common module in this process. Lazy
geometry/energy harnesses are checked and pinned after their own loader returns.
Dataset checks remain intact; runners record the new pin in their ordinary
preregistration. This does not support evaluate_selected.py, which creates a
new common module inside its runtime function and needs manifest-aware final
evaluation separately.
"""
from __future__ import annotations

import argparse
from functools import wraps
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import ModuleType

HERE = Path(__file__).resolve().parent
RUNNERS = ("run_experiments.py", "fixed_quantile.py", "calibration_study.py",
           "localized_study.py", "dynamic_rank_study.py", "geometry_study.py",
           "energy_study.py", "saliency_study.py")
LAZY_RUNNERS = ("geometry_study.py", "energy_study.py")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_pin(path):
    path = path.resolve(strict=True)
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != 1 or manifest.get("complete") is not True:
        raise ValueError("a completed train_original manifest is required")
    expected = manifest.get("original_sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("manifest must explicitly supply original_sha256")
    if manifest.get("original_file") != "original.pt":
        raise ValueError("unexpected Original filename in manifest")
    checkpoint = (path.parent / manifest["original_file"]).resolve(strict=True)
    if sha256(checkpoint) != expected:
        raise ValueError("Original checkpoint differs from the manifest pin")
    config = manifest.get("configuration", {})
    if (config.get("dataset") != "cifar10" or config.get("train_size") != 45000
            or config.get("seed") != manifest.get("seed")
            or config.get("train_sha256") != manifest.get("train_sha256")
            or config.get("train_membership_sha256") != manifest.get("train_membership_sha256")):
        raise ValueError("inconsistent Original training provenance")
    if not re.fullmatch(r"[0-9a-f]{64}", str(config.get("train_membership_sha256"))):
        raise ValueError("manifest must explicitly supply training-membership SHA256")
    return manifest, checkpoint


def shared_common(module):
    """Check every exposed study/base/fixed alias before changing any pin."""
    pending, seen, instances = [module], set(), []
    while pending:
        candidate = pending.pop()
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        common = getattr(candidate, "common", None)
        if common is not None:
            instances.append(common)
        for name in ("base", "fixed", "study"):
            child = getattr(candidate, name, None)
            if isinstance(child, ModuleType):
                pending.append(child)
    if not instances:
        raise RuntimeError("runner/harness exposes no common module to pin")
    if any(common is not instances[0] for common in instances[1:]):
        raise RuntimeError("runner has multiple common instances; refusing an incomplete pin override")
    return instances[0]


def pin_instance(module, manifest, checkpoint, provenance, stage):
    common = shared_common(module)
    config = manifest["configuration"]
    common_path = HERE.parents[1] / "tasks/security-machine-unlearning/scripts/common.py"
    if (Path(common.__file__).resolve() != common_path.resolve()
            or config["train_sha256"] != common.SHA256_TRAIN["cifar10"]
            or config.get("canonical_validation_split_seed") != common.SPLIT_SEED
            or config.get("source", {}).get("common_sha256") != sha256(common_path)):
        raise ValueError("Original training data/common version differs from the runner")
    previous_pin = common.SHA256_ORIGINAL["cifar10"]
    common.SHA256_ORIGINAL["cifar10"] = manifest["original_sha256"]
    print(json.dumps(dict(event="explicit_process_local_original_pin", **provenance,
                          pin_stage=stage, common_source_sha256=sha256(common_path),
                          harness_source_sha256=sha256(module.__file__),
                          checkpoint=str(checkpoint), original_training_seed=manifest["seed"],
                          previous_original_sha256=previous_pin,
                          original_sha256=manifest["original_sha256"],
                          train_sha256=config["train_sha256"],
                          train_membership_sha256=config["train_membership_sha256"])), flush=True)
    return common


def load_and_pin(runner, manifest, checkpoint, manifest_path):
    if runner not in RUNNERS:
        raise ValueError(f"unsupported study runner: {runner}")
    source = HERE / runner
    if not source.is_file():
        raise FileNotFoundError(f"supported study runner is absent: {source}")
    provenance = dict(runner=runner, runner_sha256=sha256(source),
                      wrapper_sha256=sha256(__file__),
                      manifest=str(Path(manifest_path).resolve(strict=True)),
                      manifest_sha256=sha256(manifest_path))
    spec = importlib.util.spec_from_file_location("manifest_pinned_study", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if runner in LAZY_RUNNERS:
        if getattr(module, "common", None) is not None or any(
                isinstance(getattr(module, name, None), ModuleType)
                for name in ("base", "fixed", "study")):
            raise RuntimeError("lazy runner unexpectedly exposes eager harness aliases; refusing partial pinning")
        loader = getattr(module, "load_harness", None)
        if not callable(loader):
            raise RuntimeError("lazy runner has no callable load_harness")

        @wraps(loader)
        def pinned_loader(*args, **kwargs):
            harness = loader(*args, **kwargs)
            pin_instance(harness, manifest, checkpoint, provenance, "load_harness")
            return harness

        # Only this module's loader is replaced. Every returned harness has its
        # own aliases/source/data pins checked before the caller can use it.
        module.load_harness = pinned_loader
        print(json.dumps(dict(event="original_pin_deferred_until_load_harness", **provenance)), flush=True)
    else:
        pin_instance(module, manifest, checkpoint, provenance, "module_import")
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runner", choices=RUNNERS, required=True)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    runner_args = args.runner_args
    if runner_args and runner_args[0] == "--":
        runner_args = runner_args[1:]
    if any(arg == "--original" or arg.startswith("--original=") for arg in runner_args):
        parser.error("omit --original: it is resolved exclusively from --manifest")
    manifest, checkpoint = manifest_pin(args.manifest)
    print(json.dumps(dict(event="original_manifest_verified", manifest=str(args.manifest.resolve()),
                          manifest_sha256=sha256(args.manifest))), flush=True)
    module = load_and_pin(args.runner, manifest, checkpoint, args.manifest)
    sys.argv = [str(HERE / args.runner), "--original", str(checkpoint), *runner_args]
    module.main()


if __name__ == "__main__":
    main()
