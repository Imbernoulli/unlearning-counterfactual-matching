"""Use the unchanged one-way evaluator with a verified independent Original.

--original-manifest points to train_original.py's completed original.json.
All remaining arguments are forwarded to evaluate_selected.py.  --original
may be omitted; if supplied, it must resolve to the manifest's original.pt.
Freeze, audit and test bind the same wrapper, helper, training source,
Original manifest and checkpoint hashes.  The replacement Original pin is
explicit, process-local, and applied only after the evaluator builds common.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
COMMON_PATH = HERE.parents[1] / "tasks/security-machine-unlearning/scripts/common.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def constants(path, names):
    """Read literal provenance pins without importing Torch or initializing CUDA."""
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    result[target.id] = ast.literal_eval(node.value)
    if set(result) != set(names):
        raise ValueError("could not read the canonical data pins")
    return result


def verify_original(path):
    """CPU-only checkpoint, dataset-pin, and source-provenance verification."""
    helper = load_module("independent_eval_manifest_helper", HERE / "with_original_manifest.py")
    path = path.expanduser().resolve(strict=True)
    manifest, checkpoint = helper.manifest_pin(path)
    config = manifest["configuration"]
    pins = constants(COMMON_PATH, {"SHA256_TRAIN", "SPLIT_SEED"})
    source = config.get("source", {})
    if (config["train_sha256"] != pins["SHA256_TRAIN"]["cifar10"]
            or config.get("canonical_validation_split_seed") != pins["SPLIT_SEED"]
            or source.get("common_sha256") != helper.sha256(COMMON_PATH)
            or source.get("script_sha256") != helper.sha256(HERE / "train_original.py")):
        raise ValueError("Original training data/common/training-source provenance differs")
    if (config.get("test_used") is not False
            or config.get("validation_predictions_used") is not False
            or config.get("retrain_used") is not False
            or config.get("array_keys_read") != ["x_train", "y_train"]
            or config.get("recipe", {}).get("epochs") != 182):
        raise ValueError("Original manifest does not describe the fixed train-only 182-epoch recipe")
    return helper, manifest, checkpoint, path


def original_arguments(arguments, checkpoint):
    """Reject mismatching or duplicate --original before the evaluator runs."""
    supplied = []
    for offset, arg in enumerate(arguments):
        if arg == "--original":
            if offset + 1 >= len(arguments):
                raise ValueError("--original needs a checkpoint path")
            supplied.append(arguments[offset + 1])
        elif arg.startswith("--original="):
            supplied.append(arg.split("=", 1)[1])
    if len(supplied) > 1:
        raise ValueError("supply --original at most once")
    if supplied:
        if Path(supplied[0]).expanduser().resolve(strict=True) != checkpoint:
            raise ValueError("--original must match --original-manifest's checkpoint")
        return list(arguments)
    return ["--original", str(checkpoint), *arguments]


def bind_evaluator(helper, manifest, checkpoint, manifest_path):
    """Wrap only the freshly imported evaluator instance; no source file edits."""
    evaluator = load_module("independent_selected_evaluator", HERE / "evaluate_selected.py")
    source_artifacts, original_runtime, original_load_arrays = (
        evaluator.source_artifacts, evaluator.runtime, evaluator.load_arrays)
    manifest_hash = helper.sha256(manifest_path)
    expected_pin = manifest["original_sha256"]

    def bound_sources():
        if helper.sha256(manifest_path) != manifest_hash:
            raise ValueError("Original manifest changed during evaluation")
        if helper.sha256(checkpoint) != expected_pin:
            raise ValueError("independent Original changed during evaluation")
        extra = {
            "evaluate_independent": Path(__file__).resolve(),
            "original_pin_helper": HERE / "with_original_manifest.py",
            "original_training_source": HERE / "train_original.py",
            "independent_original_manifest": manifest_path,
            "independent_original_checkpoint": checkpoint,
        }
        return {**source_artifacts(), **{name: dict(path=str(path), sha256=helper.sha256(path))
                                         for name, path in extra.items()}}

    def pinned_runtime(args):
        if Path(args.original).expanduser().resolve(strict=True) != checkpoint:
            raise ValueError("runtime Original differs from the verified manifest")
        device = original_runtime(args)
        previous = evaluator.common.SHA256_ORIGINAL["cifar10"]
        evaluator.common.SHA256_ORIGINAL["cifar10"] = expected_pin
        print(json.dumps(dict(event="explicit_evaluation_process_local_original_pin",
              original_training_seed=manifest["seed"], checkpoint=str(checkpoint),
              original_manifest=str(manifest_path), original_manifest_sha256=manifest_hash,
              previous_original_sha256=previous, original_sha256=expected_pin,
              train_sha256=manifest["train_sha256"],
              train_membership_sha256=manifest["train_membership_sha256"],
              wrapper_sha256=helper.sha256(__file__))), flush=True)
        return device

    def checked_arrays(data_dir, full):
        arrays, digest = original_load_arrays(data_dir, full)
        if digest != (evaluator.common.SHA256_DATA if full else evaluator.common.SHA256_TRAIN)["cifar10"]:
            raise ValueError("evaluator returned data with a different pin")
        members = evaluator.common.split_indices(arrays[1], 10, forget_seed=0, forget_fraction=.1)[0]
        encoded = evaluator.np.asarray(members, dtype="<i8").tobytes(order="C")
        if hashlib.sha256(encoded).hexdigest() != manifest["train_membership_sha256"]:
            raise ValueError("independent Original's 45000 training members differ from the evaluator")
        return arrays, digest

    evaluator.source_artifacts = bound_sources
    evaluator.runtime = pinned_runtime
    evaluator.load_arrays = checked_arrays
    return evaluator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
             epilog="Forward --candidate-manifest, --freeze or --phase, --data-dir, --out-dir, --gpu and --threads as usual.")
    parser.add_argument("--original-manifest", type=Path, required=True)
    args, forwarded = parser.parse_known_args(argv)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    helper, manifest, checkpoint, path = verify_original(args.original_manifest)
    forwarded = original_arguments(forwarded, checkpoint)
    print(json.dumps(dict(event="independent_evaluation_original_verified", manifest=str(path),
          manifest_sha256=helper.sha256(path), original_sha256=manifest["original_sha256"],
          original_training_seed=manifest["seed"])), flush=True)
    evaluator = bind_evaluator(helper, manifest, checkpoint, path)
    previous = sys.argv
    try:
        sys.argv = [str(HERE / "evaluate_selected.py"), *forwarded]
        evaluator.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
