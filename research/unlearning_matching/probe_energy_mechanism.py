"""Pinned, immutable mechanism diagnostics for one frozen canonical energy model.

No training, selection, audit/test prediction, Retrain, or identity targets.
Explicit SHA256 pins bind best.pt, energy_reference.npz, preregister.json and
splits.json in one seed42_split42 run. Associated best/reference/split metadata
are validated and hashed. Only x_train/y_train are read from cifar10.npz.

Four fixed 1024-image groups are forget, target, selection and retain_subset;
Original also rechecks all 2500 target scores against the frozen reference.
Gradients use the old mechanism_probe's exact RandomState(290112) 256F/256R
draws from full F/R, with clean eval-mode inputs and the actual EnergyObjective.
Raw gradients exclude alpha, clipping, momentum, decay and retain augmentation.
Run --self-test for synthetic CPU checks; it does not read experiment data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "2")
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_PINS = {
    "research/unlearning_matching/mechanism_probe.py": "c80314b426dfa414f819c54ddcb24a81a5f065cd0998675745dda780d2a79411",
    "research/unlearning_matching/energy_study.py": "63d0de85deca05542f56e7e2e1ba0f709f8b2fb0791c0a627fb71ef6a2235c7c",
    "research/unlearning_matching/calibration_study.py": "b25cf925c71e81932023084179cdc5910bb135db3812214b84a9fc8a610f4e7b",
    "research/unlearning_matching/fixed_quantile.py": "760a2147acee205d2c34c0fbc4113be60161cea7bde74dd977bab4d81f625c79",
    "research/unlearning_matching/run_experiments.py": "b15c5d8fbc72977acc4306c12687b4f536d268a9c87db2519b623be47ec975f6",
    "tasks/security-machine-unlearning/scripts/common.py": "2cdfe76cb63217c72cca07eccba2863e3c4fd858627e9809ac53fa4724b22d87",
}
CANONICAL_ORIGINAL = "a136ae6dcb21ddcf4fa2adede89c2d7f3dd22c00217c1913ed3929aef95d9e58"
CANONICAL_TRAIN = "af4f586337eda0c26f82c4fd1864403d1ee0385e2ea10df41f8d40c8a70c8b89"
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
N_IMAGES, GRAD_SEED, REPORT_SEED = 1024, 290112, 20261108
REFERENCE_ATOL, REFERENCE_RTOL = 2e-4, 2e-5


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def source_artifacts():
    for name, expected in SOURCE_PINS.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"frozen probe dependency changed: {name}")
    if sha256(__file__) != SOURCE_SHA256:
        raise ValueError("energy probe changed after import")
    return {**SOURCE_PINS, "research/unlearning_matching/probe_energy_mechanism.py": SOURCE_SHA256}


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependencies():
    source_artifacts()
    probe = module_at("pinned_energy_mechanism_tools", HERE / "mechanism_probe.py")
    energy = module_at("pinned_energy_mechanism_loss", HERE / "energy_study.py")
    common = probe.common
    if (common.SHA256_ORIGINAL["cifar10"] != CANONICAL_ORIGINAL
            or common.SHA256_TRAIN["cifar10"] != CANONICAL_TRAIN or common.SPLIT_SEED != 1):
        raise ValueError("canonical Original/data pins were overridden")
    return probe, energy


def write_json_new(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    Path(path).chmod(0o444)


def write_npz_new(path, arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    Path(path).chmod(0o444)


def checked_artifact(path, expected, filename):
    path = Path(path).expanduser().resolve(strict=True)
    if (path.name != filename or not path.is_file()
            or not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None):
        raise ValueError(f"requires {filename} with an explicit SHA256")
    if sha256(path) != expected:
        raise ValueError(f"artifact SHA256 mismatch: {path}")
    return dict(path=str(path), sha256=expected)


def associated_artifact(path):
    return dict(path=str(path.resolve(strict=True)), sha256=sha256(path))


def validate_metadata(config, best, reference, energy):
    required = dict(seed=42, split_seed=42, train_sha256=CANONICAL_TRAIN,
        original_sha256=CANONICAL_ORIGINAL, study_kind="energy_distance_ablation",
        loss=energy.LOSS_NAME, target_kind=energy.TARGET_NAME,
        source_sha256=SOURCE_PINS["research/unlearning_matching/energy_study.py"],
        wrapper_source_sha256=SOURCE_PINS["research/unlearning_matching/energy_study.py"],
        inherited_scaffold_sha256=SOURCE_PINS["research/unlearning_matching/calibration_study.py"],
        frozen_runner_sha256=SOURCE_PINS["research/unlearning_matching/run_experiments.py"],
        fixed_target_source_sha256=SOURCE_PINS["research/unlearning_matching/fixed_quantile.py"],
        calibration_used_for_gradients=False, audit_predictions_computed=False,
        selection_split_seed=20261107, selection_size=1250, audit_size=1250,
        retain_subset=5000, batch_size=256)
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("energy preregistration differs from canonical provenance")
    statistic = config.get("energy_statistic")
    if statistic not in energy.STATISTICS:
        raise ValueError("unknown registered energy statistic")
    if (type(best.get("epoch")) is not int or type(config.get("epochs")) is not int
            or not 1 <= best["epoch"] <= config["epochs"]
            or best.get("checkpoint") != "best.pt" or best.get("feasible") is not True
            or best.get("is_original_fallback") is not False
            or best.get("schedule") not in config.get("schedules", [])
            or best.get("schedule") not in ("A", "B", "C", "D")
            or not isinstance(best.get("selection_J"), (int, float))
            or not math.isfinite(best["selection_J"])):
        raise ValueError("checkpoint is not a selected, feasible energy endpoint")
    for key in ("source_sha256", "wrapper_source_sha256", "inherited_scaffold_sha256",
                "energy_statistic", "loss", "target_kind", "lr", "bn_mode"):
        if best.get(key) != config.get(key):
            raise ValueError(f"selected checkpoint metadata differs: {key}")
    expected_reference = dict(statistic=statistic, reference_examples=2500,
        class_counts={str(cls): 250 for cls in range(10)}, audit_predicted=False,
        selection_used_for_targets=False, test_used=False, target_self_constant_included=True,
        target_self_divisor="m(m-1); ordered off-diagonal pairs",
        wrapper_source_sha256=SOURCE_PINS["research/unlearning_matching/energy_study.py"])
    if any(reference.get(key) != value for key, value in expected_reference.items()):
        raise ValueError("energy reference metadata differs from canonical provenance")
    if best.get("energy_reference_score_sha256") != reference.get("score_sha256"):
        raise ValueError("selected checkpoint and frozen reference score hashes differ")
    return statistic


def input_bundle(a, energy):
    inputs = {}
    for name, filename in (("checkpoint", "best.pt"), ("energy_reference", "energy_reference.npz"),
                           ("preregistration", "preregister.json"), ("splits", "splits.json")):
        inputs[name] = checked_artifact(getattr(a, name), getattr(a, name + "_sha256"), filename)
    root = Path(inputs["preregistration"]["path"]).parent
    checkpoint = Path(inputs["checkpoint"]["path"])
    if (root.name != "seed42_split42" or checkpoint.parent.parent != root
            or any(Path(inputs[name]["path"]).parent != root for name in ("energy_reference", "splits"))):
        raise ValueError("all artifacts must belong to one canonical seed42_split42 run")
    output = a.out_dir.expanduser().resolve()
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("output must be a new directory outside the input run")
    for name, path in {
        "selection_metadata": checkpoint.parent / "best.json",
        "reference_metadata": root / "energy_target_metadata.json",
        "selection_audit_split": root / "selection_audit_split.json",
    }.items():
        if path.is_symlink():
            raise ValueError("associated metadata must be direct files in the same run")
        inputs[name] = associated_artifact(path)
    documents = {name: json.loads(Path(item["path"]).read_text()) for name, item in inputs.items()
                 if name not in ("checkpoint", "energy_reference")}
    config, best = documents["preregistration"]["configuration"], documents["selection_metadata"]
    statistic = validate_metadata(config, best, documents["reference_metadata"], energy)
    if checkpoint.parent.name != best["schedule"]:
        raise ValueError("checkpoint parent differs from its registered schedule")
    for name in ("energy_study.py", "calibration_study.py"):
        snapshot = root / "source_snapshot" / name
        expected = SOURCE_PINS[f"research/unlearning_matching/{name}"]
        if sha256(snapshot) != expected:
            raise ValueError(f"training source snapshot differs: {name}")
        inputs["snapshot_" + name] = associated_artifact(snapshot)
    return output, inputs, documents, statistic


def canonical_splits(labels, saved, saved_selection, common):
    if labels.shape != (50000,) or not np.array_equal(np.bincount(labels, minlength=10), [5000] * 10):
        raise ValueError("expected the canonical balanced CIFAR-10 training labels")
    train, val, forget, retain = common.split_indices(labels, 10, 42, .1)
    rng, target, calibration = np.random.RandomState(20261106), [], []
    for cls in range(10):
        order = rng.permutation(val[labels[val] == cls])
        target.extend(order[:len(order) // 2]); calibration.extend(order[len(order) // 2:])
    target, calibration = np.sort(target), np.sort(calibration)
    expected = dict(seed=42, split_seed=42, validation_seed=1, target_split_seed=20261106,
        indices={key: value.tolist() for key, value in dict(train=train, forget=forget, retain=retain,
                                                           target=target, calibration=calibration).items()})
    if saved != expected:
        raise ValueError("split manifest differs from the canonical seed42 split")
    rng, selection, audit = np.random.RandomState(20261107), [], []
    for cls in range(10):
        order = rng.permutation(calibration[labels[calibration] == cls])
        selection.extend(order[:125]); audit.extend(order[125:])
    selection, audit = np.sort(selection), np.sort(audit)
    expected_selection = dict(seed=20261107, selection=selection.tolist(), audit=audit.tolist(), audit_predicted=False)
    if saved_selection != expected_selection:
        raise ValueError("selection/audit index manifest differs from the canonical split")
    positions = np.sort(np.random.RandomState(7312026).choice(len(retain), 5000, replace=False))
    return dict(forget=forget, retain=retain, target=target, selection=selection,
                retain_subset=retain[positions]), audit


def reference_arrays(path, metadata, expected_labels, statistic, energy, common):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"scores", "labels", "classes", "target_self"}:
            raise ValueError("energy reference contains unexpected arrays")
        scores, labels, classes, self_terms = (archive[name] for name in ("scores", "labels", "classes", "target_self"))
    dimension = 2 if statistic == "joint" else 1
    if (scores.shape != (2500, dimension) or scores.dtype != np.float32
            or labels.dtype != np.int64 or not np.array_equal(labels, expected_labels)
            or not np.isfinite(scores).all() or not np.array_equal(classes, np.arange(10))
            or self_terms.shape != (10,) or not np.isfinite(self_terms).all()):
        raise ValueError("energy reference scores/labels/classes have invalid shape, dtype or values")
    if common.sha256_arrays(scores, labels) != metadata.get("score_sha256"):
        raise ValueError("energy reference score SHA256 disagrees with its metadata")
    if statistic == "joint":
        difference = scores[:, 1] - scores[:, 0]
        if np.any(difference < -1e-4) or np.any(difference > np.log(9) + 1e-4):
            raise ValueError("joint scores violate the logodds/margin realizability range")
    values, yy = torch.from_numpy(scores), torch.from_numpy(labels)
    expected_self = torch.stack([energy.off_diagonal_mean_distances(values[yy == cls]) for cls in range(10)])
    if not torch.allclose(expected_self.double(), torch.from_numpy(self_terms).double(), atol=1e-5, rtol=1e-5):
        raise ValueError("stored target self terms do not match the reference scores")
    return values, yy


def checked_state(path, reference):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or set(state) != set(reference):
        raise ValueError("candidate state_dict keys differ from Original")
    for name, expected in reference.items():
        value = state[name]
        if (not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype
                or not torch.isfinite(value).all()):
            raise ValueError(f"candidate has invalid state tensor: {name}")
    return state


def report_indices(indices):
    groups = {}
    for name in ("forget", "target", "selection", "retain_subset"):
        full = indices[name]
        if len(full) < N_IMAGES:
            raise ValueError(f"{name} has fewer than the fixed {N_IMAGES} report images")
        keep = np.sort(np.random.RandomState(REPORT_SEED).choice(len(full), N_IMAGES, replace=False))
        groups[name] = full[keep]
    return groups


def gradient_indices(indices):
    rng = np.random.RandomState(GRAD_SEED)
    fi = np.sort(rng.choice(len(indices["forget"]), 256, replace=False))
    ri = np.sort(rng.choice(len(indices["retain"]), 256, replace=False))
    return dict(forget=indices["forget"][fi], retain=indices["retain"][ri])


def guard_forward_indices(groups, indices, audit, gradient_ids):
    allowed = np.unique(np.concatenate([indices[key] for key in ("forget", "retain", "target", "selection")]))
    for name, ids in {**groups, "reference_verification": indices["target"],
                      "gradient_forget": gradient_ids["forget"], "gradient_retain": gradient_ids["retain"]}.items():
        if (not np.isin(ids, allowed).all() or np.intersect1d(ids, audit).size
                or np.any(ids < 0) or np.any(ids >= 50000)):
            raise ValueError(f"forbidden audit/test index in forward group: {name}")


def feature_report(probe, original, candidate, images, labels, device):
    fo, zo = probe.feature_logits(original, images, device)
    fc, zc = probe.feature_logits(candidate, images, device)
    if not all(torch.isfinite(value).all() for value in (fo, fc, zo, zc)):
        raise ValueError("nonfinite features/logits in energy probe")
    so, sc = probe.scores(zo, labels), probe.scores(zc, labels)
    cosine = F.cosine_similarity(fo, fc, dim=1)
    combinations = dict(original_feature_original_head=zo, candidate_feature_candidate_head=zc,
        candidate_feature_original_head=F.linear(fc, original.fc.weight.detach().cpu(), original.fc.bias.detach().cpu()),
        original_feature_candidate_head=F.linear(fo, candidate.fc.weight.detach().cpu(), candidate.fc.bias.detach().cpu()))
    summary = dict(n=len(labels), groups=probe.group_summary(so, sc, labels),
        feature_cosine_mean=float(cosine.mean()), feature_cosine_min=float(cosine.min()),
        linear_cka=probe.linear_cka(fo, fc), cka_images=len(fo),
        head_feature_combinations={name: probe.basic_metrics(value, labels) for name, value in combinations.items()},
        hybrid_interpretation="descriptive head/feature swaps in their shared coordinate system; not causal attribution")
    arrays = dict(labels=labels.numpy(), original_features=fo.numpy(), candidate_features=fc.numpy(),
                  original_logits=zo.numpy(), candidate_logits=zc.numpy(), feature_cosine=cosine.numpy())
    for name, logits in combinations.items():
        arrays[name + "_logits"] = logits.numpy()
    for name, values in (("original", so), ("candidate", sc)):
        arrays.update({f"{name}_{key}": value.numpy() for key, value in values.items()})
    for key in ("confidence", "logodds", "margin"):
        arrays["delta_" + key] = (sc[key] - so[key]).numpy()
    arrays.update(correct_to_error=(~so["error"] & sc["error"]).numpy(),
                  error_to_correct=(so["error"] & ~sc["error"]).numpy())
    return summary, arrays


def energy_gradient_report(probe, energy, model, images_f, labels_f, images_r, labels_r,
                           reference_scores, reference_labels, statistic):
    model.eval()
    params = list(model.named_parameters())
    before = probe.cal.bn_buffers(model)
    objective = energy.EnergyObjective(reference_scores, reference_labels, statistic)
    loss_f = objective(model(images_f), labels_f)
    gf = torch.autograd.grad(loss_f, [value for _, value in params])
    loss_r = F.cross_entropy(model(images_r), labels_r)
    gr = torch.autograd.grad(loss_r, [value for _, value in params])
    if not all(torch.isfinite(value).all() for value in (*gf, *gr)):
        raise ValueError("nonfinite raw energy/retain gradients")
    if any(not torch.equal(value, probe.cal.bn_buffers(model)[key]) for key, value in before.items()):
        raise RuntimeError("BN buffers changed during an eval-mode gradient probe")
    report = dict(forget_energy_loss=float(loss_f.detach()), retain_ce_loss=float(loss_r.detach()),
        gradient_units="raw unweighted class-macro energy vs batch-mean retain CE; no alpha/clip/momentum/decay/augmentation",
        energy_statistic=statistic, energy_statistics=objective.snapshot(),
        global_statistics=probe.grad_statistics(gf, gr), blocks={})
    for block in probe.BLOCKS:
        positions = [i for i, (name, _) in enumerate(params) if probe.block_name(name) == block]
        report["blocks"][block] = probe.grad_statistics([gf[i] for i in positions], [gr[i] for i in positions])
    return report


def execute(a):
    probe, energy = dependencies()
    common, base = probe.common, probe.base
    output, inputs, documents, statistic = input_bundle(a, energy)
    x, y = base.train_arrays(a)
    if x.shape != (50000, 32, 32, 3) or x.dtype != np.uint8:
        raise ValueError("invalid canonical training images")
    indices, audit = canonical_splits(y, documents["splits"], documents["selection_audit_split"], common)
    reference, reference_labels = reference_arrays(inputs["energy_reference"]["path"],
        documents["reference_metadata"], y[indices["target"]], statistic, energy, common)
    original_state = base.original_state(a)
    candidate_state = checked_state(inputs["checkpoint"]["path"], original_state)
    groups, gradient_ids = report_indices(indices), gradient_indices(indices)
    guard_forward_indices(groups, indices, audit, gradient_ids)
    inputs["original"] = dict(path=str(Path(a.original).resolve()), sha256=CANONICAL_ORIGINAL)
    data_artifact = dict(path=str((a.data_dir / "cifar10.npz").resolve()),
                         decoded_train_arrays_sha256=CANONICAL_TRAIN, members_read=["x_train", "y_train"])
    # No existing directory can be resumed/overwritten, including failed probes.
    output.mkdir(parents=True, exist_ok=False)
    write_json_new(output / "started.json", dict(schema=1, complete=False,
        created_utc=datetime.now(timezone.utc).isoformat(), inputs=inputs, data=data_artifact, sources=source_artifacts()))
    original, candidate = probe.model_on(original_state, a.device), probe.model_on(candidate_state, a.device)
    with torch.no_grad():
        target_images = common.to_gpu_u8(x[indices["target"]], a.device)
        target_logits = probe.fixed.clean_logits(original, target_images)
        actual = energy.scores_from_logits(target_logits, reference_labels, statistic)
        reference_delta = float((actual - reference).abs().max())
        if not torch.allclose(actual, reference, atol=REFERENCE_ATOL, rtol=REFERENCE_RTOL):
            raise ValueError("frozen reference differs from canonical Original target scores")
        del target_images, target_logits, actual
    arrays, summaries = {}, {}
    for name, positions in groups.items():
        report, values = feature_report(probe, original, candidate, x[positions], torch.from_numpy(y[positions]), a.device)
        summaries[name] = report
        arrays.update({f"{name}_{key}": value for key, value in values.items()})
        arrays[name + "_indices"] = positions
        base.log("energy_mechanism_group", group=name, n=len(positions), linear_cka=report["linear_cka"],
                 original_error=report["groups"]["all"]["original_error_pct"],
                 candidate_error=report["groups"]["all"]["candidate_error_pct"])
    changes = probe.parameter_changes(original_state, candidate_state, original, candidate)
    gf, gr = gradient_ids["forget"], gradient_ids["retain"]
    arguments = (common.to_gpu_u8(x[gf], a.device).float() / 255., torch.from_numpy(y[gf]).to(a.device),
                 common.to_gpu_u8(x[gr], a.device).float() / 255., torch.from_numpy(y[gr]).to(a.device),
                 reference.to(a.device), reference_labels.to(a.device), statistic)
    gradients = {name: energy_gradient_report(probe, energy, model, *arguments)
                 for name, model in (("original", original), ("candidate", candidate))}
    # Recheck every artifact and dependency after forwards, before publishing.
    sources = source_artifacts()
    for item in inputs.values():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("an input artifact changed during probing")
    write_npz_new(output / "per_example.npz", arrays)
    write_json_new(output / "mechanism.json", dict(schema=1, complete=True,
        created_utc=datetime.now(timezone.utc).isoformat(), sources=sources, inputs=inputs, data=data_artifact,
        seed=42, split_seed=42, energy_statistic=statistic, selected=documents["selection_metadata"],
        reported_images_per_group=N_IMAGES, report_seed=REPORT_SEED,
        reference_verification=dict(n=2500, max_absolute_score_delta=reference_delta,
            atol=REFERENCE_ATOL, rtol=REFERENCE_RTOL,
            interpretation="strict cached byte/provenance pins plus numerical Original re-evaluation tolerance"),
        gradient_sampling=dict(seed=GRAD_SEED, forget_indices=gf.tolist(), retain_indices=gr.tolist(),
            forget_source="full canonical F, 4500", retain_source="full canonical R, 40500",
            exact_indices_match_old_mechanism_probe=True, images_per_group=256),
        groups=summaries, changes=changes, gradients=gradients,
        identity_assignment=False, per_example_target_residuals=False, audit_predicted=False,
        test_used=False, retrain_used=False, checkpoint_reselected=False,
        warning="descriptive mechanism diagnostics; loss scales differ from MSE; no causal/deletion guarantee"))
    write_json_new(output / "complete.json", dict(schema=1, complete=True,
        artifacts={name: sha256(output / name) for name in ("started.json", "per_example.npz", "mechanism.json")}))
    output.chmod(0o555)
    base.log("energy_mechanism_complete", output=str(output), sha256=sha256(output / "mechanism.json"))


def self_test():
    torch.set_num_threads(1)
    probe, energy = dependencies()
    rng = np.random.RandomState(20261107)
    y = np.repeat(np.arange(10), 5000)
    # Canonical split reconstruction uses the real fixed splitter; only labels
    # are synthetic. No experiment data or checkpoint is opened by this test.
    train, val, forget, retain = probe.common.split_indices(y, 10, 42, .1)
    target, calibration, split_rng = [], [], np.random.RandomState(20261106)
    for cls in range(10):
        order = split_rng.permutation(val[y[val] == cls]); target.extend(order[:250]); calibration.extend(order[250:])
    target, calibration = np.sort(target), np.sort(calibration)
    saved = dict(seed=42, split_seed=42, validation_seed=1, target_split_seed=20261106,
        indices={key: value.tolist() for key, value in dict(train=train, forget=forget, retain=retain,
                                                           target=target, calibration=calibration).items()})
    selection, audit, split_rng = [], [], np.random.RandomState(20261107)
    for cls in range(10):
        order = split_rng.permutation(calibration[y[calibration] == cls]); selection.extend(order[:125]); audit.extend(order[125:])
    split_metadata = dict(seed=20261107, selection=sorted(selection), audit=sorted(audit), audit_predicted=False)
    indices, audit = canonical_splits(y, saved, split_metadata, probe.common)
    groups, gradient_ids = report_indices(indices), gradient_indices(indices)
    guard_forward_indices(groups, indices, audit, gradient_ids)
    assert all(len(value) == 1024 for value in groups.values())
    assert all(np.array_equal(value, report_indices(indices)[key]) for key, value in groups.items())
    legacy_rng = np.random.RandomState(290112)
    assert np.array_equal(gradient_ids["forget"], forget[np.sort(legacy_rng.choice(4500, 256, replace=False))])
    assert np.array_equal(gradient_ids["retain"], retain[np.sort(legacy_rng.choice(40500, 256, replace=False))])
    try:
        guard_forward_indices({"audit": audit[:1]}, indices, audit, gradient_ids)
    except ValueError:
        pass
    else:
        raise AssertionError("audit forward was accepted")

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1, self.bn1 = torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4)
            for index in range(1, 5):
                setattr(self, f"layer{index}", torch.nn.Linear(4, 4))
            self.fc = torch.nn.Linear(4, 10)

        def forward(self, images):
            hidden = torch.tanh(self.bn1(self.conv1(images.flatten(1))))
            for index in range(1, 5):
                hidden = torch.tanh(getattr(self, f"layer{index}")(hidden))
            return self.fc(hidden)

    torch.manual_seed(17)
    original, candidate = ToyModel().eval(), ToyModel().eval()
    candidate.load_state_dict(original.state_dict())
    with torch.no_grad():
        candidate.layer4.weight.add_(.03); candidate.fc.weight.mul_(1.03)
    images, labels = rng.randint(0, 256, size=(64, 1, 1, 3), dtype=np.uint8), torch.arange(64) % 10
    summary, arrays = feature_report(probe, original, candidate, images, labels, "cpu")
    assert len(summary["head_feature_combinations"]) == 4 and 0 <= summary["linear_cka"] <= 1 + 1e-12
    assert not any("target" in key for key in arrays)
    original_state = {key: value.detach().clone() for key, value in original.state_dict().items()}
    candidate_state = {key: value.detach().clone() for key, value in candidate.state_dict().items()}
    changes = probe.parameter_changes(original_state, candidate_state, original, candidate)
    assert changes["bn_buffer_max_delta"] == 0 and changes["parameter_blocks"]["layer4"]["delta_norm"] > 0
    tensor_images = probe.common.to_gpu_u8(images, "cpu").float() / 255.
    for statistic in energy.STATISTICS:
        reference_labels = torch.arange(40) % 10
        reference_scores = energy.scores_from_logits(torch.randn(40, 10), reference_labels, statistic)
        report = energy_gradient_report(probe, energy, candidate, tensor_images[:32], labels[:32],
            tensor_images[32:], labels[32:], reference_scores, reference_labels, statistic)
        assert math.isfinite(report["global_statistics"]["cosine"])
        assert report["energy_statistics"]["used_class_terms"] == 10
        assert all(parameter.grad is None for parameter in candidate.parameters())
    with tempfile.TemporaryDirectory(prefix="energy-mechanism-cpu-") as temporary:
        folder = Path(temporary)
        checkpoint = folder / "best.pt"; torch.save(candidate_state, checkpoint)
        checked_artifact(checkpoint, sha256(checkpoint), "best.pt")
        checked_state(checkpoint, original_state)
        for digest, filename in (("0" * 64, "best.pt"), (sha256(checkpoint), "retrain.pt")):
            try:
                checked_artifact(checkpoint, digest, filename)
            except ValueError:
                pass
            else:
                raise AssertionError("wrong artifact hash/name was accepted")
        output = folder / "record.json"; write_json_new(output, dict(ok=True))
        try:
            write_json_new(output, dict(overwrite=True))
        except FileExistsError:
            pass
        else:
            raise AssertionError("immutable JSON was overwritten")
        statistic, labels_ref = "joint", torch.from_numpy(y[target])
        scores_ref = energy.scores_from_logits(torch.randn(2500, 10), labels_ref, statistic)
        self_terms = np.asarray([float(energy.off_diagonal_mean_distances(scores_ref[labels_ref == cls])) for cls in range(10)])
        path = folder / "energy_reference.npz"
        np.savez(path, scores=scores_ref.numpy(), labels=labels_ref.numpy(), classes=np.arange(10), target_self=self_terms)
        metadata = dict(score_sha256=probe.common.sha256_arrays(scores_ref.numpy(), labels_ref.numpy()))
        reference_arrays(path, metadata, y[target], statistic, energy, probe.common)
        try:
            reference_arrays(path, dict(score_sha256="0" * 64), y[target], statistic, energy, probe.common)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong frozen reference score hash was accepted")
    source_artifacts()
    assert not torch.cuda.is_initialized()
    print("energy mechanism CPU checks passed: source pins, canonical split reconstruction, fixed report/legacy gradient indices, audit rejection, feature/head/CKA/block/BN probes, real energy gradients, artifact/reference integrity and immutable writes; CUDA uninitialized")


def main():
    if "--self-test" in sys.argv:
        if len(sys.argv) != 2:
            raise ValueError("--self-test takes no other arguments")
        self_test(); return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--original", default="/srv/home/bohanlyu/data_mlsbench/unlearn-lab/original/cifar10_resnet18_original.pt")
    parser.add_argument("--out-dir", type=Path, required=True, help="new immutable output directory; no resume or overwrite")
    for name in ("checkpoint", "energy-reference", "preregistration", "splits"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--threads", type=int, choices=(1, 2, 4), default=2)
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    execute(args)


if __name__ == "__main__":
    main()
