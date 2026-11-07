"""Class-conditional energy-distance ablation around the frozen study harness.

The wrapper matches frozen Original held-out scores to forget minibatches,
using unsquared Euclidean distance on joint (log-odds, signed margin) by
(default), or log-odds alone. It inherits batches, retain CE/anchor, schedules,
optimizer, BN handling, selection, and audit isolation from calibration_study.

Both self terms exclude diagonal pairs and use n(n-1). This is the ordinary
U-statistic energy estimator; it can be negative on finite samples, including
when the two empirical sets are identical. The target self term is a frozen,
precomputed constant. Under iid sampling it estimates population energy;
finite-dataset sampling without replacement and skipped singleton classes
require their own finite-population interpretation. No claim of exact full-
data Wasserstein equivalence or certified deletion is made. Energy distance
and Cramer/energy objectives are existing methods, not a novelty claim.

Use the normal harness CLI, --target-kind global_logodds --loss logodds_mse,
plus --statistic joint|logodds. These two inherited options are parser
placeholders only: every persisted method configuration records the actual
energy loss/target instead. --self-test is CPU-only and reads no dataset.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "2")
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
STATISTICS = ("joint", "logodds")
LOSS_NAME = "class_conditional_energy_u_statistic"
TARGET_NAME = "frozen_original_heldout_score_distribution"
INTERPRETATION = (
    "Unsquared Euclidean distances; both self terms are off-diagonal U-statistics. "
    "Finite-sample estimates can be negative. Iid unbiasedness does not imply "
    "exact full finite-dataset energy under without-replacement minibatches or "
    "random singleton-class omission. No Wasserstein or deletion guarantee.")


def pairwise_distances(x, y):
    """Unsquared Euclidean norm; zero-distance derivatives use PyTorch's zero."""
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("score arrays must be N x d with the same d")
    return torch.linalg.vector_norm(x[:, None, :] - y[None, :, :], dim=-1)


def off_diagonal_mean_distances(x):
    """Mean over ordered distinct sample pairs, with divisor n(n-1)."""
    n = len(x)
    if n < 2:
        raise ValueError("an off-diagonal U-statistic needs at least two samples")
    distances = pairwise_distances(x, x)
    return (distances.sum() - distances.diagonal().sum()) / (n * (n - 1))


def energy_u_statistic(x, y, target_self=None):
    if len(x) < 2 or len(y) < 2:
        raise ValueError("both score samples need at least two points")
    y = y.detach()
    if target_self is None:
        target_self = off_diagonal_mean_distances(y)
    return (2 * pairwise_distances(x, y).mean()
            - off_diagonal_mean_distances(x) - target_self.detach())


def scores_from_logits(logits, labels, statistic):
    if statistic not in STATISTICS:
        raise ValueError("unknown score statistic")
    true = logits.gather(1, labels[:, None]).squeeze(1)
    mask = torch.nn.functional.one_hot(labels, logits.shape[1]).bool()
    other = logits.masked_fill(mask, -torch.inf)
    logodds = true - torch.logsumexp(other, dim=1)
    if statistic == "logodds":
        return logodds[:, None]
    margin = true - other.max(dim=1).values
    return torch.stack((logodds, margin), dim=1)


class EnergyObjective:
    """Frozen class references and per-schedule diagnostics, without selection."""
    def __init__(self, scores, labels, statistic):
        if scores.ndim != 2 or labels.ndim != 1 or len(scores) != len(labels):
            raise ValueError("reference score/label shapes differ")
        if scores.shape[1] != (2 if statistic == "joint" else 1):
            raise ValueError("reference score dimension differs from statistic")
        if not torch.isfinite(scores).all():
            raise ValueError("reference scores must be finite")
        self.statistic = statistic
        self.references, self.target_self = {}, {}
        for cls in labels.unique(sorted=True).tolist():
            reference = scores[labels == cls].detach().clone()
            if len(reference) < 2:
                raise ValueError(f"class {cls} has fewer than two held-out references")
            self.references[int(cls)] = reference
            self.target_self[int(cls)] = off_diagonal_mean_distances(reference).detach()
        self.reset()

    def reset(self):
        self.calls = self.used_class_terms = self.skipped_class_terms = self.all_skipped_calls = 0
        self.skipped_by_class = {}

    def snapshot(self):
        return dict(loss_calls=self.calls, used_class_terms=self.used_class_terms,
                    skipped_class_terms=self.skipped_class_terms,
                    all_skipped_calls=self.all_skipped_calls,
                    skipped_by_class={str(k): v for k, v in self.skipped_by_class.items()},
                    singleton_rule="skip current class with n<2; macro-average remaining classes")

    def __call__(self, logits, labels):
        current = scores_from_logits(logits, labels, self.statistic)
        terms = []
        self.calls += 1
        for cls in labels.unique(sorted=True).tolist():
            cls = int(cls)
            if cls not in self.references:
                raise ValueError(f"class {cls} has no held-out reference")
            values = current[labels == cls]
            if len(values) < 2:
                self.skipped_class_terms += 1
                self.skipped_by_class[cls] = self.skipped_by_class.get(cls, 0) + 1
                continue
            terms.append(energy_u_statistic(values, self.references[cls], self.target_self[cls]))
        self.used_class_terms += len(terms)
        if not terms:
            self.all_skipped_calls += 1
            # Keep a valid zero gradient graph for a singleton-only batch.
            return logits.sum() * 0.
        return torch.stack(terms).mean()


def load_harness():
    spec = importlib.util.spec_from_file_location("energy_calibration_harness", HERE / "calibration_study.py")
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    return study


def reference_from_target_only(a, teacher, data, study, statistic):
    """The only data lookup is target; a poisoned selection entry is irrelevant."""
    xx, yy = data["target"]
    if xx.device.type == "cuda":
        torch.cuda.synchronize(xx.device)
    started = time.monotonic()
    logits = study.fixed.clean_logits(teacher, xx)
    labels = yy.detach().cpu()
    scores = scores_from_logits(logits, labels, statistic)
    device_scores, device_labels = scores.to(yy.device), labels.to(yy.device)
    objective = EnergyObjective(device_scores, device_labels, statistic)
    if xx.device.type == "cuda":
        torch.cuda.synchronize(xx.device)
    seconds = time.monotonic() - started
    # The non-data-dependent source/split pins are supplied by preregistration.
    # Persist actual scores too, so the frozen reference is inspectable.
    arrays = dict(scores=scores.numpy(), labels=labels.numpy(),
                  classes=np.asarray(list(objective.target_self), dtype=np.int64),
                  target_self=np.asarray([float(v) for v in objective.target_self.values()]))
    np.savez(a.run_dir / "energy_reference.npz", **arrays)
    metadata = dict(statistic=statistic, reference_examples=len(labels),
        class_counts={str(k): len(v) for k, v in objective.references.items()},
        score_sha256=study.common.sha256_arrays(scores.numpy(), labels.numpy()),
        target_generation_seconds=seconds, target_self_constant_included=True,
        target_self_divisor="m(m-1); ordered off-diagonal pairs",
        audit_predicted=False, selection_used_for_targets=False, test_used=False,
        wrapper_source_sha256=SOURCE_SHA256, interpretation=INTERPRETATION)
    study.base.dump(a.run_dir / "energy_target_metadata.json", metadata)
    study.base.log("energy_reference_prepared", **metadata)
    return objective, metadata


def install_hooks(study, statistic):
    original_preregister, original_schedule = study.preregister, study.study_schedule
    original_save_last, original_dump = study.save_last, study.base.dump
    harness_sha = hashlib.sha256(Path(study.__file__).read_bytes()).hexdigest()
    active = {"objective": None, "target_metadata": None}

    def provenance():
        return dict(loss=LOSS_NAME, target_kind=TARGET_NAME, energy_statistic=statistic,
            source_sha256=SOURCE_SHA256, wrapper_source_sha256=SOURCE_SHA256,
            inherited_scaffold_sha256=harness_sha,
            energy_distance="unsquared Euclidean; no kernel bandwidth",
            energy_reduction="macro-average current classes with n>=2",
            energy_self_terms="ordered non-diagonal / n(n-1), m(m-1)",
            energy_interpretation=INTERPRETATION,
            parser_loss_placeholder="logodds_mse", parser_target_placeholder="global_logodds")

    def preregister(a):
        if a.target_kind != "global_logodds" or a.loss != "logodds_mse":
            raise ValueError("energy wrapper requires parser placeholders --target-kind global_logodds --loss logodds_mse")
        with tempfile.TemporaryDirectory(prefix="unlearning-energy-config-") as temporary:
            shadow = SimpleNamespace(**{**vars(a), "run_dir": Path(temporary)})
            original_preregister(shadow)
            config = json.loads((shadow.run_dir / "preregister.json").read_text())["configuration"]
        config.update(provenance(), study_kind="energy_distance_ablation")
        path = a.run_dir / "preregister.json"
        if path.exists():
            if json.loads(path.read_text()).get("configuration") != config:
                raise ValueError("energy preregistration mismatch; use a separate out-dir")
        else:
            original_dump(path, dict(created_utc=datetime.now(timezone.utc).isoformat(), configuration=config))
        snapshots = a.run_dir / "source_snapshot"
        snapshots.mkdir(exist_ok=True)
        for name, digest in (("energy_study.py", SOURCE_SHA256), ("calibration_study.py", harness_sha)):
            contents = (HERE / name).read_bytes()
            if hashlib.sha256(contents).hexdigest() != digest:
                raise RuntimeError("energy study source changed during setup")
            destination = snapshots / name
            if destination.exists() and destination.read_bytes() != contents:
                raise ValueError("existing energy source snapshot differs")
            destination.write_bytes(contents)

    def targets(a, teacher, data):
        objective, metadata = reference_from_target_only(a, teacher, data, study, statistic)
        active.update(objective=objective, target_metadata=metadata)
        # The inherited loop passes this object only to the overridden loss.
        return {"energy_objective": objective}

    def loss(logits, labels, ids, targets, loss_kind):
        if loss_kind != "logodds_mse" or active["objective"] is None:
            raise RuntimeError("energy loss called before its frozen reference was prepared")
        if targets.get("energy_objective") is not active["objective"]:
            raise RuntimeError("energy objective reference changed")
        return active["objective"](logits, labels)

    def runtime_metadata():
        extra = provenance()
        if active["objective"] is not None:
            extra["energy_statistics"] = active["objective"].snapshot()
            extra["target_generation_seconds"] = active["target_metadata"]["target_generation_seconds"]
            extra["energy_reference_score_sha256"] = active["target_metadata"]["score_sha256"]
        return extra

    def save_last(destination, model, epoch, metadata):
        metadata.update(runtime_metadata())
        original_save_last(destination, model, epoch, metadata)

    def dump(path, value):
        if Path(path).name == "best.json":
            # Mutate the caller's selected metadata too, keeping summary equal.
            value.update(runtime_metadata())
        return original_dump(path, value)

    def schedule(*args, **kwargs):
        if active["objective"] is not None:
            active["objective"].reset()
        return original_schedule(*args, **kwargs)

    study.preregister, study.study_schedule = preregister, schedule
    study.study_targets, study.target_loss = targets, loss
    study.save_last, study.base.dump = save_last, dump
    return active


def self_test():
    torch.set_num_threads(1)
    # Point masses agree; separated point masses have energy 2*distance.
    x = torch.zeros(3, 2, dtype=torch.float64, requires_grad=True)
    y = torch.zeros(4, 2, dtype=torch.float64)
    assert energy_u_statistic(x, y).item() == 0.
    shifted = y + torch.tensor([3., 4.], dtype=torch.float64)
    assert energy_u_statistic(x, shifted).item() == 10.
    # Ordered off-diagonal distances for [0,2,5] sum to 20 over six pairs.
    assert torch.allclose(off_diagonal_mean_distances(torch.tensor([[0.], [2.], [5.]], dtype=torch.float64)), torch.tensor(20./6, dtype=torch.float64))
    # An unbiased estimate need not vanish/nonnegative on identical finite sets.
    two = torch.tensor([[0.], [1.]], dtype=torch.float64)
    assert energy_u_statistic(two, two).item() == -1.
    # Exact manual ordered-pair expression and finite gradients, including ties.
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    a = torch.randn(5, 2, generator=generator, dtype=torch.float64, requires_grad=True)
    b = torch.randn(7, 2, generator=generator, dtype=torch.float64)
    manual = (2 * torch.stack([(u-v).norm() for u in a for v in b]).mean()
              - torch.stack([(a[i]-a[j]).norm() for i in range(5) for j in range(5) if i != j]).mean()
              - torch.stack([(b[i]-b[j]).norm() for i in range(7) for j in range(7) if i != j]).mean())
    value = energy_u_statistic(a, b)
    assert torch.allclose(value, manual, atol=1e-13)
    value.backward()
    assert torch.isfinite(a.grad).all()
    energy_u_statistic(x, shifted).backward()
    assert torch.isfinite(x.grad).all()
    for bad in (torch.zeros(1, 2), torch.zeros(0, 2)):
        try:
            off_diagonal_mean_distances(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("small-n U-statistic was accepted")
    ref_logits = torch.tensor([[2., 0., -1.], [1., .5, 0.], [0., 2., 1.], [0., 3., 1.]], dtype=torch.float64)
    ref_labels = torch.tensor([0, 0, 1, 1])
    for statistic in STATISTICS:
        reference = scores_from_logits(ref_logits, ref_labels, statistic)
        objective = EnergyObjective(reference, ref_labels, statistic)
        logits = ref_logits.clone().requires_grad_(True)
        value = objective(logits, ref_labels)
        value.backward()
        assert torch.isfinite(logits.grad).all() and objective.snapshot()["used_class_terms"] == 2
        singleton = logits[:1].detach().requires_grad_(True)
        assert objective(singleton, ref_labels[:1]).item() == 0.
        objective(singleton, ref_labels[:1]).backward()
        assert torch.equal(singleton.grad, torch.zeros_like(singleton))
        assert objective.snapshot()["all_skipped_calls"] == 2
    # Poison all non-target data entries; reference construction must not read them.
    class TargetOnly(dict):
        def __getitem__(self, key):
            if key != "target":
                raise AssertionError(f"forbidden data access: {key}")
            return super().__getitem__(key)
    with tempfile.TemporaryDirectory(prefix="unlearning-energy-test-") as temporary:
        path = Path(temporary)
        def simple_dump(target, value):
            Path(target).write_text(json.dumps(value))
        fake_common = SimpleNamespace(sha256_arrays=lambda *arrays: hashlib.sha256(b''.join(np.asarray(v).tobytes() for v in arrays)).hexdigest())
        fake = SimpleNamespace(fixed=SimpleNamespace(clean_logits=lambda _model, images: images),
            common=fake_common, base=SimpleNamespace(dump=simple_dump, log=lambda *args, **kwargs: None))
        objective, metadata = reference_from_target_only(SimpleNamespace(run_dir=path), None,
            TargetOnly(target=(ref_logits, ref_labels)), fake, "joint")
        assert metadata["selection_used_for_targets"] is False and metadata["audit_predicted"] is False
        assert (path/"energy_reference.npz").exists() and len(objective.references) == 2
    # Exercise preregistration and checkpoint metadata without data or CUDA.
    study = load_harness()
    active = install_hooks(study, "joint")
    with tempfile.TemporaryDirectory(prefix="unlearning-energy-register-") as temporary:
        run = Path(temporary)
        args = SimpleNamespace(run_dir=run, seed=42, split_seed=42, schedules=["C"], lr=.003,
            forget_weight=.1, epochs=3, anchor_weight=.1, retain_subset=5000,
            target_kind="global_logodds", loss="logodds_mse", bn_mode="frozen")
        study.preregister(args)
        study.preregister(args)
        config = json.loads((run/"preregister.json").read_text())["configuration"]
        assert config["loss"] == LOSS_NAME and config["target_kind"] == TARGET_NAME
        assert config["energy_statistic"] == "joint"
        active.update(objective=EnergyObjective(scores_from_logits(ref_logits, ref_labels,"joint"), ref_labels,"joint"),
                      target_metadata={"target_generation_seconds": .2, "score_sha256": "test"})
        selected = {"epoch":1}
        study.base.dump(run/"best.json", selected)
        study.base.dump(run/"summary.json", {"selection":selected})
        best=json.loads((run/"best.json").read_text())
        summary=json.loads((run/"summary.json").read_text())
        assert best==summary["selection"] and best["loss"]==LOSS_NAME
    assert not torch.cuda.is_initialized()
    print("energy CPU checks passed: norms, U-statistic weights, finite/tied gradients, singleton handling, target-only access, honest metadata; CUDA uninitialized")


def main():
    if "--self-test" in sys.argv:
        if len(sys.argv) != 2:
            raise ValueError("--self-test takes no other arguments")
        self_test()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--statistic", choices=STATISTICS, default="joint")
    options, remaining = parser.parse_known_args()
    study = load_harness()
    install_hooks(study, options.statistic)
    sys.argv = [sys.argv[0], *remaining]
    study.main()


if __name__ == "__main__":
    main()
