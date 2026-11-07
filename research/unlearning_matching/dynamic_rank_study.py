"""Epoch-refreshed global quantile targets around the frozen calibration runner.

The only training-target change is to reassign the same frozen Original
held-out log-odds quantiles by CURRENT full-forget classwise rank at each epoch
boundary. The CLI and schedules A/B/C/D are inherited from calibration_study;
only --target-kind global_logodds --loss logodds_mse are accepted. The inherited
selection rule still chooses checkpoints, but selection predictions never
construct targets. No audit, official-test or Retrain prediction is loaded.

At an epoch boundary, barycentric quantile MSE has the same score gradient as
full empirical 1-D W2 squared, up to a score-independent residual. Assignment
is stale within an epoch: minibatch updates are NOT exact per-step full W2,
and this wrapper makes no unbiased-gradient or certified-deletion claim.
Use --self-test for CPU-only assignment and gradient checks without training.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "2")
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("dynamic_rank_calibration", HERE / "calibration_study.py")
study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(study)
fixed, base, common = study.fixed, study.base, study.common
SOURCE_SHA256 = common.sha256_file(__file__)
SCAFFOLD_SHA256 = common.sha256_file(HERE / "calibration_study.py")


class EpochRankTargets:
    """Pure CPU rank assignment; its inputs contain no selection predictions."""
    def __init__(self, forget_labels, frozen_target_scores, target_labels):
        self.labels = torch.as_tensor(forget_labels).detach().cpu().long().clone()
        reference = torch.as_tensor(frozen_target_scores).detach().cpu().clone()
        target_labels = torch.as_tensor(target_labels).detach().cpu().long()
        if self.labels.ndim != 1 or len(self.labels) == 0:
            raise ValueError("forget labels must be a nonempty vector")
        if reference.ndim != 1 or target_labels.shape != reference.shape or not torch.isfinite(reference).all():
            raise ValueError("target scores and labels must be matching finite vectors")
        self.rows, self.sorted_targets = {}, {}
        for cls in self.labels.unique(sorted=True).tolist():
            ids = torch.where(self.labels == cls)[0]
            ref = reference[target_labels == cls]
            if not len(ref):
                raise ValueError(f"missing held-out reference class {cls}")
            weights = fixed.overlap_transport_weights(len(ids), len(ref))
            self.rows[cls] = ids
            self.sorted_targets[cls] = (weights @ ref.double().sort().values)
        self.frozen_reference_sha256 = common.sha256_arrays(reference.numpy(), target_labels.numpy())

    def assign(self, current_forget_scores):
        current = torch.as_tensor(current_forget_scores).detach().cpu()
        if current.shape != self.labels.shape or not current.is_floating_point() or not torch.isfinite(current).all():
            raise ValueError("current forget scores must be a matching finite vector")
        assigned = torch.empty_like(current)
        ranks = torch.empty(len(current), dtype=torch.long)
        for cls, ids in self.rows.items():
            ordered = ids[torch.argsort(current[ids], stable=True)]
            assigned[ordered] = self.sorted_targets[cls].to(current.dtype)
            ranks[ordered] = torch.arange(len(ordered))
        return assigned, ranks

    def update(self, shared_targets, current_forget_scores):
        assigned, ranks = self.assign(current_forget_scores)
        # Retain both the dict and tensor identities held by the inherited loop.
        shared_targets["logodds"].copy_(assigned.to(shared_targets["logodds"]))
        return assigned, ranks


_shared_targets = None
_original_study_targets = study.study_targets


def study_targets(a, teacher, data):
    global _shared_targets
    _shared_targets = _original_study_targets(a, teacher, data)
    return _shared_targets


class DynamicDiagnostics(study.Diagnostics):
    def __init__(self, a, data, original_model):
        self.current_forget_scores = None
        super().__init__(a, data, original_model)
        # The superclass initialization queried the frozen ORIGINAL teacher.
        # Copy only its held-out target scores; never use its selection fields.
        self.refresher = EpochRankTargets(data["forget"][1],
            self.original["target"]["logodds"], data["target"][1])
        self.original_forget_scores = self.original["forget"]["logodds"].clone()
        self.previous_scores, self.previous_ranks = self.refresher.assign(self.original_forget_scores)

    def distributions(self, model):
        values = super().distributions(model)
        self.current_forget_scores = values["forget"]["logodds"].detach().cpu().clone()
        return values

    def record(self, model, destination, epoch, stage):
        record = super().record(model, destination, epoch, stage)
        if stage not in ("initial", "epoch_end", "after_retain"):
            return record
        history = destination / "dynamic_target_history.jsonl"
        if stage == "initial":
            history.unlink(missing_ok=True)
            self.previous_scores, self.previous_ranks = self.refresher.assign(self.original_forget_scores)
        assigned, ranks = self.refresher.update(_shared_targets, self.current_forget_scores)
        artifact = destination / f"dynamic_targets_epoch{epoch:03d}_{stage}.npz"
        np.savez_compressed(artifact, current_forget_logodds=self.current_forget_scores.numpy(),
                            target_logodds=assigned.numpy(), classwise_ranks=ranks.numpy())
        diagnostic = dict(epoch=epoch, stage=stage, applies_to_epoch=epoch + 1,
            refresh_used_selection_predictions=False,
            frozen_reference_sha256=self.refresher.frozen_reference_sha256,
            artifact=artifact.name, artifact_sha256=common.sha256_file(artifact),
            targets_sha256=common.sha256_arrays(assigned.numpy()),
            ranks_sha256=common.sha256_arrays(ranks.numpy()),
            rank_changed_fraction=float((ranks != self.previous_ranks).double().mean()),
            target_changed_fraction=float((assigned != self.previous_scores).double().mean()),
            mean_absolute_rank_change=float((ranks - self.previous_ranks).abs().double().mean()),
            mean_absolute_target_change=float((assigned - self.previous_scores).abs().double().mean()))
        with history.open("a") as stream:
            stream.write(json.dumps(diagnostic) + "\n")
        self.previous_scores, self.previous_ranks = assigned.clone(), ranks.clone()
        base.log("dynamic_target_refresh", schedule=destination.name, **diagnostic)
        return record


def preregister(a):
    if a.target_kind != "global_logodds" or a.loss != "logodds_mse":
        raise ValueError("dynamic rank supports only global_logodds / logodds_mse")
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()}
    configuration = dict(schema=1, method="epoch_dynamic_global_quantile_logodds",
        source_sha256=SOURCE_SHA256, inherited_scaffold_sha256=SCAFFOLD_SHA256,
        source_manifest={name: common.sha256_file(HERE / name) for name in
            ("dynamic_rank_study.py", "calibration_study.py", "fixed_quantile.py", "run_experiments.py")},
        common_sha256=common.sha256_file(base.TASK / "scripts/common.py"),
        train_sha256=common.SHA256_TRAIN["cifar10"], original_sha256=common.SHA256_ORIGINAL["cifar10"],
        arguments=arguments, momentum=.9, weight_decay=5e-4, batch_size=256,
        anchor_temperature=2., forget_gradient_clip_norm=5.,
        target="frozen Original held-out logodds barycentric quantiles; current full-forget classwise rank",
        refresh="initial and after each complete epoch, assignment fixed within epoch",
        stale_unused_target_fields=["margin", "centered_logits", "log_probs"],
        only_logodds_targets_enter_loss=True, target_generation_uses_selection=False,
        audit_predictions_computed=False, official_test_read=False, retrain_read=False,
        selection_rule=dict(retain_accuracy_loss_max_pp=.5, selection_accuracy_loss_max_pp=1.,
            objective="max(abs(error_forget-error_selection), pooled_confidence_KS); probability units",
            fallback="Original if no feasible epoch endpoint"),
        torch_version=torch.__version__, numpy_version=np.__version__)
    path = a.run_dir / "preregister.json"
    if path.exists():
        if json.loads(path.read_text()).get("configuration") != configuration:
            raise ValueError("dynamic preregistration mismatch; use a separate out-dir")
    else:
        base.dump(path, dict(created_utc=datetime.now(timezone.utc).isoformat(), configuration=configuration))
    # Preserve exact source snapshots alongside their hashes for this run.
    snapshots = a.run_dir / "source_snapshot"
    snapshots.mkdir(exist_ok=True)
    for name, digest in configuration["source_manifest"].items():
        payload = (HERE / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise RuntimeError(f"source changed during snapshot: {name}")
        destination = snapshots / name
        if destination.exists() and destination.read_bytes() != payload:
            raise ValueError(f"existing source snapshot differs: {name}")
        destination.write_bytes(payload)


def self_test():
    torch.set_num_threads(1)
    labels = torch.tensor([0, 0, 1, 1])
    scores = torch.tensor([3., 1., 9., 2.], dtype=torch.float64)
    updater = EpochRankTargets(labels, scores, labels)
    assigned, ranks = updater.assign(scores)
    assert torch.equal(assigned, scores)
    permutation = torch.tensor([3, 0, 2, 1])
    perm_updater = EpochRankTargets(labels[permutation], scores, labels)
    assert torch.equal(perm_updater.assign(scores[permutation])[0], scores[permutation])
    changed = torch.tensor([0., 4., 1., 8.], dtype=torch.float64)
    shared = {"logodds": assigned.clone(), "selection": object(), "margin": object()}
    original_pointer = shared["logodds"].data_ptr()
    updated, _ = updater.update(shared, changed)
    assert updated.tolist() == [1., 3., 2., 9.]
    assert shared["logodds"].data_ptr() == original_pointer
    assert torch.equal(shared["logodds"], updated)
    tied, tied_ranks = updater.assign(torch.zeros(4, dtype=torch.float64))
    assert tied.tolist() == [1., 3., 2., 9.] and tied_ranks.tolist() == [0, 1, 0, 1]
    # Unequal-size exact W2 score gradients agree at the refresh boundary.
    x = torch.tensor([1.1, -2., 3., .4, 2.], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor([-3., 0., 1.7], dtype=torch.float64)
    unequal = EpochRankTargets(torch.zeros(5, dtype=torch.long), reference, torch.zeros(3, dtype=torch.long))
    target, _ = unequal.assign(x)
    weights = fixed.overlap_transport_weights(5, 3)
    exact = (weights * (x.sort().values[:, None] - reference.sort().values[None, :]).square()).sum() / 5
    gradient_exact = torch.autograd.grad(exact, x, retain_graph=True)[0]
    gradient_assigned = torch.autograd.grad((x - target).square().mean(), x)[0]
    assert torch.allclose(gradient_exact, gradient_assigned, atol=1e-12)
    print("dynamic rank CPU checks passed: identity, permutation, ties, shared update, boundary W2 gradient; no selection input")


def main():
    if "--self-test" in sys.argv:
        if len(sys.argv) != 2:
            raise ValueError("--self-test takes no training arguments")
        self_test()
        return
    study.preregister = preregister
    study.study_targets = study_targets
    study.Diagnostics = DynamicDiagnostics
    study.SOURCE_SHA256 = SOURCE_SHA256
    study.__doc__ = __doc__
    study.main()


if __name__ == "__main__":
    main()
