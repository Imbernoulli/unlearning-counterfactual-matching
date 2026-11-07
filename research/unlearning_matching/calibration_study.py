"""Calibration-only study of forgetting/retention update schedules.

Training and selection diagnostics use hash-verified training arrays only.
This script has no official-test evaluation and does not read Retrain models
or any earlier result files.  Fixed global class-conditional quantile targets
are estimated once from Original and the target half of validation.  The
independent calibration half is split by class into 1250 selection images
and 1250 audit images.  Only selection predictions are computed here; audit
indices are recorded for a separate final audit program.

A: complete forget pass, then complete retain pass (diagnose both boundaries).
B: place every forget batch uniformly across a complete retain pass, taking
   one joint optimizer step at paired batches; keep the Original KL anchor.
C: B without the Original KL anchor.  A/B/C see exactly one complete pass of
   each dataset per epoch.  Joint steps clip the forget gradient before the
   retain gradient is added; this matches A's forget-only clipping rule.
D: pair each forget batch with an equally sized random retain batch, no KL
   anchor.  D sees 4500 retain samples per epoch, rather than 40500; it is a
   separately budgeted candidate, not an equal-compute ablation.

Every epoch atomically replaces last.pt.  Among feasible epoch endpoints,
best.pt minimizes max(abs(forget error - selection error), pooled confidence
KS), using probability units; ties prefer the earlier epoch.  Feasibility
requires retain accuracy loss <=0.5 percentage point and selection accuracy
loss <=1 point relative to Original.  If no endpoint is feasible, best.pt is
Original.  Selection never uses audit or official test predictions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
import shutil
from pathlib import Path
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "2")
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("calibration_fixed_targets", HERE / "fixed_quantile.py")
fixed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixed)
base, common = fixed.base, fixed.common
SOURCE_SHA256 = common.sha256_file(__file__)
QUANTILES = torch.tensor([0., .01, .025, .05, .1, .25, .5, .75, .9, .95, .975, .99, 1.])


def stop_requested(a):
    paths = [a.out_dir / "STOP", Path("/root/study/out/STOP")]
    if a.stop_file is not None:
        paths.append(a.stop_file)
    for path in paths:
        try:
            exists = path.exists()
        except PermissionError:
            # /root/study is a remote default; local unprivileged runs may not
            # traverse /root.  Their explicit/out-dir sentinels still apply.
            continue
        if exists:
            base.log("graceful_stop_requested", stop_file=str(path))
            return True
    return False


def bn_buffers(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.named_buffers()
            if name.endswith(("running_mean", "running_var", "num_batches_tracked"))}


def confidence_ks(x, y):
    """Exact two-sample empirical KS, including repeated confidence values."""
    x, y = x.detach().cpu().double().sort().values, y.detach().cpu().double().sort().values
    if not len(x) or not len(y):
        raise ValueError("KS requires nonempty samples")
    points = torch.cat((x, y)).sort().values
    cdf_x = torch.searchsorted(x, points, right=True).double() / len(x)
    cdf_y = torch.searchsorted(y, points, right=True).double() / len(y)
    return float((cdf_x - cdf_y).abs().max())


def preregister(a):
    config = dict(
        schema=1, source_sha256=SOURCE_SHA256,
        frozen_runner_sha256=common.sha256_file(HERE / "run_experiments.py"),
        fixed_target_source_sha256=common.sha256_file(HERE / "fixed_quantile.py"),
        train_sha256=common.SHA256_TRAIN["cifar10"],
        original_sha256=common.SHA256_ORIGINAL["cifar10"],
        seed=a.seed, split_seed=a.split_seed, schedules=a.schedules,
        lr=a.lr, forget_weight=a.forget_weight, epochs=a.epochs,
        anchor_weight_requested=a.anchor_weight,
        effective_anchor={name: (a.anchor_weight if name in ("A", "B") else 0.) for name in a.schedules},
        momentum=.9, weight_decay=5e-4, batch_size=256, anchor_temperature=2.,
        forget_gradient_clip_norm=5., retain_subset=a.retain_subset,
        retain_subset_seed=7312026, target_kind=a.target_kind, loss=a.loss,
        rank_donor_source_sha256=(common.sha256_file(HERE / "rank_donor.py")
                                 if a.target_kind == "rank_donor" else None),
        network="all parameters, FP32", bn_mode=a.bn_mode,
        selection_split_seed=20261107, selection_size=1250, audit_size=1250,
        selection_rule=dict(retain_accuracy_loss_max_pp=.5, selection_accuracy_loss_max_pp=1.,
            objective="max(abs(error_forget-error_selection), pooled_confidence_KS); probability units",
            within_run_tie="earlier epoch", cross_run_tie="earlier epoch, smaller lr, id",
            fallback="Original if no feasible epoch endpoint"),
        calibration_used_for_gradients=False, audit_predictions_computed=False,
        budgets={name: dict(forget_samples_per_epoch=4500,
                          retain_samples_per_epoch=4500 if name == "D" else 40500)
                 for name in a.schedules},
        selection="preregistered selection-only objective; best.pt plus last.pt retained",
        torch_version=torch.__version__, numpy_version=np.__version__,
        sklearn_version=importlib.metadata.version("scikit-learn"))
    path = a.run_dir / "preregister.json"
    if path.exists():
        if json.loads(path.read_text()).get("configuration") != config:
            raise ValueError("preregistration mismatch; use a separate out-dir")
    else:
        base.dump(path, dict(created_utc=datetime.now(timezone.utc).isoformat(), configuration=config))


class Diagnostics:
    def __init__(self, a, data, original_model):
        self.a, self.data = a, {}
        for name in ("forget", "target", "selection"):
            self.data[name] = data[name]
        positions = np.random.RandomState(7312026).choice(len(data["retain"][1]), a.retain_subset, replace=False)
        pos = torch.from_numpy(np.sort(positions)).cuda()
        self.data["retain_subset"] = data["retain"][0][pos], data["retain"][1][pos]
        self.original_bn = bn_buffers(original_model)
        np.savez(a.run_dir / "original_bn_buffers.npz",
                 **{name: value.numpy() for name, value in self.original_bn.items()})
        self.original = self.distributions(original_model)
        self.original_summary = self.summarize(self.original)
        base.dump(a.run_dir / "original_diagnostics.json", self.original_summary)
        self.save_distributions(a.run_dir / "original_distributions.npz", self.original)

    @torch.no_grad()
    def distributions(self, model):
        was_training = model.training
        out = {}
        for name, (xx, yy) in self.data.items():
            z = fixed.clean_logits(model, xx)
            labels = yy.cpu()
            score, margin = base.logodds_margin(z, labels)
            confidence = F.softmax(z, dim=1).gather(1, labels[:, None]).squeeze(1)
            out[name] = dict(logodds=score, margin=margin, confidence=confidence,
                             error=z.argmax(1).ne(labels), labels=labels)
        model.train(was_training)
        return out

    @staticmethod
    def save_distributions(path, distributions):
        np.savez_compressed(path, **{f"{name}_{key}": val.numpy()
                                  for name, dist in distributions.items() for key, val in dist.items()})

    @staticmethod
    def summarize(distributions):
        summary = {}
        for name, dist in distributions.items():
            statistics = {}
            for cls in ("all", *range(10)):
                mask = torch.ones(len(dist["labels"]), dtype=torch.bool) if cls == "all" else dist["labels"] == cls
                error = float(100 * dist["error"][mask].float().mean())
                statistics[str(cls)] = dict(
                    n=int(mask.sum()), error_pct=error, accuracy_pct=100 - error,
                    confidence_mean=float(dist["confidence"][mask].mean()),
                    quantile_levels=QUANTILES.tolist(),
                    **{f"{key}_quantiles": torch.quantile(dist[key][mask], QUANTILES).tolist()
                       for key in ("confidence", "logodds", "margin")})
            summary[name] = statistics
        return summary

    def bn_delta(self, model):
        current = bn_buffers(model)
        if set(current) != set(self.original_bn):
            raise ValueError("BatchNorm buffer names changed")
        return max(float((current[name].double() - reference.double()).abs().max())
                   for name, reference in self.original_bn.items())

    def record(self, model, destination, epoch, stage):
        start = time.monotonic()
        values = self.distributions(model)
        summary = self.summarize(values)
        ff, cc = values["forget"], values["selection"]
        original_cal = self.original["selection"]
        classwise, frozen_classwise = {}, {}
        for c in range(10):
            cf, cv = ff["labels"] == c, cc["labels"] == c
            original_cv = original_cal["labels"] == c
            classwise[str(c)] = confidence_ks(ff["confidence"][cf], cc["confidence"][cv])
            frozen_classwise[str(c)] = confidence_ks(ff["confidence"][cf], original_cal["confidence"][original_cv])
        err_gap = summary["forget"]["all"]["error_pct"] - summary["selection"]["all"]["error_pct"]
        metrics = dict(
            forget_selection_confidence_ks=confidence_ks(ff["confidence"], cc["confidence"]),
            forget_original_selection_confidence_ks=confidence_ks(ff["confidence"], original_cal["confidence"]),
            classwise_confidence_ks=classwise,
            classwise_original_selection_confidence_ks=frozen_classwise,
            mean_classwise_confidence_ks=float(np.mean(list(classwise.values()))),
            worst_classwise_confidence_ks=max(classwise.values()),
            forget_selection_error_gap_pp=err_gap,
            absolute_forget_selection_error_gap_pp=abs(err_gap),
            forget_original_selection_error_gap_pp=(summary["forget"]["all"]["error_pct"]
                - self.original_summary["selection"]["all"]["error_pct"]),
            selection_accuracy_drift_pp=(summary["selection"]["all"]["accuracy_pct"]
                - self.original_summary["selection"]["all"]["accuracy_pct"]),
            retain_subset_accuracy_drift_pp=(summary["retain_subset"]["all"]["accuracy_pct"]
                - self.original_summary["retain_subset"]["all"]["accuracy_pct"]))
        metrics["bn_buffer_max_delta"] = self.bn_delta(model)
        metrics["bn_buffers_equal_original"] = metrics["bn_buffer_max_delta"] == 0.
        if self.a.bn_mode == "frozen" and not metrics["bn_buffers_equal_original"]:
            raise RuntimeError("frozen BatchNorm buffers changed")
        metrics["feasible"] = (metrics["retain_subset_accuracy_drift_pp"] >= -.5 - 1e-8
                               and metrics["selection_accuracy_drift_pp"] >= -1. - 1e-8)
        metrics["selection_J"] = max(metrics["absolute_forget_selection_error_gap_pp"] / 100.,
                                     metrics["forget_selection_confidence_ks"])
        distribution_path = destination / f"epoch{epoch:03d}_{stage}_distributions.npz"
        self.save_distributions(distribution_path, values)
        record = dict(epoch=epoch, stage=stage, metrics=metrics, distributions_summary=summary,
                      distributions_file=distribution_path.name, diagnostic_seconds=time.monotonic() - start)
        with (destination / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        base.log("calibration_diagnostics", schedule=destination.name, epoch=epoch, stage=stage, **metrics)
        return record


def paired_positions(n_retain_batches, n_forget_batches):
    """Distinct positions spanning the complete retain pass, including its end."""
    if n_forget_batches > n_retain_batches:
        raise ValueError("complete-retain schedules need at least as many retain batches")
    return {int(round(float(p))): i for i, p in enumerate(np.linspace(0, n_retain_batches - 1, n_forget_batches))}


def study_targets(a, teacher, data):
    """Fixed calibration-study targets; calibration examples never enter targets."""
    path = a.run_dir / "calibration_targets.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as cache:
            return {key: torch.from_numpy(cache[key]).cuda()
                    for key in ("logodds", "margin", "centered_logits", "log_probs")}
    if a.target_kind == "global_logodds":
        score, logq = fixed.fixed_targets(a, teacher, data)
        score, logq = score.cpu(), logq.cpu()
        labels = data["forget"][1].cpu()
        _, margin = base.logodds_margin(logq, labels)
        values = dict(logodds=score, margin=margin,
                      centered_logits=logq - logq.mean(dim=1, keepdim=True), log_probs=logq)
    else:
        builder = base.module_at("calibration_rank_donor", HERE / "rank_donor.py")
        zf = fixed.clean_logits(teacher, data["forget"][0])
        zv = fixed.clean_logits(teacher, data["target"][0])
        output = builder.build_rank_donor_targets(zf, data["forget"][1].cpu(),
                                                 zv, data["target"][1].cpu())
        values = dict(logodds=output["target_logodds"], margin=output["target_margin"],
                      centered_logits=output["target_centered_logits"], log_probs=output["target_log_probs"])
        np.savez(a.run_dir / "rank_donor_assignment.npz", donor_ids=output["donor_ids"].cpu().numpy())
    np.savez(path, **{key: val.detach().cpu().numpy() for key, val in values.items()})
    return {key: val.cuda() for key, val in values.items()}


def target_loss(logits, labels, ids, targets, loss_kind):
    if loss_kind == "margin_ascent":
        # Matched control: minimize mean true-class logodds without a target.
        score, _ = base.logodds_margin(logits, labels)
        return score.mean()
    if loss_kind == "kl":
        return F.kl_div(F.log_softmax(logits, dim=1), targets["log_probs"][ids],
                        reduction="batchmean", log_target=True)
    if loss_kind == "centered_logits_mse":
        centered = logits - logits.mean(dim=1, keepdim=True)
        return F.mse_loss(centered, targets["centered_logits"][ids])
    score, margin = base.logodds_margin(logits, labels)
    loss = F.mse_loss(score, targets["logodds"][ids])
    if loss_kind == "scores_mse":
        loss = .5 * (loss + F.mse_loss(margin, targets["margin"][ids]))
    return loss


def forget_gradient(model, optimizer, xx, yy, ids, targets, weight, loss_kind):
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    z = model(xx[ids].float() / 255.)
    loss = target_loss(z, yy[ids], ids, targets, loss_kind)
    (weight * loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
    return loss.detach()


def retain_gradient(model, teacher, xx, yy, ids, anchor_weight, bn_mode):
    # Frozen mode still augments retain images and updates every parameter.
    model.train(bn_mode != "frozen")
    xb = common.gpu_augment(xx[ids], True)
    z = model(xb)
    loss = F.cross_entropy(z, yy[ids])
    if anchor_weight:
        with torch.no_grad():
            q = F.softmax(teacher(xb) / 2., dim=1)
        loss = loss + anchor_weight * 4. * F.kl_div(F.log_softmax(z / 2., dim=1), q, reduction="batchmean")
    loss.backward()
    return loss.detach()


def save_last(destination, model, epoch, metadata):
    path = destination / "last.pt"
    temporary = destination / "last.pt.tmp"
    torch.save({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, temporary)
    os.replace(temporary, path)
    base.dump(destination / "last.json", dict(epoch=epoch, checkpoint=path.name, **metadata))
    np.savez(destination / "last_bn_buffers.npz",
             **{name: value.numpy() for name, value in bn_buffers(model).items()})


def study_schedule(a, name, data, teacher, targets, diagnostics, state):
    if stop_requested(a):
        return
    destination = a.run_dir / name
    destination.mkdir(exist_ok=True)
    if (destination / "summary.json").exists():
        base.log("skip_calibration_schedule", schedule=name)
        return
    # A restarted incomplete schedule discards its own trace, but no selected file.
    (destination / "history.jsonl").unlink(missing_ok=True)
    model = base.model_from(state)
    common.set_determinism(a.seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=.9, weight_decay=5e-4)
    anchor = a.anchor_weight if name in ("A", "B") else 0.
    xf, yf = data["forget"]
    xr, yr = data["retain"]
    total_training_seconds = 0.
    best_selection = None
    start = time.monotonic()
    initial = diagnostics.record(model, destination, 0, "initial")
    for epoch in range(1, a.epochs + 1):
        if stop_requested(a):
            return
        if name == "A":
            diagnostics.record(model, destination, epoch, "before_forget")
        torch.cuda.synchronize()
        begin_training = time.monotonic()
        forget_order = torch.randperm(len(yf), device="cuda")
        forget_batches = list(forget_order.split(256))
        retain_order = torch.randperm(len(yr), device="cuda")
        forget_losses, retain_losses = [], []
        if name == "A":
            for ids in forget_batches:
                forget_losses.append(forget_gradient(model, optimizer, xf, yf, ids, targets, a.forget_weight, a.loss))
                optimizer.step()
            torch.cuda.synchronize()
            total_training_seconds += time.monotonic() - begin_training
            diagnostics.record(model, destination, epoch, "after_forget")
            begin_training = time.monotonic()
            for ids in retain_order.split(256):
                optimizer.zero_grad(set_to_none=True)
                retain_losses.append(retain_gradient(model, teacher, xr, yr, ids, anchor, a.bn_mode))
                optimizer.step()
        else:
            if name == "D":
                retain_batches = list(retain_order[:len(yf)].split(256))
                locations = {i: i for i in range(len(forget_batches))}
            else:
                retain_batches = list(retain_order.split(256))
                locations = paired_positions(len(retain_batches), len(forget_batches))
            for step, retain_ids in enumerate(retain_batches):
                if step in locations:
                    forget_ids = forget_batches[locations[step]]
                    forget_losses.append(forget_gradient(model, optimizer, xf, yf, forget_ids, targets, a.forget_weight, a.loss))
                else:
                    optimizer.zero_grad(set_to_none=True)
                retain_losses.append(retain_gradient(model, teacher, xr, yr, retain_ids, anchor, a.bn_mode))
                optimizer.step()
        torch.cuda.synchronize()
        total_training_seconds += time.monotonic() - begin_training
        metadata = dict(schedule=name, seed=a.seed, split_seed=a.split_seed,
            lr=a.lr, forget_weight=a.forget_weight, anchor_weight=anchor,
            target_kind=a.target_kind, loss=a.loss, bn_mode=a.bn_mode,
            train_seconds=total_training_seconds, wall_seconds=time.monotonic() - start,
            forget_samples_per_epoch=len(yf), retain_samples_per_epoch=len(yf) if name == "D" else len(yr),
            source_sha256=SOURCE_SHA256, calibration_only=True,
            bn_buffer_max_delta=diagnostics.bn_delta(model))
        save_last(destination, model, epoch, metadata)
        record = diagnostics.record(model, destination, epoch, "after_retain" if name == "A" else "epoch_end")
        base.log("calibration_epoch", epoch=epoch,
                 forget_loss=float(torch.stack(forget_losses).mean()),
                 retain_loss=float(torch.stack(retain_losses).mean()), **metadata)
        if record["metrics"]["feasible"] and (best_selection is None or
                record["metrics"]["selection_J"] < best_selection["selection_J"]):
            best_selection = dict(epoch=epoch, selection_J=record["metrics"]["selection_J"],
                feasible=True, is_original_fallback=False, lr=a.lr, schedule=name,
                target_kind=a.target_kind, loss=a.loss, bn_mode=a.bn_mode,
                retain_subset_accuracy_drift_pp=record["metrics"]["retain_subset_accuracy_drift_pp"],
                selection_accuracy_drift_pp=record["metrics"]["selection_accuracy_drift_pp"],
                checkpoint="best.pt")
            shutil.copyfile(destination / "last.pt", destination / "best.pt.tmp")
            os.replace(destination / "best.pt.tmp", destination / "best.pt")
            base.dump(destination / "best.json", best_selection)
        base.dump(destination / "latest_calibration.json", record)
    if best_selection is None:
        torch.save(state, destination / "best.pt.tmp")
        os.replace(destination / "best.pt.tmp", destination / "best.pt")
        best_selection = dict(epoch=0, selection_J=initial["metrics"]["selection_J"],
            feasible=True, is_original_fallback=True, reason="no feasible trained epoch endpoint",
            lr=a.lr, schedule=name, checkpoint="best.pt")
        base.dump(destination / "best.json", best_selection)
    base.dump(destination / "summary.json", dict(
        **metadata, epochs=a.epochs, final_calibration=record["metrics"],
        initial_calibration=initial["metrics"], checkpoint="last.pt",
        selection=best_selection,
        budget_note=("4500 retain examples/epoch: separately budgeted candidate"
                     if name == "D" else "40500 retain + 4500 forget examples/epoch")))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--base-runs-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--original", default="/srv/home/bohanlyu/data_mlsbench/unlearn-lab/original/cifar10_resnet18_original.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--gpu", default=None)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--lr", type=float, default=.003)
    p.add_argument("--forget-weight", type=float, default=.1)
    p.add_argument("--anchor-weight", type=float, default=.1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--schedules", default="A,B,C")
    p.add_argument("--target-kind", choices=["global_logodds", "rank_donor"], default="global_logodds")
    p.add_argument("--loss", choices=["logodds_mse", "scores_mse", "centered_logits_mse", "kl", "margin_ascent"],
                   default="logodds_mse")
    p.add_argument("--retain-subset", type=int, default=5000)
    p.add_argument("--bn-mode", choices=["inherited", "frozen"], default="inherited")
    p.add_argument("--stop-file", type=Path, default=None)
    a = p.parse_args()
    a.schedules = [name.strip().upper() for name in a.schedules.split(",")]
    if not a.schedules or any(name not in ("A", "B", "C", "D") for name in a.schedules):
        raise ValueError("--schedules must list A/B/C/D")
    if len(set(a.schedules)) != len(a.schedules):
        raise ValueError("duplicate schedules")
    if a.epochs < 1 or a.lr <= 0 or a.forget_weight < 0 or a.anchor_weight < 0:
        raise ValueError("invalid training hyperparameters")
    if not 1 <= a.retain_subset <= 40500:
        raise ValueError("invalid retain subset size")
    if a.out_dir.resolve() == a.base_runs_dir.resolve():
        raise ValueError("calibration outputs must be separate from base runs")
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    stem = f"seed{a.seed}_split{a.split_seed}"
    a.run_dir, a.base_run_dir = a.out_dir / stem, a.base_runs_dir / stem
    a.run_dir.mkdir(parents=True, exist_ok=True)
    preregister(a)
    if stop_requested(a):
        return
    x, y = base.train_arrays(a)
    indices = fixed.matching_splits(a, y)
    state = base.original_state(a)
    # The canonical calibration pool is never predicted as a whole.  Its
    # stratified audit half is only recorded for an independent later phase.
    rng = np.random.RandomState(20261107)
    selection, audit = [], []
    for cls in range(10):
        pool = indices["calibration"][y[indices["calibration"]] == cls]
        order = rng.permutation(pool)
        selection.extend(int(i) for i in order[:125])
        audit.extend(int(i) for i in order[125:])
    selection, audit = np.asarray(sorted(selection)), np.asarray(sorted(audit))
    assert len(selection) == len(audit) == 1250 and not np.intersect1d(selection, audit).size
    base.dump(a.run_dir / "selection_audit_split.json", dict(
        seed=20261107, selection=selection.tolist(), audit=audit.tolist(), audit_predicted=False))
    training_indices = {key: value for key, value in indices.items() if key != "calibration"}
    training_indices["selection"] = selection
    data = base.gpu_sets(x, y, training_indices)
    teacher = base.model_from(state).eval().requires_grad_(False)
    targets = study_targets(a, teacher, data)
    diagnostics = Diagnostics(a, data, teacher)
    for name in a.schedules:
        if stop_requested(a):
            break
        study_schedule(a, name, data, teacher, targets, diagnostics, state)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
