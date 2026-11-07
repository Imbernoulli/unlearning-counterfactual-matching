"""Predeclared global-quantile targets and information projection for unlearning.

This is an independent next experiment, not a change to the frozen baseline
runner.  Training reads only the official training arrays, split manifests,
and the Original checkpoint.  It never reads previous metric/result files,
Retrain predictions, or test data.  Both variants use the same fixed targets.

For each class, uniformly weighted forget points are ranked by Original
true-class log-odds.  Their target scores are exact bin averages of the
held-out empirical quantile function (the barycentric 1-D OT projection).
Given target u_i=sigmoid(s_i), the unique minimizer of KL(q_i || p0_i) subject
to q_i[y_i]=u_i is q_i[k!=y_i]=(1-u_i)*p0_i[k]/(1-p0_i[y_i]).  This follows by
the KL chain rule into Bernoulli KL and conditional non-true-class KL.
For a fixed multiset of u_i, the minimum-sum-KL assignment pairs sorted u_i
with sorted Original log-odds: the permutation-dependent term is
-sum_i u_pi(i)*logit(p0_i[y_i]), so the rearrangement inequality applies.
Assumptions: positive Original probabilities, a common class, uniform point
weights, and a fixed target multiset.  These are output-space optimality
claims; they imply neither minimum parameter displacement nor true deletion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "2")
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("frozen_unlearning_study", HERE / "run_experiments.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
common = base.common
CODE_SHA256 = common.sha256_file(__file__)
VARIANTS = {"kl": "global_quantile_iprojection", "logodds_mse": "global_quantile_logodds_mse"}


def overlap_transport_weights(n: int, m: int) -> torch.Tensor:
    """Conditional target-bin weights per uniform source bin; row sums are one."""
    if n <= 0 or m <= 0:
        raise ValueError("both empirical distributions must be nonempty")
    left_source = torch.arange(n, dtype=torch.float64)[:, None] / n
    right_source = (torch.arange(n, dtype=torch.float64)[:, None] + 1) / n
    left_target = torch.arange(m, dtype=torch.float64)[None, :] / m
    right_target = (torch.arange(m, dtype=torch.float64)[None, :] + 1) / m
    return n * (torch.minimum(right_source, right_target)
                - torch.maximum(left_source, left_target)).clamp_min(0.)


def barycentric_quantile_targets(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Map exact reference quantile-bin averages back to the original point order."""
    if source.ndim != 1 or reference.ndim != 1:
        raise ValueError("expected scalar scores")
    source_cpu, reference_cpu = source.detach().cpu(), reference.detach().cpu()
    order = torch.argsort(source_cpu, stable=True)
    target_sorted = overlap_transport_weights(len(source_cpu), len(reference_cpu)) @ reference_cpu.double().sort().values
    target = torch.empty(len(source_cpu), dtype=torch.float64)
    target[order] = target_sorted
    return target.to(device=source.device, dtype=source.dtype)


def iprojection_log_probs(original_logits: torch.Tensor, labels: torch.Tensor,
                          target_logodds: torch.Tensor) -> torch.Tensor:
    """Stable log q with prescribed true-class mass and Original other-class ratios."""
    s = target_logodds.to(original_logits).reshape(-1)
    other = original_logits.masked_fill(F.one_hot(labels, original_logits.shape[1]).bool(), -torch.inf)
    logq = F.logsigmoid(-s)[:, None] + F.log_softmax(other, dim=1)
    return logq.scatter(1, labels[:, None], F.logsigmoid(s)[:, None])


def preregister(a):
    configuration = dict(
        schema=1, code_sha256=CODE_SHA256,
        frozen_runner_sha256=common.sha256_file(HERE / "run_experiments.py"),
        common_sha256=common.sha256_file(base.TASK / "scripts/common.py"),
        train_sha256=common.SHA256_TRAIN["cifar10"], original_sha256=common.SHA256_ORIGINAL["cifar10"],
        seed=a.seed, split_seed=a.split_seed, planned_variants=VARIANTS,
        epochs=10, lr=.003, momentum=.9, weight_decay=5e-4, batch_size=256,
        forget_weight=.1, retain_anchor_weight=.1, anchor_temperature=2.,
        forget_gradient_clip_norm=5., network="all parameters; FP32",
        target="fixed, class-conditional global OT barycentric logodds; Original on 2500 held-out images",
        target_rank="Original clean true-class logodds of ALL 4500 forget points, separately by class",
        training_order="per epoch: clean eval-mode forget pass, augmented train-mode retain pass",
        selection="both variants specified before new test inspection; no test or Retrain-target tuning",
        torch_version=torch.__version__, numpy_version=np.__version__,
        sklearn_version=importlib.metadata.version("scikit-learn"))
    path = a.run_dir / "preregister.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("configuration") != configuration:
            raise ValueError("preregistration mismatch; use a separate out-dir")
    else:
        base.dump(path, dict(created_utc=datetime.now(timezone.utc).isoformat(), configuration=configuration))


def matching_splits(a, y):
    idx = base.splits(a, y)
    own = json.loads((a.run_dir / "splits.json").read_text())
    prior_path = a.base_run_dir / "splits.json"
    if not prior_path.exists() or json.loads(prior_path.read_text()) != own:
        raise ValueError("paired baseline split manifest missing or different")
    return idx


@torch.no_grad()
def clean_logits(model, images, batch_size=256):
    model.eval()
    return torch.cat([model(images[i:i + batch_size].float() / 255.).cpu()
                      for i in range(0, len(images), batch_size)])


def fixed_targets(a, teacher, data):
    path = a.run_dir / "fixed_targets.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            return torch.from_numpy(z["target_logodds"]).cuda(), torch.from_numpy(z["target_log_probs"]).cuda()
    with torch.no_grad():
        zf = clean_logits(teacher, data["forget"][0])
        zv = clean_logits(teacher, data["target"][0])
        yf, yv = data["forget"][1].cpu(), data["target"][1].cpu()
        tf, _ = base.logodds_margin(zf, yf)
        tv, _ = base.logodds_margin(zv, yv)
        s = torch.empty_like(tf)
        for c in range(10):
            sel_f, sel_v = yf == c, yv == c
            s[sel_f] = barycentric_quantile_targets(tf[sel_f], tv[sel_v])
        logq = iprojection_log_probs(zf, yf, s)
        assert torch.isfinite(logq).all()
        assert torch.allclose(logq.exp().sum(1), torch.ones(len(yf)), atol=2e-6)
    np.savez(path, target_logodds=s.numpy(), target_log_probs=logq.numpy(),
             forget_original_logits=zf.numpy(), validation_original_logits=zv.numpy(),
             forget_labels=yf.numpy(), validation_labels=yv.numpy())
    base.log("fixed_targets", n_forget=len(yf), n_target=len(yv),
             target_probability_mean=float(torch.sigmoid(s).mean()), test_used=False)
    return s.cuda(), logq.cuda()


def train(a, x, y, idx):
    state = base.original_state(a)
    data = base.gpu_sets(x, y, idx)
    teacher = base.model_from(state).eval().requires_grad_(False)
    s_target, logq = fixed_targets(a, teacher, data)
    losses = list(VARIANTS) if a.loss == "both" else [a.loss]
    for variant in losses:
        name = VARIANTS[variant]
        if base.complete(a, name):
            base.log("skip", method=name)
            continue
        model = base.model_from(state)
        common.set_determinism(a.seed)
        opt = torch.optim.SGD(model.parameters(), lr=.003, momentum=.9, weight_decay=5e-4)
        start = time.monotonic()
        for epoch in range(10):
            model.eval()
            xx, yy = data["forget"]
            order = torch.randperm(len(yy), device="cuda")
            forget_losses = []
            for start_index in range(0, len(yy), 256):
                ii = order[start_index:start_index + 256]
                z = model(xx[ii].float() / 255.)
                if variant == "kl":
                    loss = F.kl_div(F.log_softmax(z, dim=1), logq[ii], reduction="batchmean", log_target=True)
                else:
                    score, _ = base.logodds_margin(z, yy[ii])
                    loss = F.mse_loss(score, s_target[ii])
                opt.zero_grad(set_to_none=True)
                (.1 * loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                opt.step()
                forget_losses.append(loss.detach())
            model.train()
            xx, yy = data["retain"]
            order = torch.randperm(len(yy), device="cuda")
            for start_index in range(0, len(yy), 256):
                ii = order[start_index:start_index + 256]
                xb = common.gpu_augment(xx[ii], True)
                z = model(xb)
                with torch.no_grad():
                    q = F.softmax(teacher(xb) / 2., dim=1)
                anchor = 4. * F.kl_div(F.log_softmax(z / 2., dim=1), q, reduction="batchmean")
                loss = F.cross_entropy(z, yy[ii]) + .1 * anchor
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            base.log("fixed_quantile_epoch", method=name, epoch=epoch + 1,
                     forget_loss=float(torch.stack(forget_losses).mean()), seconds=time.monotonic() - start)
        torch.cuda.synchronize()
        base.save_model(a, name, model, dict(
            seed=a.seed, split_seed=a.split_seed, epochs=10, loss=variant,
            seconds=time.monotonic() - start, code_sha256=CODE_SHA256,
            source="preregister.json", targets_sha256=common.sha256_file(a.run_dir / "fixed_targets.npz"),
            test_used_during_training=False))
        del model
        torch.cuda.empty_cache()


def evaluate(a, x, y, idx):
    missing = [name for name in VARIANTS.values() if not base.complete(a, name)]
    if missing:
        raise ValueError(f"final evaluation requires both predeclared variants; missing {missing}")
    reference_path = a.base_run_dir / "retrain.pt"
    reference_meta_path = a.base_run_dir / "retrain.json"
    if not reference_path.exists() or not reference_meta_path.exists():
        raise ValueError("paired Retrain artifact missing")
    reference_meta = json.loads(reference_meta_path.read_text())
    if reference_meta["epochs"] != 182:
        raise ValueError("full 182-epoch paired Retrain is required")
    # Test arrays are opened for the first time only after every method is fixed.
    with np.load(Path(a.data_dir) / "cifar10.npz", allow_pickle=False) as z:
        xt, yt = z["x_test"], z["y_test"].astype(np.int64)
    if common.sha256_arrays(x, y, xt, yt) != common.SHA256_DATA["cifar10"]:
        raise ValueError("full dataset hash mismatch")
    evaluator = base.module_at("fixed_quantile_evaluator", base.TASK / "scripts/evaluate.py")
    metrics = {}
    paths = {"retrain": reference_path, **{name: a.run_dir / f"{name}.pt" for name in VARIANTS.values()}}
    for name, path in paths.items():
        state = torch.load(path, map_location="cpu", weights_only=True)
        values = evaluator.evaluate_state("cifar10", state, idx["forget"], arrays=(x, y, xt, yt))
        meta = reference_meta if name == "retrain" else json.loads((a.run_dir / f"{name}.json").read_text())
        values["seconds"] = meta["seconds"]
        metrics[name] = values
        base.log("fixed_quantile_evaluation", method=name, **values)
    ref = metrics["retrain"]
    for name, values in metrics.items():
        values["gaps"] = {k: abs(values[k] - ref[k]) for k in ("ua", "ra", "ta", "mia")}
        values["avg_gap"] = sum(values["gaps"].values()) / 4.
    base.dump(a.run_dir / "results.json", dict(
        seed=a.seed, split_seed=a.split_seed, code_sha256=CODE_SHA256,
        reference="same split, independently initialized 182-epoch Retrain; recomputed raw metrics",
        methods=metrics, warning="empirical matching only; no certified unlearning guarantee"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["train", "evaluate", "all"], default="train")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--base-runs-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--original", default="/srv/home/bohanlyu/data_mlsbench/unlearn-lab/original/cifar10_resnet18_original.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--gpu", default=None)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--loss", choices=["kl", "logodds_mse", "both"], default="kl",
                   help="train one or both fixed-target variants; evaluation requires both")
    a = p.parse_args()
    if a.out_dir.resolve() == a.base_runs_dir.resolve():
        raise ValueError("new experiment outputs must be separate from frozen baseline runs")
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    stem = f"seed{a.seed}_split{a.split_seed}"
    a.run_dir, a.base_run_dir = a.out_dir / stem, a.base_runs_dir / stem
    a.run_dir.mkdir(parents=True, exist_ok=True)
    preregister(a)
    x, y = base.train_arrays(a)
    idx = matching_splits(a, y)
    base.original_state(a)
    base.log("fixed_quantile_prepared", seed=a.seed, split_seed=a.split_seed,
             code_sha256=CODE_SHA256, planned_variants=list(VARIANTS.values()), test_used=False)
    if a.phase in ("train", "all"):
        train(a, x, y, idx)
    if a.phase in ("evaluate", "all"):
        evaluate(a, x, y, idx)


if __name__ == "__main__":
    main()
