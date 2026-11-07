"""Independent CIFAR-10 unlearning study; does not modify benchmark tasks.

Training loads ONLY x_train/y_train from the hash-pinned HF-order npz.  The
original official pickle ordering is different and MUST NOT be substituted.
Test arrays are opened only by the explicit final evaluation phase.  No test
metric, retrained output, or historical reference enters an update or target.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import subprocess
import time
from types import SimpleNamespace

# Set explicit limits before importing numerical libraries.
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "2")
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "tasks/security-machine-unlearning"


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


common = module_at("study_common", TASK / "scripts/common.py")
SOURCE_SHA256 = common.sha256_file(__file__)
KNOWN_METHODS = ("original", "ft", "ga", "rl", "l1_sparse", "salun",
                 "regun_mean_control", "w2_logodds", "w2_logodds_margin")


def dump(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def log(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def train_arrays(a):
    # np.load is lazy: neither test member is read/decompressed here.
    with np.load(Path(a.data_dir) / "cifar10.npz", allow_pickle=False) as z:
        x, y = z["x_train"], z["y_train"].astype(np.int64)
    got = common.sha256_arrays(x, y)
    if got != common.SHA256_TRAIN["cifar10"]:
        raise ValueError(f"train ordering/content hash {got} != pinned HF-order data")
    return x, y


def original_state(a):
    path = Path(a.original)
    if common.sha256_file(path) != common.SHA256_ORIGINAL["cifar10"]:
        raise ValueError("Original checkpoint hash mismatch")
    return torch.load(path, map_location="cpu", weights_only=True)


def splits(a, y):
    tr, val, forget, retain = common.split_indices(y, 10, a.split_seed, 0.1)
    rng = np.random.RandomState(20261106)
    target, calibration = [], []
    for c in range(10):
        idx = rng.permutation(val[y[val] == c])
        target.extend(idx[:len(idx) // 2])
        calibration.extend(idx[len(idx) // 2:])
    result = {"train": tr.tolist(), "forget": forget.tolist(), "retain": retain.tolist(),
              "target": sorted(int(i) for i in target),
              "calibration": sorted(int(i) for i in calibration)}
    assert [len(result[k]) for k in ("train", "forget", "retain", "target", "calibration")] == [45000, 4500, 40500, 2500, 2500]
    path = a.run_dir / "splits.json"
    manifest = dict(seed=a.seed, split_seed=a.split_seed, validation_seed=common.SPLIT_SEED,
                    target_split_seed=20261106, indices=result)
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("existing run directory has a different split")
    else:
        # Run prepare once before parallel reference/experiments.  Readers of
        # an identical manifest must not race on the same temporary filename.
        dump(path, manifest)
    return {k: np.asarray(v, dtype=np.int64) for k, v in result.items()}


def model_from(state=None, channels_last=False):
    model = common.build_model("cifar10")
    if state is not None:
        model.load_state_dict(state, strict=True)
    model = model.cuda()
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model


def save_model(a, name, model, meta):
    target = a.run_dir / f"{name}.pt"
    tmp = target.with_suffix(".tmp")
    torch.save({k: t.detach().cpu().contiguous() for k, t in model.state_dict().items()}, tmp)
    os.replace(tmp, target)
    dump(a.run_dir / f"{name}.json", meta)
    log("saved", method=name, **meta)


def complete(a, name):
    return (a.run_dir / f"{name}.pt").exists() and (a.run_dir / f"{name}.json").exists()


def gpu_sets(x, y, indices):
    return {k: (common.to_gpu_u8(x[ii], "cuda"), torch.from_numpy(y[ii]).cuda())
            for k, ii in indices.items() if k != "train"}


def reference(a, x, y, idx):
    if complete(a, "retrain") and not a.force:
        log("skip", method="retrain")
        return
    common.set_determinism(a.seed)
    model = model_from(channels_last=a.amp)
    if a.amp:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    xx = common.to_gpu_u8(x[idx["retain"]], "cuda")
    yy = torch.from_numpy(y[idx["retain"]]).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[91, 136], gamma=0.1)
    start = time.monotonic()
    for epoch in range(a.reference_epochs):
        model.train()
        order = torch.randperm(len(yy), device="cuda")
        losses, correct = [], torch.zeros((), device="cuda")
        for s in range(0, len(yy), 256):
            ids = order[s:s + 256]
            xb = common.gpu_augment(xx[ids], True)
            if a.amp:
                xb = xb.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
                z = model(xb)
                loss = F.cross_entropy(z.float(), yy[ids])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.detach())
            correct += (z.argmax(1) == yy[ids]).sum()
        sched.step()
        if epoch % 5 == 0 or epoch + 1 == a.reference_epochs:
            log("reference_epoch", epoch=epoch + 1, loss=float(torch.stack(losses).mean()),
                train_acc=float(100 * correct / len(yy)), seconds=time.monotonic() - start)
    torch.cuda.synchronize()
    save_model(a, "retrain", model, dict(seed=a.seed, split_seed=a.split_seed,
               epochs=a.reference_epochs, amp=a.amp, seconds=time.monotonic() - start,
               recipe="SGD lr=.1,momentum=.9,wd=.0005,batch=256,milestones=91/136,gamma=.1",
               test_used_during_training=False))


class WorkerRNG:
    """Keep method randomness separate, as in the original parent/worker API."""
    def __init__(self, seed):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed)
            self.cpu = torch.get_rng_state()
            self.gpu = torch.cuda.get_rng_state()

    def call(self, fn, *args):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.set_rng_state(self.cpu)
            torch.cuda.set_rng_state(self.gpu)
            result = fn(*args)
            self.cpu, self.gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
            return result


def baseline(a, name, data, state):
    model = model_from(state)
    common.set_determinism(a.seed)
    scope = {}
    src = runpy.run_path(str(TASK / "edits" / f"{name}.edit.py"))["_CONTENT"]
    exec(compile(src, f"baseline:{name}", "exec"), scope)
    method = scope["UnlearningMethod"]()
    ctx = SimpleNamespace(dataset="cifar10", setting="cifar10_random10", forget_fraction=.1,
                          num_classes=10, n_retain=40500, n_forget=4500, batch_size=256,
                          retain_passes=10, forget_passes=11, seed=a.seed,
                          time_limit_sec=3600, device=torch.device("cuda"))
    worker = WorkerRNG(a.seed)
    plan = worker.call(method.plan, ctx)
    params = dict(model.named_parameters())
    for p in params.values():
        p.grad = torch.zeros_like(p)
    if hasattr(method, "begin"):
        worker.call(method.begin, params, ctx)
    generator = torch.Generator(device="cpu").manual_seed(a.seed)
    start = time.monotonic()
    for pi, ps in enumerate(plan):
        model.train(not ps.get("eval", False))
        keys = ["retain", "forget"] if ps["set"] == "both" else [ps["set"]]
        srcs = torch.cat([torch.full((len(data[k][1]),), int(k == "forget"), dtype=torch.long) for k in keys])
        ids = torch.cat([torch.arange(len(data[k][1])) for k in keys])
        order = torch.randperm(len(ids), generator=generator)
        srcs, ids = srcs[order], ids[order]
        accumulate = ps.get("mode", "train") == "accumulate"
        if accumulate:
            model.zero_grad(set_to_none=False)
        for step, s in enumerate(range(0, len(ids), 256)):
            ib, fb = ids[s:s + 256].cuda(), srcs[s:s + 256].cuda().bool()
            xb = torch.empty((len(ib), 3, 32, 32), dtype=torch.uint8, device="cuda")
            yb = torch.empty(len(ib), dtype=torch.long, device="cuda")
            for k, select in (("retain", ~fb), ("forget", fb)):
                if select.any():
                    xb[select], yb[select] = data[k][0][ib[select]], data[k][1][ib[select]]
            xb = common.gpu_augment(xb, True)  # Harness augments even eval saliency passes.
            if not accumulate:
                model.zero_grad(set_to_none=False)
            logits = model(xb)
            info = SimpleNamespace(pass_index=pi, step=step, train=not accumulate)
            loss = worker.call(method.loss, logits, yb, fb, ib, info)
            loss.backward()
            if not accumulate:
                worker.call(method.update, params, info)
        if hasattr(method, "end_pass"):
            worker.call(method.end_pass, params, SimpleNamespace(pass_index=pi))
        log("baseline_pass", method=name, pass_index=pi, seconds=time.monotonic() - start)
    if hasattr(method, "finish"):
        worker.call(method.finish, params, ctx)
    torch.cuda.synchronize()
    save_model(a, name, model, dict(seed=a.seed, split_seed=a.split_seed,
              seconds=time.monotonic() - start, passes=plan,
              baseline_source_sha256=hashlib.sha256(src.encode()).hexdigest(), fp32=True,
              test_used_during_training=False))


def logodds_margin(z, labels):
    true = z.gather(1, labels[:, None]).squeeze(1)
    other = z.masked_fill(F.one_hot(labels, z.shape[1]).bool(), -torch.inf)
    return true - torch.logsumexp(other, dim=1), true - other.max(dim=1).values


def conditional_w2(current, labels, targets, target_labels):
    terms = []
    for c in range(10):
        v = current[labels == c]
        ref = targets[target_labels == c]
        if v.numel():
            # Exact integral of the two empirical quantile functions, including
            # unequal sample counts.  Midpoint interpolation alone is biased:
            # it gives a nonzero distance for identical empirical samples.
            ref = ref.detach().sort().values
            value = v.sort().values
            edges = torch.cat((torch.arange(len(v) + 1, device=v.device) / len(v),
                               torch.arange(len(ref) + 1, device=v.device) / len(ref)))
            edges = edges.sort().values
            mass = edges[1:] - edges[:-1]
            midpoint = .5 * (edges[1:] + edges[:-1])
            vi = (midpoint * len(v)).long().clamp(max=len(v) - 1)
            ri = (midpoint * len(ref)).long().clamp(max=len(ref) - 1)
            terms.append((mass * (value[vi] - ref[ri]).square()).sum())
    return torch.stack(terms).mean()


def candidate(a, name, data, state):
    model, teacher = model_from(state), model_from(state)
    teacher.eval().requires_grad_(False)
    common.set_determinism(a.seed)
    tx, ty = data["target"]
    with torch.no_grad():
        tz = torch.cat([teacher(tx[s:s + 256].float() / 255.) for s in range(0, len(ty), 256)])
        targets = logodds_margin(tz, ty)
        target_probs = F.softmax(tz, dim=1)
        target_class_ids = [torch.where(ty == c)[0] for c in range(10)]
    opt = torch.optim.SGD(model.parameters(), lr=a.candidate_lr, momentum=.9, weight_decay=5e-4)
    start = time.monotonic()
    for epoch in range(a.candidate_epochs):
        # One forget pass, eval mode: no BN changes or random augmentation.
        model.eval()
        xx, yy = data["forget"]
        order = torch.randperm(len(yy), device="cuda")
        forget_losses = []
        for s in range(0, len(yy), 256):
            ii = order[s:s + 256]
            z = model(xx[ii].float() / 255.)
            if name == "regun_mean_control":
                # Simplified ReGUn-style control, NOT an exact reproduction:
                # class-histogram-matched reference batch, one shared mean q.
                sampled = []
                for c in range(10):
                    count = int((yy[ii] == c).sum())
                    if count:
                        pool = target_class_ids[c]
                        sampled.append(pool[torch.randint(len(pool), (count,), device="cuda")])
                q = target_probs[torch.cat(sampled)].mean(dim=0).expand(len(ii), -1)
                loss = F.kl_div(F.log_softmax(z, dim=1), q, reduction="batchmean")
            else:
                values = logodds_margin(z, yy[ii])
                loss = conditional_w2(values[0], yy[ii], targets[0], ty)
                if name == "w2_logodds_margin":
                    loss = .5 * (loss + conditional_w2(values[1], yy[ii], targets[1], ty))
            opt.zero_grad(set_to_none=True)
            (a.forget_weight * loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
            forget_losses.append(loss.detach())
        # One retain pass: supervised utility + fixed Original distillation.
        model.train()
        xx, yy = data["retain"]
        order = torch.randperm(len(yy), device="cuda")
        for s in range(0, len(yy), 256):
            ii = order[s:s + 256]
            xb = common.gpu_augment(xx[ii], True)
            z = model(xb)
            with torch.no_grad():
                q = F.softmax(teacher(xb) / 2., dim=1)
            anchor = 4. * F.kl_div(F.log_softmax(z / 2., dim=1), q, reduction="batchmean")
            loss = F.cross_entropy(z, yy[ii]) + a.anchor_weight * anchor
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        log("candidate_epoch", method=name, epoch=epoch + 1,
            w2_loss=float(torch.stack(forget_losses).mean()), seconds=time.monotonic() - start)
    torch.cuda.synchronize()
    save_model(a, name, model, dict(seed=a.seed, split_seed=a.split_seed,
              seconds=time.monotonic() - start, epochs=a.candidate_epochs, lr=a.candidate_lr,
              forget_weight=a.forget_weight, anchor_weight=a.anchor_weight,
              target="Original on stratified 2500 held-out training images; fixed, detached",
              candidate_primary=name == "w2_logodds_margin", fp32=True,
              comparator_note=("ReGUn-inspired mean-target control; fixed teacher, shared anchor; "
                               "not an exact reproduction" if name == "regun_mean_control" else None),
              selection="predeclared hyperparameters; no selection on test or Retrain",
              test_used_during_training=False))


def experiments(a, x, y, idx):
    state = original_state(a)
    data = gpu_sets(x, y, idx)
    for name in a.methods.split(","):
        if complete(a, name) and not a.force:
            log("skip", method=name)
            continue
        if name in ("regun_mean_control", "w2_logodds", "w2_logodds_margin"):
            candidate(a, name, data, state)
        elif name in ("original", "ft", "ga", "rl", "l1_sparse", "salun"):
            baseline(a, name, data, state)
        else:
            raise ValueError(f"unknown method: {name}")
        torch.cuda.empty_cache()


def evaluate(a, x, y, idx):
    planned = json.loads((a.run_dir / "run_config.json").read_text())["planned_methods"]
    missing = [name for name in ["retrain", *planned] if not complete(a, name)]
    if missing:
        raise ValueError(f"final evaluation refused; incomplete preregistered methods: {missing}")
    # First access to test members: all training for this run is complete.
    with np.load(Path(a.data_dir) / "cifar10.npz", allow_pickle=False) as z:
        xt, yt = z["x_test"], z["y_test"].astype(np.int64)
    if common.sha256_arrays(x, y, xt, yt) != common.SHA256_DATA["cifar10"]:
        raise ValueError("full data hash mismatch")
    evaluator = module_at("study_evaluate", TASK / "scripts/evaluate.py")
    result = {}
    for path in sorted(a.run_dir.glob("*.pt")):
        name = path.stem
        state = torch.load(path, map_location="cpu", weights_only=True)
        values = evaluator.evaluate_state("cifar10", state, idx["forget"], arrays=(x, y, xt, yt))
        meta = json.loads((a.run_dir / f"{name}.json").read_text())
        values["seconds"] = meta["seconds"]
        result[name] = values
        log("evaluation", method=name, **values)
    if "retrain" in result:
        ref = result["retrain"]
        for name, values in result.items():
            gaps = {k: abs(values[k] - ref[k]) for k in ("ua", "ra", "ta", "mia")}
            values["gaps"] = gaps
            values["avg_gap"] = sum(gaps.values()) / 4
    reference_epochs = json.loads((a.run_dir / "retrain.json").read_text())["epochs"]
    dump(a.run_dir / "results.json", dict(seed=a.seed, split_seed=a.split_seed,
         reference=f"same deletion split; independently initialized {reference_epochs}-epoch retraining",
         reference_epochs=reference_epochs,
         warning="single seed is preliminary; empirical metrics are not certified deletion",
         methods=result))


def diagnostics(a, x, y, idx):
    """Clean predictions on forget/target/calibration; never reads official test."""
    device = torch.device(a.device)
    model = common.build_model("cifar10")
    model.load_state_dict(original_state(a), strict=True)
    model = model.to(device).eval()
    outputs, statistics = {}, {}
    start = time.monotonic()
    for name in ("forget", "target", "calibration"):
        xx, yy = x[idx[name]], y[idx[name]]
        logits = []
        with torch.no_grad():
            for i in range(0, len(yy), 128):
                xb = common.to_gpu_u8(xx[i:i + 128], device).float() / 255.
                logits.append(model(xb).cpu())
        z = torch.cat(logits)
        labels = torch.from_numpy(yy)
        t, margin = logodds_margin(z, labels)
        confidence = F.softmax(z, dim=1).gather(1, labels[:, None]).squeeze(1)
        outputs[f"{name}_logits"] = z.numpy()
        outputs[f"{name}_labels"] = yy
        outputs[f"{name}_indices"] = idx[name]
        statistics[name] = {}
        for group in ("all", *range(10)):
            select = torch.ones(len(yy), dtype=torch.bool) if group == "all" else labels == group
            quantiles = torch.tensor([0., .01, .05, .1, .25, .5, .75, .9, .95, .99, 1.])
            statistics[name][str(group)] = dict(
                n=int(select.sum()), ua=float(100 * (margin[select] < 0).float().mean()),
                confidence_mean=float(confidence[select].mean()),
                quantile_levels=quantiles.tolist(),
                confidence_quantiles=torch.quantile(confidence[select], quantiles).tolist(),
                logodds_quantiles=torch.quantile(t[select], quantiles).tolist(),
                margin_quantiles=torch.quantile(margin[select], quantiles).tolist())
        log("diagnostic_set", set=name, **statistics[name]["all"],
            seconds=time.monotonic() - start)
    np.savez(a.run_dir / "diagnostic_logits.npz", **outputs)
    dump(a.run_dir / "diagnostics.json", dict(
        device=str(device), seconds=time.monotonic() - start,
        seed=a.seed, split_seed=a.split_seed, code_sha256=SOURCE_SHA256,
        test_used=False, interpretation="Original membership/generalization diagnostics only; no unlearning result",
        statistics=statistics))


def run_config(a):
    """Immutable preregistration and resume check, including implementation hash."""
    current = dict(schema=1, code_sha256=SOURCE_SHA256,
        common_sha256=common.sha256_file(TASK / "scripts/common.py"),
        train_sha256=common.SHA256_TRAIN["cifar10"],
        original_sha256=common.SHA256_ORIGINAL["cifar10"],
        seed=a.seed, split_seed=a.split_seed, reference_epochs=a.reference_epochs,
        reference_amp=a.amp, candidate_epochs=a.candidate_epochs,
        candidate_lr=a.candidate_lr, forget_weight=a.forget_weight,
        anchor_weight=a.anchor_weight, torch_version=torch.__version__,
        numpy_version=np.__version__, sklearn_version=importlib.metadata.version("scikit-learn"),
        baseline_sha256={name: common.sha256_file(TASK / "edits" / f"{name}.edit.py")
                         for name in KNOWN_METHODS if (TASK / "edits" / f"{name}.edit.py").exists()},
        planned_methods=list(KNOWN_METHODS))
    path = a.run_dir / "run_config.json"
    if path.exists():
        previous = json.loads(path.read_text())
        mismatched = [key for key in current if current[key] != previous.get(key)]
        if mismatched:
            raise ValueError(f"resume config mismatch {mismatched}; use a new out-dir")
    else:
        dump(path, current)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["prepare", "diagnostics", "reference", "experiments", "evaluate", "all"], default="all")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--original", default="/srv/home/bohanlyu/data_mlsbench/unlearn-lab/original/cifar10_resnet18_original.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--gpu", default=None, help="sets CUDA_VISIBLE_DEVICES before CUDA initialization")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="diagnostics phase only")
    p.add_argument("--amp", action="store_true", help="BF16/channels_last for reference only; baseline/candidates stay FP32")
    p.add_argument("--reference-epochs", type=int, default=182)
    p.add_argument("--candidate-epochs", type=int, default=10)
    p.add_argument("--candidate-lr", type=float, default=.003)
    p.add_argument("--forget-weight", type=float, default=.1)
    p.add_argument("--anchor-weight", type=float, default=.1)
    p.add_argument("--methods", default=",".join(KNOWN_METHODS))
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    a.run_dir = a.out_dir / f"seed{a.seed}_split{a.split_seed}"
    a.run_dir.mkdir(parents=True, exist_ok=True)
    x, y = train_arrays(a)
    idx = splits(a, y)
    original_state(a)
    run_config(a)
    if any(name not in KNOWN_METHODS for name in a.methods.split(",")):
        raise ValueError("--methods must be a subset of the preregistered methods")
    log("prepared", seed=a.seed, split_seed=a.split_seed, n_forget=len(idx["forget"]),
        n_retain=len(idx["retain"]), n_target=len(idx["target"]), n_calibration=len(idx["calibration"]),
        train_sha256=common.SHA256_TRAIN["cifar10"], code_sha256=SOURCE_SHA256)
    if a.phase == "diagnostics":
        diagnostics(a, x, y, idx)
    if a.phase in ("all", "reference"):
        reference(a, x, y, idx)
    if a.phase in ("all", "experiments"):
        experiments(a, x, y, idx)
    if a.phase in ("all", "evaluate"):
        evaluate(a, x, y, idx)


if __name__ == "__main__":
    main()
