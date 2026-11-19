"""Run a paired CIFAR-10 architecture comparison on the original membership split."""

import argparse
from contextlib import contextmanager, nullcontext
import fcntl
from functools import partial
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import run_experiments as base
from classification_models import ARCHITECTURES, build_model

import numpy as np
import torch
from torch.nn import functional as F


DEVGAP = Path(__file__).resolve().parent / "devgap"
FROZEN_CONFIG = DEVGAP / "frozen_A2_A3_20261107.json"
BATCH = 256


def split_manifest(args, labels):
    indices = base.splits(args, labels)
    path = args.run_dir / "selection_audit_split.json"
    if not path.exists():
        rng = np.random.RandomState(20261107)
        selection, audit = [], []
        for cls in range(10):
            pool = indices["calibration"][labels[indices["calibration"]] == cls]
            order = rng.permutation(pool)
            selection.extend(order[:125].tolist())
            audit.extend(order[125:].tolist())
        base.dump(path, dict(seed=20261107, selection=sorted(selection), audit=sorted(audit)))
    return indices


def fit_classifier(args, x, y, members, role, factory):
    base.common.set_determinism(args.seed)
    model = factory().cuda()
    if args.architecture == "resnet18":
        model = model.to(memory_format=torch.channels_last)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=5e-4)
        schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[91, 136], gamma=.1)
        recipe = {"optimizer": "SGD", "lr": .1, "momentum": .9, "weight_decay": 5e-4,
                  "milestones": [91, 136], "gamma": .1}
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        lr, min_lr, warmup = .001, .00001, min(5, args.epochs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.05, fused=True)

        def factor(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            fraction = (epoch - warmup) / max(1, args.epochs - warmup - 1)
            return (min_lr + (lr - min_lr) * (1 + math.cos(math.pi * fraction)) / 2) / lr

        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
        recipe = {"optimizer": "AdamW", "lr": lr, "min_lr": min_lr, "weight_decay": .05,
                  "warmup_epochs": warmup, "schedule": "cosine"}

    images = base.common.to_gpu_u8(x[members], "cuda")
    labels = torch.from_numpy(y[members]).cuda()
    precision = "fp32" if args.fp32 else "bf16"
    cfg = dict(architecture=args.architecture, seed=args.seed, split_seed=args.split_seed,
               membership_sha256=base.common.sha256_arrays(members), n_train=len(members),
               epochs=args.epochs, precision=precision, batch_size=BATCH,
               gpu=torch.cuda.get_device_name(), gpu_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
               torch_version=torch.__version__, parameters=sum(p.numel() for p in model.parameters()),
               recipe=recipe, train_sha256=base.common.SHA256_TRAIN["cifar10"],
               test_used_during_training=False)
    base.log("classifier_started", role=role, **cfg)
    torch.cuda.synchronize()
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(labels), device="cuda")
        losses = []
        correct = torch.zeros((), device="cuda")
        for ids in order.split(BATCH):
            xb = base.common.gpu_augment(images[ids], True)
            if args.architecture == "resnet18":
                xb = xb.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16) if not args.fp32 else nullcontext():
                logits = model(xb)
                loss = F.cross_entropy(logits.float(), labels[ids])
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
            correct += (logits.detach().argmax(1) == labels[ids]).sum()
        epoch_lr = optimizer.param_groups[0]["lr"]
        schedule.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            base.log("classifier_epoch", role=role, epoch=epoch, lr=epoch_lr,
                     loss=float(torch.stack(losses).mean()), train_acc=100 * float(correct) / len(labels),
                     seconds=time.monotonic() - started)
    torch.cuda.synchronize()
    base.save_model(args, role, model, dict(cfg, seconds=time.monotonic() - started))


def evaluate(args, factory, indices):
    task_scripts = base.TASK / "scripts"
    sys.path.insert(0, str(task_scripts))
    evaluator = base.module_at("architecture_eval", task_scripts / "evaluate.py")
    with np.load(args.data_dir / "cifar10.npz", allow_pickle=False) as data:
        arrays = tuple(data[key] for key in ("x_train", "y_train", "x_test", "y_test"))
    if base.common.sha256_arrays(*arrays) != base.common.SHA256_DATA["cifar10"]:
        raise ValueError("Full CIFAR-10 hash mismatch")

    paths = {role: args.run_dir / f"{role}.pt" for role in ("original", "retrain", "salun")}
    paths["a2"] = args.run_dir / "a2" / "last.pt"
    timings = {role: json.loads((args.run_dir / f"{role}.json").read_text())["seconds"]
               for role in ("original", "retrain", "salun")}
    a2_timing = json.loads((args.run_dir / "a2" / "summary.json").read_text())["last"]
    timings["a2"] = dict(target_seconds=a2_timing["target_s"],
                         update_seconds=a2_timing["train_s"],
                         forget_passes=a2_timing["forget_passes"])
    forget = indices["forget"].tolist()
    rows = {}
    for role, path in paths.items():
        started = time.monotonic()
        state = torch.load(path, map_location="cpu", weights_only=True)
        metrics = evaluator.evaluate_state("cifar10", state, forget, arrays, model_factory=factory)
        rows[role] = dict(metrics, eval_seconds=time.monotonic() - started,
                          checkpoint_sha256=base.common.sha256_file(path))
        base.log("evaluated", role=role, **rows[role])

    measures = ("ua", "ra", "ta", "mia")
    ref = rows["retrain"]
    for row in rows.values():
        row["absolute_gaps"] = {m: abs(row[m] - ref[m]) for m in measures}
        row["avg_gap"] = sum(row["absolute_gaps"].values()) / len(measures)
    result = dict(architecture=args.architecture, split_seed=args.split_seed, seed=args.seed,
                  data_sha256=base.common.SHA256_DATA["cifar10"], rows=rows, timings=timings)
    base.dump(args.run_dir / "results.json", result)


@contextmanager
def phase_lock(run_dir, phase):
    with (run_dir / f".{phase}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def complete(args):
    if args.phase in ("original", "retrain", "salun"):
        return base.complete(args, args.phase)
    if args.phase == "a2":
        return all((args.run_dir / "a2" / name).is_file() for name in ("last.pt", "summary.json"))
    return (args.run_dir / "results.json").is_file()


def run_phase(args, x, y, indices, factory):
    if args.phase == "original":
        fit_classifier(args, x, y, indices["train"], "original", factory)
    elif args.phase == "retrain":
        fit_classifier(args, x, y, indices["retain"], "retrain", factory)
    elif args.phase == "a2":
        frozen_args = shlex.split(json.loads(FROZEN_CONFIG.read_text())["A2"]["args"])
        map_arg = frozen_args.index("--target-map") + 1
        frozen_args[map_arg] = str(base.ROOT / frozen_args[map_arg])
        subprocess.run([
            sys.executable, str(DEVGAP / "unlearn_dev.py"),
            "--architecture", args.architecture, "--training-only", "--threads", "2",
            "--original", str(args.run_dir / "original.pt"),
            "--splits", str(args.run_dir / "splits.json"),
            "--selection-audit", str(args.run_dir / "selection_audit_split.json"),
            "--data", str(args.data_dir / "cifar10.npz"), "--out", str(args.run_dir / "a2"),
            "--seed", str(args.seed), *frozen_args,
        ], check=True)
    elif args.phase == "salun":
        state = torch.load(args.run_dir / "original.pt", map_location="cpu", weights_only=True)
        base.baseline(args, "salun", base.gpu_sets(x, y, indices), state, model_factory=factory)
    else:
        evaluate(args, factory, indices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "original", "retrain", "a2", "salun", "evaluate"])
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=43)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--epochs", type=int, default=182)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    x, y = base.train_arrays(args)
    with phase_lock(args.run_dir, "prepare"):
        indices = split_manifest(args, y)
    factory = partial(build_model, base.common, args.architecture)
    if args.phase == "prepare":
        return
    with phase_lock(args.run_dir, args.phase):
        if complete(args):
            base.log("skip", phase=args.phase, reason="already completed")
            return
        run_phase(args, x, y, indices, factory)


if __name__ == "__main__":
    main()
