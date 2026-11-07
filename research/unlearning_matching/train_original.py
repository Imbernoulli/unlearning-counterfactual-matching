"""Train an independent CIFAR-10 Original on the canonical 45,000 members.

Only x_train/y_train are read from the hash-verified HF-order cifar10.npz.
No validation predictions, test arrays, prior models, Retrain artifacts, or
result files are read.  The fixed recipe matches run_experiments.reference,
except that all 45,000 canonical Original-training members are used.
Only the final state_dict and its provenance manifest are saved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COMMON_PATH = ROOT / "tasks/security-machine-unlearning/scripts/common.py"
RECIPE = dict(epochs=182, batch_size=256, optimizer="SGD", lr=.1, momentum=.9,
              weight_decay=5e-4, milestones=[91, 136], lr_gamma=.1,
              augmentation="RandomCrop(32,padding=4)+RandomHorizontalFlip",
              validation_selection=False, intermediate_checkpoints=False)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def log(event, **values):
    print(json.dumps(dict(event=event, **values), allow_nan=False), flush=True)


def load_components(threads):
    """Set limits before importing numerical libraries; no CUDA initialization."""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(threads)
    global np, torch, F
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    spec = importlib.util.spec_from_file_location("independent_original_common", COMMON_PATH)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    return common


def train_arrays(data_dir, common):
    # NpzFile is lazy: no other archive member is opened/decompressed.
    with np.load(Path(data_dir) / "cifar10.npz", allow_pickle=False) as archive:
        x = archive["x_train"]
        y = archive["y_train"].astype(np.int64)
    actual = common.sha256_arrays(x, y)
    if actual != common.SHA256_TRAIN["cifar10"]:
        raise ValueError(f"training arrays SHA256 {actual} differs from pinned HF-order data")
    if x.shape != (50000, 32, 32, 3) or x.dtype != np.uint8 or y.shape != (50000,):
        raise ValueError("unexpected CIFAR-10 training-array shape or dtype")
    if not np.array_equal(np.bincount(y, minlength=10), np.full(10, 5000)):
        raise ValueError("unexpected CIFAR-10 class counts")
    return x, y


def training_membership(y, common):
    # The first return value depends on common.SPLIT_SEED, not forget_seed.
    # No validation images, targets, calibration predictions or deletion split
    # are materialized by this script.
    members = common.split_indices(y, 10, forget_seed=0, forget_fraction=.1)[0]
    members = np.asarray(members, dtype="<i8")
    if len(members) != 45000 or len(np.unique(members)) != 45000:
        raise ValueError("canonical Original must have exactly 45,000 members")
    return members


def configuration(a, common, members):
    return dict(
        schema=1, dataset="cifar10", architecture="common.ResNet18",
        seed=a.seed, recipe=RECIPE, amp=a.amp, channels_last=a.amp,
        parameter_dtype="float32", autocast_dtype="bfloat16" if a.amp else None,
        loss_dtype="float32", threads=a.threads, interop_threads=1,
        cudnn_benchmark=a.amp, cudnn_deterministic=not a.amp,
        train_sha256=common.SHA256_TRAIN["cifar10"],
        train_membership_sha256=hashlib.sha256(members.tobytes(order="C")).hexdigest(),
        train_membership_encoding="sorted absolute indices, little-endian int64 C-order bytes",
        train_size=len(members), canonical_validation_split_seed=common.SPLIT_SEED,
        array_keys_read=["x_train", "y_train"],
        test_used=False, validation_predictions_used=False, retrain_used=False,
        source=dict(script="research/unlearning_matching/train_original.py",
                    script_sha256=sha256(__file__),
                    common="tasks/security-machine-unlearning/scripts/common.py",
                    common_sha256=sha256(COMMON_PATH)),
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      torch=str(torch.__version__), cuda=torch.version.cuda))


def existing_complete(output, config):
    checkpoint, manifest = output / "original.pt", output / "original.json"
    if not checkpoint.exists() and not manifest.exists():
        return False
    if not checkpoint.is_file() or not manifest.is_file():
        raise FileExistsError("incomplete Original artifacts; refusing to overwrite; use a new out-dir")
    previous = json.loads(manifest.read_text())
    if previous.get("configuration") != config or previous.get("complete") is not True:
        raise ValueError("existing Original configuration differs; use a new out-dir")
    if previous.get("original_file") != "original.pt" or sha256(checkpoint) != previous.get("original_sha256"):
        raise ValueError("existing Original checkpoint SHA256 mismatch")
    return True


def publish_new(temporary, destination):
    """Atomically publish on the same filesystem, refusing existing names."""
    os.link(temporary, destination)
    Path(temporary).unlink()


def save_final(output, model, config, seconds, started):
    checkpoint, manifest = output / "original.pt", output / "original.json"
    if checkpoint.exists() or manifest.exists():
        raise FileExistsError("Original artifacts appeared during training; refusing to overwrite")
    state = {key: tensor.detach().cpu().contiguous()
             for key, tensor in model.state_dict().items()}
    if any(not torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError("final Original contains nonfinite values; refusing to publish")
    temporary_paths = []
    try:
        with tempfile.NamedTemporaryFile(dir=output, prefix=".original-", suffix=".pt", delete=False) as stream:
            temporary_paths.append(Path(stream.name))
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256(temporary_paths[0])
        metadata = dict(schema=1, complete=True, original_file="original.pt", original_sha256=digest,
                        original_bytes=temporary_paths[0].stat().st_size,
                        configuration=config, seed=config["seed"],
                        train_sha256=config["train_sha256"],
                        train_membership_sha256=config["train_membership_sha256"],
                        training_loop_seconds=seconds, started_utc=started, completed_utc=stamp(),
                        selection="fixed final epoch 182; no validation/test selection")
        with tempfile.NamedTemporaryFile(dir=output, prefix=".original-", suffix=".json", mode="w", delete=False) as stream:
            temporary_paths.append(Path(stream.name))
            stream.write(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        # A crash between these publications leaves an explicit incomplete pair
        # which existing_complete refuses; it never silently replaces weights.
        publish_new(temporary_paths[0], checkpoint)
        publish_new(temporary_paths[1], manifest)
        log("original_saved", original_file=str(checkpoint), original_sha256=digest,
            training_loop_seconds=seconds, seed=config["seed"])
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


def train(a, common, x, y, members, config):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the fixed full-scale training; no CPU training fallback")
    if a.amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError("--amp requires CUDA BF16 support")
    common.set_determinism(a.seed)
    model = common.build_model("cifar10").cuda()
    if a.amp:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        model = model.to(memory_format=torch.channels_last)
    xx = common.to_gpu_u8(x[members], "cuda")
    yy = torch.from_numpy(y[members]).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=RECIPE["lr"], momentum=RECIPE["momentum"],
                          weight_decay=RECIPE["weight_decay"])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=RECIPE["milestones"], gamma=RECIPE["lr_gamma"])
    log("original_training_started", configuration=config, gpu=torch.cuda.get_device_name(),
        gpu_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    started = stamp()
    torch.cuda.synchronize()
    start = time.monotonic()
    for epoch in range(RECIPE["epochs"]):
        model.train()
        order = torch.randperm(len(yy), device="cuda")
        losses, correct = [], torch.zeros((), device="cuda")
        epoch_lr = opt.param_groups[0]["lr"]
        for offset in range(0, len(yy), RECIPE["batch_size"]):
            ids = order[offset:offset + RECIPE["batch_size"]]
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
        if (epoch + 1) % 5 == 0 or epoch + 1 == RECIPE["epochs"]:
            log("original_epoch", epoch=epoch + 1, lr=epoch_lr,
                loss=float(torch.stack(losses).mean()),
                augmented_training_accuracy=100 * float(correct) / len(yy),
                seconds=time.monotonic() - start)
    torch.cuda.synchronize()
    save_final(a.out_dir, model, config, time.monotonic() - start, started)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES before CUDA initialization")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="BF16 autocast and channels_last, as in the reference recipe")
    a = parser.parse_args(argv)
    if a.threads <= 0 or not 0 <= a.seed < 2 ** 32:
        parser.error("threads must be positive and seed must be in [0, 2**32)")
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    common = load_components(a.threads)
    x, y = train_arrays(a.data_dir, common)
    members = training_membership(y, common)
    config = configuration(a, common, members)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    lock = a.out_dir / ".train_original.lock"
    # Only this exact process owns this lock.  A stale lock is never removed
    # automatically: interrupted runs should use a fresh output directory.
    with lock.open("x") as stream:
        stream.write(json.dumps(dict(pid=os.getpid(), started_utc=stamp(), seed=a.seed)) + "\n")
    try:
        if existing_complete(a.out_dir, config):
            log("original_skip_verified", out_dir=str(a.out_dir), seed=a.seed)
            return
        train(a, common, x, y, members, config)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
