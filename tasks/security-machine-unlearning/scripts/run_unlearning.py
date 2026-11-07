"""Trusted unlearning driver for security-machine-unlearning (READ-ONLY).

The PARENT (this process, never imports the editable module) owns everything
that decides the score:

* the data: it verifies the pinned sha256 of the dataset and of the Original
  checkpoint, draws the forget set from FRESH entropy (unknowable before the
  run), and is the only process that ever holds an image;
* the model: the network lives here, on the GPU; every forward and
  backward pass is run here, in fp32, on batches the parent draws;
* the loop: the method declares a plan of passes (retain / forget / both;
  train or gradient-accumulate; train or eval mode), the parent validates it
  against SalUn's budget (<= RETAIN_PASSES passes over D_r, <= FORGET_PASSES
  over D_f) and executes it itself, batch by batch, batch size 256;
* the result: it checks the final weights are an update of the Original
  (global conv/linear cosine), and writes the checkpoint and the run status
  (including the forget indices) that evaluate.py reads.

The editable UnlearningMethod runs in worker.py, a Landlock-sandboxed child
that never receives an image.  Per batch it gets the logits the parent
computed (plus labels, set membership and sample ids) and returns the
gradient of its loss w.r.t. those logits; after the parent's backward pass it
may update the parameters in place (they are shared GPU tensors: the
optimiser, masks, momentum, regularisers are the method's).  Without images
it cannot train on anything beyond the batches the parent chooses to run, so
compute and data are bounded by the plan, not by a clock.
"""

import argparse
import importlib.util
import io
import json
import os
import select
import signal
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing  # noqa: F401  (registers CUDA IPC reductions)
from multiprocessing.reduction import ForkingPickler

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("unlearn_common_parent", _HERE / "common.py")
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)

# parent -> worker ops
OP_LOSS, OP_UPDATE, OP_PASS_END, OP_FINISH = 1, 2, 3, 4
# worker -> parent status
ST_OK, ST_ERR = 0, 1
SETUP_SEC = 300           # worker start-up (CUDA init, imports) and plan()
CALL_SEC = 60             # any single hook call
MAX_PASSES = 64


def _read_exact(fd, n, deadline):
    buf = bytearray()
    while len(buf) < n:
        left = deadline - time.time()
        if left <= 0:
            raise TimeoutError
        r, _, _ = select.select([fd], [], [], min(left, 5.0))
        if not r:
            continue
        chunk = os.read(fd, min(n - len(buf), 1 << 22))
        if not chunk:
            raise EOFError
        buf += chunk
    return bytes(buf)


def _write_all(fd, data):
    view = memoryview(data)
    while view:
        k = os.write(fd, view)
        view = view[k:]


class WorkerError(RuntimeError):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(common.DATASETS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--forget-fraction", type=float, default=0.10,
                    help="random data forgetting scenario (SalUn Table 1: 0.1 or 0.5)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--time-limit-sec", type=float, required=True)
    ap.add_argument("--custom", required=True, help="path of the editable custom_unlearning.py")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path, status_path = out_dir / "unlearned.pt", out_dir / "run_status.json"
    for p in (ckpt_path, status_path):
        if p.exists():
            p.unlink()
    status = {"dataset": args.dataset, "seed": args.seed, "valid": False,
              "forget_fraction": args.forget_fraction}

    def fail(reason, rc=3):
        status["reason"] = reason
        status_path.write_text(json.dumps(status))
        print(f"UNLEARN_INVALID {reason}", flush=True)
        return rc

    # ---- editable source (read once, size-capped) ----
    custom_path = Path(args.custom)
    try:
        custom_src = custom_path.read_bytes()
    except OSError as exc:
        return fail(f"cannot read the editable file: {exc}")
    if len(custom_src) > common.MAX_CUSTOM_BYTES:
        return fail(f"editable file is {len(custom_src)} bytes (cap {common.MAX_CUSTOM_BYTES})")

    # ---- verified data + Original, fresh forget split ----
    try:
        frac = common.check_forget_fraction(args.forget_fraction)
    except ValueError as exc:
        return fail(str(exc))
    spec = common.DATASETS[args.dataset]
    try:
        (x_tr, y_tr, _, _), original = common.load_verified(args.dataset, with_test=False)
    except Exception as exc:  # noqa: BLE001
        return fail(str(exc))
    split_seed = int.from_bytes(os.urandom(8), "little")
    _, _, forget_idx, retain_idx = common.split_indices(y_tr, spec["num_classes"], split_seed,
                                                        frac)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ONE shared GPU allocation holds the parameters, their gradients and the
    # logits / loss-gradient buffers, and nothing else.  CUDA IPC exports a
    # whole caching-allocator segment, so it is allocated FIRST (before any
    # image reaches the GPU) with a size that is a multiple of the 2 MiB
    # segment granularity, and the segment is checked to contain this single
    # block -- the worker can map exactly this memory and no image.
    model = common.build_model(args.dataset)
    model.load_state_dict(original, strict=True)
    bs = common.BATCH_SIZE
    n_cls = spec["num_classes"]
    names = [k for k, _ in model.named_parameters()]
    shapes = {k: tuple(p.shape) for k, p in model.named_parameters()}
    n_param = sum(p.numel() for p in model.parameters())
    n_total = 2 * n_param + 2 * bs * n_cls
    unit = (2 << 20) // 4
    flat = torch.zeros(((n_total + unit - 1) // unit) * unit, device=device)
    if device.type == "cuda":
        segs = [sg for sg in torch.cuda.memory_snapshot()
                if sg["address"] <= flat.data_ptr() < sg["address"] + sg["total_size"]]
        if len(segs) != 1 or len(segs[0]["blocks"]) != 1:
            return fail("could not isolate the shared GPU buffer")
    layout, off = [], 0
    model = model.to(device)
    params = dict(model.named_parameters())
    for k in names:
        n = params[k].numel()
        view = flat[off:off + n].view(shapes[k])
        view.copy_(params[k].data)
        params[k].data = view
        layout.append((k, "param", off, list(shapes[k])))
        off += n
    for k in names:
        n = params[k].numel()
        params[k].grad = flat[off:off + n].view(shapes[k])
        layout.append((k, "grad", off, list(shapes[k])))
        off += n
    logits_buf = flat[off:off + bs * n_cls].view(bs, n_cls)
    layout.append(("logits", "buf", off, [bs, n_cls]))
    off += bs * n_cls
    dlogits_buf = flat[off:off + bs * n_cls].view(bs, n_cls)
    layout.append(("dlogits", "buf", off, [bs, n_cls]))
    grad_ptrs = {k: params[k].grad.data_ptr() for k in names}
    common.set_determinism(args.seed)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    x = [common.to_gpu_u8(x_tr[retain_idx], device), common.to_gpu_u8(x_tr[forget_idx], device)]
    y = [torch.from_numpy(y_tr[retain_idx]).to(device), torch.from_numpy(y_tr[forget_idx]).to(device)]
    n_set = [len(retain_idx), len(forget_idx)]
    del x_tr

    # ---- worker ----
    req_r, req_w = os.pipe()
    resp_r, resp_w = os.pipe()
    scratch = tempfile.mkdtemp(prefix="unlearn-worker-")
    os.chmod(scratch, 0o700)
    env = {k: v for k, v in os.environ.items()
           if k in ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
                    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG", "LANG", "LC_ALL")}
    env.update(HOME=scratch, TMPDIR=scratch, XDG_CACHE_HOME=scratch, TRITON_CACHE_DIR=scratch,
               TORCHINDUCTOR_CACHE_DIR=scratch, PYTHONDONTWRITEBYTECODE="1")
    worker = subprocess.Popen(
        [sys.executable, "-I", "-B", str(_HERE / "worker.py"), "--req-fd", str(req_w),
         "--resp-fd", str(resp_r), "--scratch", str(scratch), "--custom-name", str(custom_path)],
        pass_fds=(req_w, resp_r), env=env, start_new_session=True, cwd=scratch)
    os.close(req_w)
    os.close(resp_r)

    def kill_worker():
        try:
            os.killpg(worker.pid, signal.SIGKILL)
        except OSError:
            pass
        worker.wait()

    def send(obj_bytes):
        _write_all(resp_w, struct.pack("<q", len(obj_bytes)) + obj_bytes)

    def reply(deadline):
        st, = struct.unpack("<i", _read_exact(req_r, 4, deadline))
        if st == ST_OK:
            return
        n, = struct.unpack("<i", _read_exact(req_r, 4, deadline))
        msg = _read_exact(req_r, max(0, min(n, 1 << 16)), deadline).decode("utf-8", "replace")
        raise WorkerError(msg)

    t_start = time.time()
    hard_deadline = t_start + SETUP_SEC + args.time_limit_sec

    def call_deadline():
        return min(time.time() + CALL_SEC, hard_deadline)

    ctx = {"dataset": args.dataset, "setting": common.setting_name(args.dataset, frac),
           "forget_fraction": frac, "num_classes": n_cls, "n_retain": n_set[0],
           "n_forget": n_set[1], "batch_size": bs, "retain_passes": common.RETAIN_PASSES,
           "forget_passes": common.FORGET_PASSES, "seed": args.seed,
           "time_limit_sec": args.time_limit_sec}
    used = [0, 0]
    try:
        # setup: shared GPU tensors (the parent pickles; it never unpickles
        # anything from the worker), labels, context, editable source.
        buf = io.BytesIO()
        ForkingPickler(buf).dump({"flat": flat, "layout": layout})
        send(buf.getvalue())
        send(y[0].cpu().numpy().tobytes())
        send(y[1].cpu().numpy().tobytes())
        send(json.dumps(ctx).encode())
        send(custom_src)
        # plan (JSON, validated here)
        n, = struct.unpack("<q", _read_exact(req_r, 8, t_start + SETUP_SEC))
        if not 0 < n <= 1 << 16:
            raise WorkerError("malformed plan")
        raw = _read_exact(req_r, n, t_start + SETUP_SEC)
        if raw.startswith(b"ERR"):
            raise WorkerError(raw[3:].decode("utf-8", "replace"))
        plan = json.loads(raw.decode())
        if not isinstance(plan, list) or len(plan) > MAX_PASSES:
            raise WorkerError(f"plan must be a list of at most {MAX_PASSES} passes")
        passes = [0, 0]
        norm = []
        for ps in plan:
            if not isinstance(ps, dict):
                raise WorkerError("each pass must be a dict")
            which = ps.get("set")
            mode = ps.get("mode", "train")
            if which not in ("retain", "forget", "both") or mode not in ("train", "accumulate"):
                raise WorkerError(f"bad pass {ps}")
            if which in ("retain", "both"):
                passes[0] += 1
            if which in ("forget", "both"):
                passes[1] += 1
            norm.append((which, mode, bool(ps.get("eval", False))))
        if passes[0] > common.RETAIN_PASSES or passes[1] > common.FORGET_PASSES:
            raise WorkerError(f"plan exceeds the budget: {passes[0]} retain passes "
                              f"(max {common.RETAIN_PASSES}), {passes[1]} forget passes "
                              f"(max {common.FORGET_PASSES})")

        # ---- execute the plan ----
        for pi, (which, mode, eval_mode) in enumerate(norm):
            model.train(not eval_mode)
            if which == "both":
                src = torch.cat([torch.zeros(n_set[0], dtype=torch.long),
                                 torch.ones(n_set[1], dtype=torch.long)])
                ids = torch.cat([torch.arange(n_set[0]), torch.arange(n_set[1])])
            else:
                k = common.RETAIN if which == "retain" else common.FORGET
                src = torch.full((n_set[k],), k, dtype=torch.long)
                ids = torch.arange(n_set[k])
            order = torch.randperm(len(src), generator=gen)
            src, ids = src[order], ids[order]
            if which in ("retain", "both"):
                used[0] += n_set[0]
            if which in ("forget", "both"):
                used[1] += n_set[1]
            if mode == "accumulate":
                for p in params.values():
                    p.grad.zero_()
            for step, s in enumerate(range(0, len(src), bs)):
                bsrc, bids = src[s:s + bs], ids[s:s + bs]
                nb = len(bsrc)
                sel_f = bsrc == common.FORGET
                xb = torch.empty(nb, 3, 32, 32, dtype=torch.uint8, device=device)
                yb = torch.empty(nb, dtype=torch.long, device=device)
                for k, sel in ((common.RETAIN, ~sel_f), (common.FORGET, sel_f)):
                    if bool(sel.any()):
                        ii = bids[sel].to(device)
                        xb[sel.to(device)] = x[k][ii]
                        yb[sel.to(device)] = y[k][ii]
                xb = common.gpu_augment(xb, spec["augment"])
                if mode == "train":
                    for p in params.values():
                        p.grad.zero_()
                logits = model(xb)
                logits_buf[:nb].copy_(logits.detach())
                if device.type == "cuda":
                    torch.cuda.synchronize()
                head = struct.pack("<iiiii", OP_LOSS, pi, step, nb, 1 if mode == "train" else 0)
                _write_all(resp_w, head + bids.to(torch.int32).numpy().tobytes()
                           + bsrc.to(torch.int8).numpy().tobytes())
                reply(call_deadline())
                g = dlogits_buf[:nb].clone()
                if not bool(torch.isfinite(g).all()):
                    raise WorkerError("non-finite loss gradient")
                logits.backward(g)
                if any(params[k].grad is None or params[k].grad.data_ptr() != grad_ptrs[k]
                       for k in names):
                    raise WorkerError("internal: gradient buffer was reallocated")
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if mode == "train":
                    _write_all(resp_w, struct.pack("<iii", OP_UPDATE, pi, step))
                    reply(call_deadline())
            _write_all(resp_w, struct.pack("<iii", OP_PASS_END, pi, 0))
            reply(call_deadline())
        _write_all(resp_w, struct.pack("<iii", OP_FINISH, 0, 0))
        reply(call_deadline())
    except TimeoutError:
        kill_worker()
        return fail(f"time limit exceeded (a hook call over {CALL_SEC}s or the run over "
                    f"{args.time_limit_sec:.0f}s + {SETUP_SEC}s setup)")
    except EOFError:
        kill_worker()
        return fail("worker exited unexpectedly")
    except WorkerError as exc:
        kill_worker()
        return fail(f"method error: {str(exc)[-3000:]}")
    kill_worker()          # also stops anything it left running on the shared tensors
    wall = time.time() - t_start
    if device.type == "cuda":
        torch.cuda.synchronize()

    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if not all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
        return fail("non-finite weights")
    cos = common.descent_cosine(state, original)
    status.update(wall_sec=round(wall, 2), retain_samples=used[0], forget_samples=used[1],
                  descent_cosine=round(cos, 4), split_seed=str(split_seed),
                  forget_idx=[int(i) for i in forget_idx])
    if cos < common.MIN_DESCENT_COSINE:
        return fail(f"not an update of the Original model: conv/linear weight cosine {cos:.3f} "
                    f"< {common.MIN_DESCENT_COSINE}")
    torch.save(state, ckpt_path)
    status["valid"] = True
    status_path.write_text(json.dumps(status))
    print(f"UNLEARN_DONE wall_sec={wall:.1f} retain_samples={used[0]} forget_samples={used[1]} "
          f"descent_cosine={cos:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
