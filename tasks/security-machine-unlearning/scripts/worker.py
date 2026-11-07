"""Sandboxed worker for security-machine-unlearning (READ-ONLY).

Started by run_unlearning.py.  Maps the ONE shared GPU buffer that holds the
model's parameters, their gradients and the logits / loss-gradient buffers,
installs a Landlock file-system sandbox, THEN compiles the editable
custom_unlearning.py and serves the parent's hook calls:

* ``plan(ctx)``            -> list of passes, sent to the parent as JSON;
* ``begin(params, ctx)``   once, before the first pass (optional);
* ``loss(logits, labels, is_forget, ids, info)`` per batch -> scalar; the
  worker differentiates it w.r.t. the logits and writes that gradient into
  the shared buffer, the parent back-propagates it through the model;
* ``update(params, info)`` after the parent's backward pass of a "train"
  pass; ``params`` are nn.Parameters sharing the parent's memory, with
  ``.grad`` holding this batch's gradient;
* ``end_pass(params, info)`` after every pass (for an "accumulate" pass
  ``.grad`` holds the gradient summed over the pass);
* ``finish(params, ctx)``  once, at the end (optional).

This process never receives an image, so nothing it does can train the
network beyond the batches the parent runs.  Sandbox (Landlock ABI >= 1,
unprivileged, fails CLOSED): read-only access to the system and Python
trees only -- the fixed list below plus this interpreter's prefix and
site-packages (no dataset, no checkpoints, no workspace, no task or /tests
files, no /root, /home or shared /tmp), read-write only to a private scratch
dir and /dev; no ptrace of processes outside the sandbox.
"""

import argparse
import ctypes
import json
import os
import pickle
import struct
import sys
import traceback

OP_LOSS, OP_UPDATE, OP_PASS_END, OP_FINISH = 1, 2, 3, 4
ST_OK, ST_ERR = 0, 1

_EXEC, _WRITE, _READ, _READ_DIR = 1, 2, 4, 8
_ALL_V1 = (1 << 13) - 1
_FILE_RIGHTS = _EXEC | _WRITE | _READ
_RO = _EXEC | _READ | _READ_DIR
# The only trees the method may read after the sandbox is installed.
_READ_ONLY = ("/usr", "/lib", "/lib32", "/lib64", "/libx32", "/bin", "/sbin", "/etc",
              "/opt/conda", "/proc", "/sys")


def _landlock(rw_dirs):
    libc = ctypes.CDLL(None, use_errno=True)
    abi = libc.syscall(444, None, 0, 1)
    if abi < 1:
        raise RuntimeError(f"Landlock unavailable (abi={abi}, errno={ctypes.get_errno()})")

    class RulesetAttr(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathBeneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    attr = RulesetAttr(_ALL_V1)
    rfd = libc.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if rfd < 0:
        raise RuntimeError(f"landlock_create_ruleset failed errno={ctypes.get_errno()}")

    def allow(path, rights):
        try:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        except OSError:
            return
        try:
            if not os.path.isdir(path):
                rights &= _FILE_RIGHTS
            pb = PathBeneath(rights, fd)
            if libc.syscall(445, rfd, 1, ctypes.byref(pb), 0) != 0:
                raise RuntimeError(f"landlock_add_rule({path}) errno={ctypes.get_errno()}")
        finally:
            os.close(fd)

    import site
    ro = list(_READ_ONLY) + [sys.prefix, sys.base_prefix, sys.exec_prefix]
    ro += [d for d in site.getsitepackages() + [site.getusersitepackages() or ""] if d]
    for p in ro:
        allow(p, _RO)
    for d in rw_dirs:
        allow(d, _ALL_V1)
    if libc.prctl(38, 1, 0, 0, 0) != 0:          # PR_SET_NO_NEW_PRIVS
        raise RuntimeError("prctl(NO_NEW_PRIVS) failed")
    if libc.syscall(446, rfd, 0) != 0:
        raise RuntimeError(f"landlock_restrict_self failed errno={ctypes.get_errno()}")
    os.close(rfd)


def _read_exact(fd, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = os.read(fd, min(n - len(buf), 1 << 22))
        if not chunk:
            raise EOFError("parent closed the pipe")
        buf += chunk
    return bytes(buf)


def _write_all(fd, data):
    view = memoryview(data)
    while view:
        k = os.write(fd, view)
        view = view[k:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--req-fd", type=int, required=True)
    ap.add_argument("--resp-fd", type=int, required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--custom-name", required=True)
    a = ap.parse_args()
    req, resp = a.req_fd, a.resp_fd

    def recv():
        n, = struct.unpack("<q", _read_exact(resp, 8))
        return _read_exact(resp, n)

    def ok():
        _write_all(req, struct.pack("<i", ST_OK))

    def err(msg):
        b = msg[-8000:].encode("utf-8", "replace")
        _write_all(req, struct.pack("<ii", ST_ERR, len(b)) + b)

    import numpy as np
    import torch
    import torch.multiprocessing  # noqa: F401  (CUDA IPC reductions)
    import torch.nn.functional as F  # noqa: F401

    shared = pickle.loads(recv())            # pickled by the trusted parent
    flat, layout = shared["flat"], shared["layout"]
    labels_np = [np.frombuffer(recv(), dtype=np.int64).copy() for _ in range(2)]
    ctx = json.loads(recv().decode())
    custom_src = recv()
    device = flat.device

    params, bufs = {}, {}
    for name, kind, off, shape in layout:
        n = 1
        for d in shape:
            n *= d
        view = flat[off:off + n].view(*shape)
        if kind == "param":
            params[name] = torch.nn.Parameter(view, requires_grad=True)
        elif kind == "grad":
            params[name].grad = view
        else:
            bufs[name] = view
    labels = [torch.from_numpy(l).to(device) for l in labels_np]
    torch.manual_seed(int(ctx["seed"]))
    torch.cuda.manual_seed_all(int(ctx["seed"]))
    if device.type == "cuda":
        torch.cuda.synchronize()

    _landlock([a.scratch, "/dev"])

    from types import SimpleNamespace
    ctx_ns = SimpleNamespace(**ctx, device=device)
    try:
        import types
        custom = types.ModuleType("custom_unlearning")
        custom.__file__ = a.custom_name
        sys.modules["custom_unlearning"] = custom
        exec(compile(custom_src, a.custom_name, "exec"), custom.__dict__)
        method = custom.UnlearningMethod()
        plan = method.plan(ctx_ns)
        plan_b = json.dumps(plan).encode()
    except Exception:  # noqa: BLE001
        msg = traceback.format_exc()
        print(msg, flush=True)
        b = ("ERR" + msg[-8000:]).encode("utf-8", "replace")
        _write_all(req, struct.pack("<q", len(b)) + b)
        return 1
    _write_all(req, struct.pack("<q", len(plan_b)) + plan_b)

    begun = False
    try:
        while True:
            op, pi, step = struct.unpack("<iii", _read_exact(resp, 12))
            if not begun:
                if hasattr(method, "begin"):
                    method.begin(params, ctx_ns)
                begun = True
            if op == OP_LOSS:
                nb, train = struct.unpack("<ii", _read_exact(resp, 8))
                ids = torch.from_numpy(np.frombuffer(_read_exact(resp, 4 * nb), dtype=np.int32)
                                       .astype(np.int64)).to(device)
                src = torch.from_numpy(np.frombuffer(_read_exact(resp, nb), dtype=np.int8)
                                       .astype(np.int64)).to(device)
                is_forget = src == 1
                lab = torch.where(is_forget, labels[1][ids.clamp(max=len(labels[1]) - 1)],
                                  labels[0][ids.clamp(max=len(labels[0]) - 1)])
                logits = bufs["logits"][:nb].detach().clone().requires_grad_(True)
                info = SimpleNamespace(pass_index=pi, step=step, train=bool(train))
                loss = method.loss(logits, lab, is_forget, ids, info)
                (g,) = torch.autograd.grad(loss, logits)
                bufs["dlogits"][:nb].copy_(g)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                ok()
            elif op == OP_UPDATE:
                method.update(params, SimpleNamespace(pass_index=pi, step=step))
                if device.type == "cuda":
                    torch.cuda.synchronize()
                ok()
            elif op == OP_PASS_END:
                if hasattr(method, "end_pass"):
                    method.end_pass(params, SimpleNamespace(pass_index=pi))
                if device.type == "cuda":
                    torch.cuda.synchronize()
                ok()
            elif op == OP_FINISH:
                if hasattr(method, "finish"):
                    method.finish(params, ctx_ns)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                ok()
                return 0
            else:
                err(f"unknown op {op}")
                return 1
    except Exception:  # noqa: BLE001
        msg = traceback.format_exc()
        print(msg, flush=True)
        try:
            err(msg)
        except OSError:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
