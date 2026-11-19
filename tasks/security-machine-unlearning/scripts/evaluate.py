"""Evaluation for security-machine-unlearning (READ-ONLY; separate process).

Reports the four accuracy-type metrics of SalUn (arXiv:2310.12508, Sec. 3 and
Table 1), all in percent, for the saved unlearned model:

* UA  -- unlearning accuracy, 100 - accuracy on the forget set D_f;
* RA  -- remaining accuracy, accuracy on the retain set D_r;
* TA  -- test accuracy on the official test split;
* MIA -- MIA-Efficacy: a confidence-based membership-inference predictor
  (RBF SVC, C=3, gamma='auto', on the true-class probability) is fit on
  members = the first |test| samples of D_r and non-members = the test set,
  then applied to D_f; MIA-Efficacy is the fraction of D_f predicted as
  NON-members (Classification/evaluation/SVC_MIA.py:55-90 and
  main_forget.py:165-185, key 'confidence').

All sets are evaluated without augmentation.  Before scoring, the model is
checked to descend from the Original: the global cosine of its learnable
parameters to the Original's must be >= common.MIN_DESCENT_COSINE.  The gap to the Retrain
reference (a model trained from scratch on D_r only) is computed by the
verifier, which alone holds the reference values.

The two files this process reads (run_status.json, unlearned.pt) were written
by the unlearning phase, which under Harbor ran as an unprivileged uid that
has since been killed; they are treated as hostile input: a regular file only
(no symlink), size-capped, the checkpoint loaded with weights_only=True and
required to be exactly the architecture's state_dict (keys, shapes, dtypes,
parameter count) before anything is computed from it.
"""

import argparse
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("unlearn_common_eval", _HERE / "common.py")
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)

# Verifier-side test split (Harbor: staged from holdout/<task>/data/withheld/ into the
# sealed task directory this file runs from; cifar10_test.npz, cifar100_test.npz).
# Natively the image's npz carries the test split and this directory is not read.
TEST_DIR = _HERE.parent / "data" / "withheld"

MAX_STATUS_BYTES = 4 << 20          # run_status.json: a few KB of JSON + the forget indices


def open_artifact(path, max_bytes):
    """Open a phase-A artifact for reading: a regular, non-symlinked file of at most
    ``max_bytes`` (O_NOFOLLOW; the size is taken from the opened descriptor)."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"{path.name} is a symlink")
    fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ValueError(f"{path.name} is not a regular file")
        if st.st_size > max_bytes:
            raise ValueError(f"{path.name} is {st.st_size} bytes (cap {max_bytes})")
    except BaseException:
        os.close(fd)
        raise
    return os.fdopen(fd, "rb")


def load_checkpoint(ckpt_path, dataset):
    """The saved state_dict, verified to be exactly the architecture's."""
    ref = common.build_model(dataset).state_dict()
    ref_bytes = sum(v.numel() * v.element_size() for v in ref.values())
    with open_artifact(ckpt_path, 2 * ref_bytes + (8 << 20)) as fh:
        state = torch.load(fh, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or set(state) != set(ref):
        raise ValueError("checkpoint keys are not the architecture's state_dict keys")
    for k, v in ref.items():
        t = state[k]
        if not isinstance(t, torch.Tensor) or tuple(t.shape) != tuple(v.shape) or t.dtype != v.dtype:
            raise ValueError(f"checkpoint entry {k} is not a {tuple(v.shape)} {v.dtype} tensor")
    n_ref = sum(v.numel() for v in ref.values())
    n_got = sum(t.numel() for t in state.values())
    if n_got != n_ref:
        raise ValueError(f"checkpoint holds {n_got} values, the architecture {n_ref}")
    return state


def svc_mia_efficacy(shadow_train_conf, shadow_test_conf, target_conf):
    from sklearn.svm import SVC
    x = np.concatenate([shadow_train_conf, shadow_test_conf]).reshape(-1, 1)
    y = np.concatenate([np.ones(len(shadow_train_conf)), np.zeros(len(shadow_test_conf))])
    clf = SVC(C=3, gamma="auto", kernel="rbf")
    clf.fit(x, y)
    return float(1.0 - clf.predict(target_conf.reshape(-1, 1)).mean())


def evaluate_state(dataset, state, forget_idx, arrays=None, forget_fraction=0.10, *, model_factory=None):
    """UA / RA / TA / MIA-Efficacy (percent) for a classifier state_dict.

    The default model is the benchmark ResNet-18. Research callers may pass
    a zero-argument model factory to reuse these metrics on another architecture.

    ``forget_idx``: the forget set (indices into the official train split)
    the run used, which must be a ``forget_fraction`` subset of D; the retain
    set is the rest of D, in index order.
    """
    forget_fraction = common.check_forget_fraction(forget_fraction)
    spec = common.DATASETS[dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common.set_determinism(0)
    x_tr, y_tr, x_te, y_te = arrays if arrays is not None else common.load_arrays(dataset, TEST_DIR)
    train_idx, _, _, _ = common.split_indices(y_tr, spec["num_classes"], 0, forget_fraction)
    forget_idx = np.asarray(sorted(int(i) for i in forget_idx), dtype=np.int64)
    if len(set(forget_idx.tolist())) != len(forget_idx) or not np.isin(forget_idx, train_idx).all() \
            or len(forget_idx) != int(round(forget_fraction * len(train_idx))):
        raise ValueError(f"forget set is not a {100 * forget_fraction:.0f}% subset of D")
    retain_idx = train_idx[~np.isin(train_idx, forget_idx)]
    model = common.build_model(dataset) if model_factory is None else model_factory()
    model.load_state_dict(state, strict=True)
    model = model.to(device).float()

    def probs_of(x, y):
        p = common.predict_probs(model, common.to_gpu_u8(x, device))
        yt = torch.from_numpy(y).to(device)
        if not torch.isfinite(p).all():          # a diverged model scores as a failed one
            return 0.0, np.zeros(len(y))
        acc = (p.argmax(1) == yt).float().mean().item()
        return acc, p.gather(1, yt[:, None]).squeeze(1).double().cpu().numpy()

    acc_f, conf_f = probs_of(x_tr[forget_idx], y_tr[forget_idx])
    acc_r, conf_r = probs_of(x_tr[retain_idx], y_tr[retain_idx])
    acc_t, conf_t = probs_of(x_te, y_te)
    n_test = len(y_te)
    mia = svc_mia_efficacy(conf_r[:n_test], conf_t, conf_f)
    return {"ua": 100 * (1 - acc_f), "ra": 100 * acc_r, "ta": 100 * acc_t, "mia": 100 * mia,
            "n_forget": len(forget_idx), "n_retain": len(retain_idx), "n_test": n_test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(common.DATASETS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--forget-fraction", type=float, default=0.10)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    status_path, ckpt_path = out_dir / "run_status.json", out_dir / "unlearned.pt"
    try:
        with open_artifact(status_path, MAX_STATUS_BYTES) as fh:
            status = json.loads(fh.read().decode("utf-8"))
        if not isinstance(status, dict):
            raise ValueError("run status is not a JSON object")
    except Exception as exc:  # noqa: BLE001
        print(f"UNLEARN_EVAL_FAILED no run status ({exc})", flush=True)
        return 2
    try:
        frac = common.check_forget_fraction(args.forget_fraction)
        status_frac = float(status.get("forget_fraction", -1.0))
    except (TypeError, ValueError) as exc:
        print(f"UNLEARN_EVAL_FAILED bad forget fraction ({exc})", flush=True)
        return 2
    if abs(status_frac - frac) > 1e-9:
        print(f"UNLEARN_EVAL_FAILED the run used forget fraction {status_frac}, the setting "
              f"expects {frac}", flush=True)
        return 2
    if not status.get("valid") or status.get("dataset") != args.dataset or \
            int(status.get("seed", -1)) != args.seed or abs(status_frac - frac) > 1e-9 or \
            not ckpt_path.exists():
        print(f"UNLEARN_EVAL_FAILED invalid run: {status.get('reason', 'no checkpoint')}", flush=True)
        return 2

    try:
        state = load_checkpoint(ckpt_path, args.dataset)
    except Exception as exc:  # noqa: BLE001 - a malformed checkpoint is a failed run
        print(f"UNLEARN_EVAL_FAILED checkpoint rejected: {str(exc)[:300]}", flush=True)
        return 2
    try:
        arrays, original = common.load_verified(args.dataset, TEST_DIR)   # pinned sha256
    except Exception as exc:  # noqa: BLE001
        print(f"UNLEARN_EVAL_FAILED {exc}", flush=True)
        return 2
    # Descent check (also done by the trusted parent before it saved the file).
    try:
        min_cos = common.descent_cosine(state, original)
    except Exception as exc:  # noqa: BLE001 - wrong keys / shapes
        print(f"UNLEARN_EVAL_FAILED descent check could not run: {exc}", flush=True)
        return 2
    if min_cos < common.MIN_DESCENT_COSINE:
        print(f"UNLEARN_EVAL_FAILED not an update of the Original model: conv/linear weight "
              f"cosine {min_cos:.3f} < {common.MIN_DESCENT_COSINE}", flush=True)
        return 2
    try:
        m = evaluate_state(args.dataset, state, status["forget_idx"], arrays, frac)
    except Exception as exc:  # noqa: BLE001
        print(f"UNLEARN_EVAL_FAILED {exc}", flush=True)
        return 2
    ua, ra, ta, mia = m["ua"], m["ra"], m["ta"], m["mia"]
    n_forget, n_retain, n_test = m["n_forget"], m["n_retain"], m["n_test"]
    try:
        rte_sec = float(status.get("wall_sec", 0.0))     # informational, not scored
    except (TypeError, ValueError):
        rte_sec = 0.0
    print(f"UNLEARN_EVAL label={args.label} seed={args.seed} ua={ua:.4f} ra={ra:.4f} "
          f"ta={ta:.4f} mia={mia:.4f} rte_sec={rte_sec:.1f} "
          f"descent_cos={min_cos:.4f} "
          f"n_forget={n_forget} n_retain={n_retain} n_test={n_test}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
