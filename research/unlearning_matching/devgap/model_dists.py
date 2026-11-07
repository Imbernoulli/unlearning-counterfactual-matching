"""Per-sample clean/eval true-class confidence and error of saved checkpoints on
one deletion split's development subsets (forget, retain_subset, target,
selection, audit).  Same preprocessing as the benchmark evaluator
(common.predict_probs).  Never opens the official test split.

Output npz keys follow calibration_study's epoch files: <subset>_confidence,
<subset>_error, <subset>_labels, plus <subset>_logodds / _margin.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "tasks/security-machine-unlearning/scripts"


def load_common():
    spec = importlib.util.spec_from_file_location("devgap_common", TASK / "common.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def subsets(split_json, sel_json, y):
    idx = json.loads(Path(split_json).read_text())["indices"]
    retain = np.asarray(idx["retain"], dtype=np.int64)
    pos = np.sort(np.random.RandomState(7312026).choice(len(retain), 5000, replace=False))
    sel = json.loads(Path(sel_json).read_text())
    out = {"forget": np.asarray(idx["forget"], dtype=np.int64), "retain_subset": retain[pos],
           "target": np.asarray(idx["target"], dtype=np.int64),
           "selection": np.asarray(sel["selection"], dtype=np.int64),
           "audit": np.asarray(sel["audit"], dtype=np.int64)}
    return out


@torch.no_grad()
def dists(common, state, x, y, sets, device):
    model = common.build_model("cifar10")
    model.load_state_dict(state, strict=True)
    model = model.to(device).float().eval()
    out = {}
    for name, ids in sets.items():
        logits = []
        for start in range(0, len(ids), 1000):
            xb = common.to_gpu_u8(x[ids[start:start + 1000]], device).float().div_(255.)
            logits.append(model(xb).double().cpu())
        z = torch.cat(logits)
        lab = torch.from_numpy(y[ids])
        zy = z.gather(1, lab[:, None]).squeeze(1)
        other = z.clone()
        other.scatter_(1, lab[:, None], -float("inf"))
        out[f"{name}_logodds"] = (zy - torch.logsumexp(other, 1)).float().numpy()
        out[f"{name}_margin"] = (zy - other.max(1).values).float().numpy()
        out[f"{name}_confidence"] = torch.softmax(z, 1).gather(1, lab[:, None]).squeeze(1).numpy()
        out[f"{name}_error"] = (z.argmax(1) != lab).numpy()
        out[f"{name}_labels"] = lab.numpy()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--selection-audit", required=True)
    p.add_argument("--ckpt", nargs="+", required=True, help="name=path pairs")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    common = load_common()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(a.data) as z:
        x, y = z["x_train"], z["y_train"].astype(np.int64)
    sets = subsets(a.splits, a.selection_audit, y)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for item in a.ckpt:
        name, path = item.split("=", 1)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        np.savez_compressed(a.out_dir / f"{name}.npz", **dists(common, state, x, y, sets, device))
        print(name, "done", flush=True)


if __name__ == "__main__":
    main()
