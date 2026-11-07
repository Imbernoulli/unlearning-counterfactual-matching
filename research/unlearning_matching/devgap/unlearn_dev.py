"""Dense-schedule target-matching unlearning with per-epoch dev tracking.

Method (fixed targets, no Retrain or test data in the update):
  * Original clean/eval true-class log-odds s(z,y)=z_y-logsumexp_{k!=y} z_k on
    forget F and on the 2500 target-validation images V.
  * Per class, sort F by Original log-odds and assign each rank the V
    empirical quantile at its rank midpoint (1-D barycentric OT target).
    Optional affine change of the target: t <- scale*t + shift.
  * Loss on clean forget batches (eval-mode BN): MSE between current and
    target log-odds ("logodds_mse"), or KL to the I-projection of Original's
    distribution onto the target true-class probability ("kl_iproj").
  * Schedule: one complete augmented retain pass per epoch (CE); a forget batch
    is added at every k-th retain step, cycling through reshuffled F.  The
    forget gradient (alpha-weighted) is clipped to norm 5 before the retain
    gradient is added.  SGD momentum .9, wd 5e-4.

Dev tracking (never test): per-epoch distributions on forget / retain subset /
target / selection, dev metrics (devgap.py) and gap to a paired Retrain's dev
metrics when --retrain-ref is given.  The audit fold is never predicted.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import devgap  # noqa: E402
from model_dists import subsets  # noqa: E402


def load_common():
    spec = importlib.util.spec_from_file_location("ud_common", ROOT / "tasks/security-machine-unlearning/scripts/common.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def scores(z, y):
    zy = z.gather(1, y[:, None]).squeeze(1)
    other = z.scatter(1, y[:, None], float("-inf"))
    return zy - torch.logsumexp(other, 1), zy - other.max(1).values


@torch.no_grad()
def logits_of(model, x_u8, bs=1000):
    model.eval()
    return torch.cat([model(x_u8[i:i + bs].float().div(255.)) for i in range(0, len(x_u8), bs)])


DATASIZE_MAP = None  # (levels, delta): measured counterfactual quantile correction, added at each rank level


def gq_targets(s_f, y_f, s_v, y_v):
    t = torch.empty_like(s_f)
    for c in range(10):
        fi = torch.nonzero(y_f == c).squeeze(1)
        order = fi[torch.argsort(s_f[fi])]
        sv = torch.sort(s_v[y_v == c]).values.double().cpu().numpy()
        n = len(order)
        levels = (np.arange(n) + .5) / n
        q = np.quantile(sv, levels)
        if DATASIZE_MAP is not None:
            q = q + np.interp(levels, DATASIZE_MAP[0], DATASIZE_MAP[1])
        t[order] = torch.from_numpy(q).to(t)
    return t


@torch.no_grad()
def dists(model, sets):
    out = {}
    for name, (x, y) in sets.items():
        z = logits_of(model, x).double()
        s, m = scores(z, y)
        out[f"{name}_logodds"] = s.float().cpu().numpy()
        out[f"{name}_margin"] = m.float().cpu().numpy()
        out[f"{name}_confidence"] = torch.softmax(z, 1).gather(1, y[:, None]).squeeze(1).cpu().numpy()
        out[f"{name}_error"] = (z.argmax(1) != y).cpu().numpy()
        out[f"{name}_labels"] = y.cpu().numpy()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--original", required=True)
    p.add_argument("--init", help="start from this checkpoint instead of Original (targets still from Original)")
    p.add_argument("--splits", required=True)
    p.add_argument("--selection-audit", required=True)
    p.add_argument("--retrain-ref", help="dev distributions npz of the paired Retrain (dev seed only)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=.003)
    p.add_argument("--alpha", type=float, default=1.)
    p.add_argument("--forget-every", type=int, default=1)
    p.add_argument("--forget-epochs", type=int, default=0, help="forget steps only in the first N epochs (0 = all)")
    p.add_argument("--ema-start-epoch", type=int, default=0,
                   help="if >0: keep a uniform running average of parameters over all optimizer steps from this epoch on; "
                        "the averaged weights are evaluated at epoch ends and returned (budget-neutral, no extra passes)")
    p.add_argument("--decay-epochs", type=int, default=0, help="linearly decay lr to 0 over the last N epochs")
    p.add_argument("--sequential", action="store_true",
                   help="harness-style schedule: per epoch one full forget pass (train, target loss) then one full retain pass")
    p.add_argument("--aug-forget", action="store_true", help="augment forget batches as the benchmark harness does")
    p.add_argument("--loss", choices=["logodds_mse", "logodds_huber", "kl_iproj"], default="logodds_mse")
    p.add_argument("--target-scale", type=float, default=1.)
    p.add_argument("--target-shift", type=float, default=0.)
    p.add_argument("--dynamic-ranks", action="store_true", help="re-assign quantile ranks from current scores each epoch")
    p.add_argument("--target-map", help="npz with levels/delta: measured quantile correction (calibration split only)")
    p.add_argument("--tail-tau", type=float, default=None,
                   help="tail-focused loss: samples with target log-odds < tau get two-sided error to target; samples "
                        "with target >= tau only a one-sided penalty if they fall below target (bulk left alone)")
    p.add_argument("--target-ref", choices=["original", "current"], default="original",
                   help="held-out reference: frozen Original outputs on V, or the current model's own outputs on V "
                        "(exchangeability target; recomputed every --refresh-steps steps, forget re-ranked too)")
    p.add_argument("--refresh-steps", type=int, default=20)
    p.add_argument("--mixed", action="store_true",
                   help="benchmark 'both' pass: per epoch one shuffled pass over retain+forget in 256-batches; each "
                        "batch's forget part gets the target loss, its retain part CE (1 retain + 1 forget pass/epoch)")
    p.add_argument("--bn", choices=["frozen", "train"], default="frozen")
    p.add_argument("--clip", type=float, default=5.)
    p.add_argument("--save-epochs", default="", help="comma list of epochs whose checkpoints are kept")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    if a.target_ref == "current" and (a.loss == "kl_iproj" or a.dynamic_ranks):
        raise SystemExit("--target-ref current implies dynamic ranks and supports logodds_mse only")
    if a.mixed and a.loss == "kl_iproj":
        raise SystemExit("--mixed supports log-odds losses only")

    def score_loss(s, t):
        r = s - t
        if a.tail_tau is not None:
            r = torch.where(t < a.tail_tau, r, torch.clamp(r, max=0.))
        if a.loss == "logodds_mse":
            return (r ** 2).mean()
        return F.smooth_l1_loss(r, torch.zeros_like(r), beta=1.0)
    torch.set_num_threads(a.threads)
    if a.target_map:
        global DATASIZE_MAP
        with np.load(a.target_map) as z:
            DATASIZE_MAP = (z["levels"], z["delta"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    common = load_common()
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / "summary.json").exists():
        print("already complete"); return
    (a.out / "config.json").write_text(json.dumps(vars(a), default=str, indent=1))
    dev = torch.device("cuda")
    with np.load(a.data) as z:
        x, y = z["x_train"], z["y_train"].astype(np.int64)
    idx = subsets(a.splits, a.selection_audit, y)
    retain = np.asarray(json.loads(Path(a.splits).read_text())["indices"]["retain"], dtype=np.int64)

    def gpu(ids):
        return common.to_gpu_u8(x[ids], dev), torch.from_numpy(y[ids]).to(dev)

    xf, yf = gpu(idx["forget"])
    xr, yr = gpu(retain)
    sets = {k: gpu(idx[k]) for k in ("forget", "retain_subset", "target", "selection")}
    ref = devgap.metrics(np.load(a.retrain_ref)) if a.retrain_ref else None
    ref_sel = devgap.metrics(np.load(a.retrain_ref), held=("selection",)) if a.retrain_ref else None

    orig_state = torch.load(a.original, map_location="cpu", weights_only=True)
    teacher = common.build_model("cifar10"); teacher.load_state_dict(orig_state); teacher = teacher.to(dev).eval()
    zf0, zv0 = logits_of(teacher, xf).double(), logits_of(teacher, sets["target"][0]).double()
    sf0, _ = scores(zf0, yf)
    sv0, _ = scores(zv0, sets["target"][1])
    base_t = gq_targets(sf0, yf, sv0, sets["target"][1])
    p0 = torch.softmax(zf0, 1)
    del teacher

    def make_targets(rank_scores, ref_scores=None):
        sv = sv0 if ref_scores is None else ref_scores
        t = gq_targets(rank_scores, yf, sv, sets["target"][1]) if rank_scores is not None else base_t.clone()
        return (a.target_scale * t + a.target_shift).float()

    @torch.no_grad()
    def current_targets():
        sf_now, _ = scores(logits_of(model, xf).double(), yf)
        sv_now, _ = scores(logits_of(model, sets["target"][0]).double(), sets["target"][1])
        return make_targets(sf_now, sv_now)

    targets = make_targets(None)

    def iproj(t):  # log-probs of I-projection of Original onto true-class prob sigmoid(t)
        u = torch.sigmoid(t.double())
        q = p0 * ((1 - u) / (1 - p0.gather(1, yf[:, None]).squeeze(1)).clamp_min(1e-300))[:, None]
        q.scatter_(1, yf[:, None], u[:, None])
        return q.clamp_min(1e-30).log().float()

    logq = iproj(targets) if a.loss == "kl_iproj" else None

    model = common.build_model("cifar10")
    model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=True) if a.init else orig_state)
    model = model.to(dev)
    common.set_determinism(a.seed)
    torch.backends.cudnn.benchmark = True
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=.9, weight_decay=5e-4)
    save_epochs = {int(e) for e in a.save_epochs.split(",") if e}
    hist = []
    best = None

    def swap_in_average():
        if avg is None or avg_n == 0:
            return None
        saved = [p_.detach().clone() for p_ in model.parameters()]
        with torch.no_grad():
            for pm, pa in zip(model.parameters(), avg):
                pm.copy_(pa)
        return saved

    def swap_back(saved):
        if saved is not None:
            with torch.no_grad():
                for pm, ps in zip(model.parameters(), saved):
                    pm.copy_(ps)

    def record(epoch, train_s):
        nonlocal best
        saved = swap_in_average()
        d = dists(model, sets)
        np.savez_compressed(a.out / f"epoch{epoch:03d}.npz", **d)
        m = devgap.metrics(d)
        row = dict(epoch=epoch, train_s=train_s, forget_passes=forget_samples / len(yf), **m)
        if ref:
            row.update({f"gap_{k}": v for k, v in devgap.gap(m, ref).items()})
            # selection-only held-out: excludes the target fold used to build (current) targets
            ms = devgap.metrics(d, held=("selection",))
            row.update({f"sel_{k}": v for k, v in ms.items()})
            row["sel_gap_avg"] = devgap.gap(ms, ref_sel)["avg"]
        # Retrain-free proxy: forget should look like held-out under this model
        row["self_gap"] = (abs(m["ua"] - (100 - m["ta"])) + abs(m["mia"] - m["held_nonmember_rate"])) / 2
        hist.append(row)
        with (a.out / "history.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        if ref and (best is None or row["gap_avg"] < best["gap_avg"]):
            best = row
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, a.out / "best_dev.pt.tmp")
            os.replace(a.out / "best_dev.pt.tmp", a.out / "best_dev.pt")
        if epoch in save_epochs:
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, a.out / f"epoch{epoch:03d}.pt")
        if epoch == a.epochs:
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, a.out / "last.pt")
        swap_back(saved)

    (a.out / "history.jsonl").unlink(missing_ok=True)
    forget_samples = 0
    avg = [p_.detach().clone() for p_ in model.parameters()] if a.ema_start_epoch > 0 else None
    avg_n = 0
    record(0, 0.)
    train_s = 0.
    for epoch in range(1, a.epochs + 1):
        if a.dynamic_ranks and epoch > 1:
            sf_now, _ = scores(logits_of(model, xf).double(), yf)
            targets = make_targets(sf_now)
            logq = iproj(targets) if a.loss == "kl_iproj" else None
        torch.cuda.synchronize(); t0 = time.monotonic()
        if a.sequential:
            nb_total = (len(yf) + 255) // 256 + (len(yr) + 255) // 256
            step = 0
            for phase, (xx, yy) in (("forget", (xf, yf)), ("retain", (xr, yr))):
                for ids in torch.randperm(len(yy), device=dev).split(256):
                    if a.decay_epochs and epoch > a.epochs - a.decay_epochs:
                        done = (epoch - (a.epochs - a.decay_epochs) - 1) * nb_total + step
                        for g in opt.param_groups:
                            g["lr"] = a.lr * (1. - done / (nb_total * a.decay_epochs))
                    opt.zero_grad(set_to_none=True)
                    if phase == "forget":
                        forget_samples += len(ids)
                        model.eval()
                        z = model(common.gpu_augment(xx[ids], True) if a.aug_forget else xx[ids].float().div(255.))
                        s, _ = scores(z, yy[ids])
                        (a.alpha * score_loss(s, targets[ids])).backward()
                        if a.clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
                    else:
                        model.train(a.bn == "train")
                        F.cross_entropy(model(common.gpu_augment(xx[ids], True)), yy[ids]).backward()
                    opt.step()
                    step += 1
            torch.cuda.synchronize(); train_s += time.monotonic() - t0
            record(epoch, train_s)
            continue
        if a.mixed:
            union = torch.randperm(len(yr) + len(yf), device=dev)
            for step, ub in enumerate(union.split(256)):
                if a.target_ref == "current" and step % a.refresh_steps == 0:
                    targets = current_targets()
                opt.zero_grad(set_to_none=True)
                fid, rid = ub[ub >= len(yr)] - len(yr), ub[ub < len(yr)]
                forget_samples += len(fid)
                if a.alpha > 0 and len(fid):
                    model.eval()
                    z = model(xf[fid].float().div(255.))
                    s, _ = scores(z, yf[fid])
                    (a.alpha * score_loss(s, targets[fid])).backward()
                    if a.clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
                if len(rid):
                    model.train(a.bn == "train")
                    F.cross_entropy(model(common.gpu_augment(xr[rid], True)), yr[rid]).backward()
                opt.step()
            torch.cuda.synchronize(); train_s += time.monotonic() - t0
            record(epoch, train_s)
            continue
        r_batches = torch.randperm(len(yr), device=dev).split(256)
        f_queue = []
        for step, rid in enumerate(r_batches):
            if a.target_ref == "current" and step % a.refresh_steps == 0:
                targets = current_targets()
                model.train(a.bn == "train")
            opt.zero_grad(set_to_none=True)
            if a.alpha > 0 and step % a.forget_every == 0 and (a.forget_epochs == 0 or epoch <= a.forget_epochs):
                if not f_queue:
                    f_queue = list(torch.randperm(len(yf), device=dev).split(256))
                fid = f_queue.pop()
                forget_samples += len(fid)
                model.eval()
                z = model(common.gpu_augment(xf[fid], True) if a.aug_forget else xf[fid].float().div(255.))
                if a.loss != "kl_iproj":
                    s, _ = scores(z, yf[fid])
                    lf = score_loss(s, targets[fid])
                else:
                    lf = F.kl_div(F.log_softmax(z, 1), logq[fid], reduction="batchmean", log_target=True)
                (a.alpha * lf).backward()
                if a.clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            model.train(a.bn == "train")
            lr_loss = F.cross_entropy(model(common.gpu_augment(xr[rid], True)), yr[rid])
            lr_loss.backward()
            if a.decay_epochs:
                total = len(r_batches) * a.decay_epochs
                done = (epoch - (a.epochs - a.decay_epochs) - 1) * len(r_batches) + step
                for g in opt.param_groups:
                    g["lr"] = a.lr * (1. - max(0, done) / total) if epoch > a.epochs - a.decay_epochs else a.lr
            opt.step()
            if avg is not None and epoch >= a.ema_start_epoch:
                avg_n += 1
                with torch.no_grad():
                    for pa, pm in zip(avg, model.parameters()):
                        pa.add_(pm.detach() - pa, alpha=1. / avg_n)
        torch.cuda.synchronize(); train_s += time.monotonic() - t0
        record(epoch, train_s)
    (a.out / "summary.json").write_text(json.dumps(dict(config=vars(a), best_dev=best, last=hist[-1]), default=str, indent=1))


if __name__ == "__main__":
    main()
