"""Read-only, training-data-only mechanism probes for unlearning checkpoints.

Compare Original to one candidate without test, Retrain, audit predictions, or
past result files.  Targets are reconstructed from all forget/target points;
--max-images only limits reported forward/feature diagnostics.  The canonical
calibration pool is stratified into the identical selection/audit split used
by calibration_study.py; audit images never enter a forward pass here.

Outputs: CPU per-example logits/scores/features and changes; class/Original-
confidence-decile summaries; parameter/BN changes; feature cosine and linear
CKA; four feature/head combinations; and clean forget-target vs retain-CE
gradient cosines/norms.  These are descriptive probes, not deletion evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "2")
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("mechanism_calibration", HERE / "calibration_study.py")
cal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cal)
fixed, base, common = cal.fixed, cal.base, cal.common
BLOCKS = ("stem", "layer1", "layer2", "layer3", "layer4", "fc")
SOURCE_SHA256 = common.sha256_file(__file__)


def block_name(name):
    prefix = name.split(".", 1)[0]
    return prefix if prefix in BLOCKS else "stem"


def linear_cka(x, y):
    """Centered linear CKA in feature space; no N x N Gram matrix is built."""
    x, y = x.detach().cpu().double(), y.detach().cpu().double()
    x, y = x - x.mean(0), y - y.mean(0)
    numerator = (x.T @ y).square().sum()
    denominator = (x.T @ x).square().sum().sqrt() * (y.T @ y).square().sum().sqrt()
    return float(numerator / denominator) if denominator > 0 else None


def grad_statistics(first, second):
    a = torch.cat([p.detach().cpu().double().flatten() for p in first])
    b = torch.cat([p.detach().cpu().double().flatten() for p in second])
    na, nb = a.norm(), b.norm()
    return dict(forget_gradient_norm=float(na), retain_gradient_norm=float(nb),
                cosine=float(torch.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else None,
                dot_product=float(torch.dot(a, b)),
                interpretation="negative cosine means retain descent opposes forget-objective descent")


@torch.no_grad()
def feature_logits(model, images, device, batch_size=128):
    """The fc pre-hook is exactly layer4 -> adaptive GAP -> flatten features."""
    features, outputs = [], []
    hook = model.fc.register_forward_pre_hook(lambda _module, args: features.append(args[0].detach().cpu()))
    model.eval()
    try:
        for start in range(0, len(images), batch_size):
            xb = common.to_gpu_u8(images[start:start + batch_size], device).float() / 255.
            outputs.append(model(xb).detach().cpu())
    finally:
        hook.remove()
    return torch.cat(features), torch.cat(outputs)


def model_on(state, device):
    model = common.build_model("cifar10")
    model.load_state_dict(state, strict=True)
    return model.to(device).float().eval()


def scores(logits, labels):
    logodds, margin = base.logodds_margin(logits, labels)
    probability = logits.softmax(1)
    return dict(logodds=logodds, margin=margin,
                confidence=probability.gather(1, labels[:, None]).squeeze(1),
                error=logits.argmax(1).ne(labels), prediction=logits.argmax(1))


def group_summary(original, candidate, labels, targets=None):
    groups = {"all": torch.ones(len(labels), dtype=torch.bool)}
    for cls in range(10):
        groups[f"class_{cls}"] = labels == cls
    order = torch.argsort(original["confidence"], stable=True)
    decile = torch.empty(len(labels), dtype=torch.long)
    decile[order] = torch.arange(len(labels)) * 10 // len(labels)
    for rank in range(10):
        groups[f"original_confidence_decile_{rank}"] = decile == rank
    results = {}
    for name, select in groups.items():
        if not select.any():
            continue
        good_to_bad = (~original["error"] & candidate["error"])[select]
        bad_to_good = (original["error"] & ~candidate["error"])[select]
        row = dict(n=int(select.sum()),
            original_error_pct=float(100 * original["error"][select].float().mean()),
            candidate_error_pct=float(100 * candidate["error"][select].float().mean()),
            correct_to_error_pct=float(100 * good_to_bad.float().mean()),
            error_to_correct_pct=float(100 * bad_to_good.float().mean()),
            changed_prediction_pct=float(100 * (original["prediction"] != candidate["prediction"])[select].float().mean()))
        for key in ("confidence", "logodds", "margin"):
            row[f"original_{key}_mean"] = float(original[key][select].mean())
            row[f"candidate_{key}_mean"] = float(candidate[key][select].mean())
            row[f"delta_{key}_mean"] = float((candidate[key] - original[key])[select].mean())
        if targets is not None:
            for key in ("logodds", "margin"):
                residual = candidate[key][select] - targets[key][select]
                before = original[key][select] - targets[key][select]
                row[f"candidate_target_{key}_residual_mean"] = float(residual.mean())
                row[f"candidate_target_{key}_rmse"] = float(residual.square().mean().sqrt())
                row[f"original_target_{key}_rmse"] = float(before.square().mean().sqrt())
        results[name] = row
    return results


def basic_metrics(logits, labels):
    values = scores(logits, labels)
    return dict(error_pct=float(100 * values["error"].float().mean()),
                confidence_mean=float(values["confidence"].mean()),
                logodds_mean=float(values["logodds"].mean()), margin_mean=float(values["margin"].mean()))


def parameter_changes(original_state, candidate_state, original_model, candidate_model):
    out = {}
    parameter_names = set(dict(original_model.named_parameters()))
    for block in BLOCKS:
        keys = [name for name in parameter_names if block_name(name) == block]
        original = torch.cat([original_state[name].double().flatten() for name in keys])
        candidate = torch.cat([candidate_state[name].double().flatten() for name in keys])
        out[block] = dict(original_norm=float(original.norm()), delta_norm=float((candidate-original).norm()),
                          relative_delta_norm=float((candidate-original).norm() / original.norm()),
                          cosine=float(F.cosine_similarity(original, candidate, dim=0)))
    original_buffers, candidate_buffers = cal.bn_buffers(original_model), cal.bn_buffers(candidate_model)
    bn = {name: dict(max_absolute_delta=float((candidate_buffers[name].double()-value.double()).abs().max()),
                     l2_delta=float((candidate_buffers[name].double()-value.double()).norm()))
          for name, value in original_buffers.items()}
    return dict(parameter_blocks=out, bn_buffers=bn,
                bn_buffer_max_delta=max(v["max_absolute_delta"] for v in bn.values()))


def shared_targets(a, original, x, y, idx):
    """Reconstruct fixed targets, or use an explicitly hash-verified study cache."""
    if a.target_cache is not None:
        if a.target_cache_sha256 is None or common.sha256_file(a.target_cache) != a.target_cache_sha256:
            raise ValueError("target cache needs an exact matching --target-cache-sha256")
        parent = a.target_cache.parent
        manifest = json.loads((parent / "splits.json").read_text())
        expected = json.loads((a.run_dir / "splits.json").read_text())
        configuration = json.loads((parent / "preregister.json").read_text())["configuration"]
        provenance_ok = (configuration.get("target_kind") == a.target_kind
                         and configuration.get("train_sha256") == common.SHA256_TRAIN["cifar10"]
                         and configuration.get("original_sha256") == common.SHA256_ORIGINAL["cifar10"]
                         and configuration.get("seed") == a.seed
                         and configuration.get("split_seed") == a.split_seed)
        if manifest != expected or not provenance_ok:
            raise ValueError("target cache split/preregistration provenance mismatch")
        with np.load(a.target_cache, allow_pickle=False) as z:
            result = {key: torch.from_numpy(z[key]).float() for key in ("logodds", "margin", "centered_logits", "log_probs")}
        if any(len(value) != 4500 or not torch.isfinite(value).all() for value in result.values()):
            raise ValueError("target cache must contain 4500 finite targets")
        if result["log_probs"].shape != (4500,10) or result["centered_logits"].shape != (4500,10):
            raise ValueError("target cache logits have incorrect shapes")
        return result
    _, zf = feature_logits(original, x[idx["forget"]], a.device)
    _, zv = feature_logits(original, x[idx["target"]], a.device)
    yf, yv = torch.from_numpy(y[idx["forget"]]), torch.from_numpy(y[idx["target"]])
    if a.target_kind == "rank_donor":
        rank = base.module_at("mechanism_rank_donor", HERE / "rank_donor.py")
        values = rank.build_rank_donor_targets(zf, yf, zv, yv)
        return {"logodds": values["target_logodds"], "margin": values["target_margin"],
                "centered_logits": values["target_centered_logits"], "log_probs": values["target_log_probs"]}
    tf, _ = base.logodds_margin(zf, yf)
    tv, _ = base.logodds_margin(zv, yv)
    target = torch.empty_like(tf)
    for cls in range(10):
        mask = yf == cls
        target[mask] = fixed.barycentric_quantile_targets(tf[mask], tv[yv == cls])
    logq = fixed.iprojection_log_probs(zf, yf, target)
    _, margin = base.logodds_margin(logq, yf)
    return dict(logodds=target, margin=margin, centered_logits=logq-logq.mean(1,keepdim=True), log_probs=logq)


def gradient_probe(a, model, x, y, idx, targets):
    model.eval()
    params = list(model.named_parameters())
    generator = np.random.RandomState(290112)
    forget_ids = np.sort(generator.choice(len(idx["forget"]), min(256, len(idx["forget"])), replace=False))
    retain_ids = np.sort(generator.choice(len(idx["retain"]), min(256, len(idx["retain"])), replace=False))
    fi, ri = idx["forget"][forget_ids], idx["retain"][retain_ids]
    xf = common.to_gpu_u8(x[fi], a.device).float() / 255.
    xr = common.to_gpu_u8(x[ri], a.device).float() / 255.
    yf, yr = torch.from_numpy(y[fi]).to(a.device), torch.from_numpy(y[ri]).to(a.device)
    moved_targets = {key: value.to(a.device) for key,value in targets.items()}
    ids = torch.from_numpy(forget_ids).to(a.device)
    loss_f = cal.target_loss(model(xf), yf, ids, moved_targets, a.loss)
    gf = torch.autograd.grad(loss_f, [p for _,p in params])
    loss_r = F.cross_entropy(model(xr), yr)
    gr = torch.autograd.grad(loss_r, [p for _,p in params])
    report = dict(forget_target_loss=float(loss_f), retain_ce_loss=float(loss_r),
                  gradient_units="unweighted batch-mean objectives; optimizer coefficients not applied",
                  forget_indices=fi.tolist(), retain_indices=ri.tolist(),
                  global_statistics=grad_statistics(gf,gr), blocks={})
    for block in BLOCKS:
        select=[i for i,(name,_) in enumerate(params) if block_name(name)==block]
        report["blocks"][block]=grad_statistics([gf[i] for i in select],[gr[i] for i in select])
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--base-runs-dir",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--original",default="/srv/home/bohanlyu/data_mlsbench/unlearn-lab/original/cifar10_resnet18_original.pt")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--split-seed",type=int,default=42)
    p.add_argument("--gpu",default=None)
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--max-images",type=int,default=0,help="maximum reported images per group; 0=all; targets remain global")
    p.add_argument("--cka-images",type=int,default=1024)
    p.add_argument("--target-kind",choices=["global_logodds","rank_donor"],default="global_logodds")
    p.add_argument("--loss",choices=["logodds_mse","scores_mse","centered_logits_mse","kl","margin_ascent"],default="logodds_mse")
    p.add_argument("--target-cache",type=Path)
    p.add_argument("--target-cache-sha256")
    a=p.parse_args()
    if a.max_images<0 or a.cka_images<2:raise ValueError("invalid diagnostic limits")
    if a.out_dir.resolve()==a.base_runs_dir.resolve():raise ValueError("probe outputs must be independent")
    if a.gpu is not None:os.environ["CUDA_VISIBLE_DEVICES"]=a.gpu
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    a.run_dir=a.out_dir/f"seed{a.seed}_split{a.split_seed}";a.run_dir.mkdir(parents=True,exist_ok=True)
    a.base_run_dir=a.base_runs_dir/a.run_dir.name
    x,y=base.train_arrays(a);idx=fixed.matching_splits(a,y)
    original_state=base.original_state(a)
    candidate_state=torch.load(a.checkpoint,map_location="cpu",weights_only=True)
    original,candidate=model_on(original_state,a.device),model_on(candidate_state,a.device)
    targets=shared_targets(a,original,x,y,idx)
    rng=np.random.RandomState(20261107);selection=[]
    for cls in range(10):
        pool=idx["calibration"][y[idx["calibration"]]==cls]
        selection.extend(int(v) for v in rng.permutation(pool)[:125])
    retain_pos=np.sort(np.random.RandomState(7312026).choice(len(idx["retain"]),5000,replace=False))
    groups={"forget":idx["forget"],"target":idx["target"],"selection":np.asarray(sorted(selection)),
            "retain_subset":idx["retain"][retain_pos]}
    arrays,summary={},{}
    for name,positions in groups.items():
        if a.max_images and len(positions)>a.max_images:
            keep=np.sort(np.random.RandomState(20261108).choice(len(positions),a.max_images,replace=False))
            positions=positions[keep]
        labels=torch.from_numpy(y[positions]);fo,zo=feature_logits(original,x[positions],a.device)
        fc,zc=feature_logits(candidate,x[positions],a.device)
        so,sc=scores(zo,labels),scores(zc,labels)
        target=None
        if name=="forget":
            rows=torch.from_numpy(np.searchsorted(idx["forget"],positions))
            target={key:value[rows] for key,value in targets.items()}
        report=group_summary(so,sc,labels,target)
        cosine=F.cosine_similarity(fo,fc,dim=1)
        count=min(a.cka_images,len(fo));cka_ids=np.sort(np.random.RandomState(817).choice(len(fo),count,replace=False))
        heads={"original_feature_original_head":zo,"candidate_feature_candidate_head":zc,
               "candidate_feature_original_head":F.linear(fc,original.fc.weight.detach().cpu(),original.fc.bias.detach().cpu()),
               "original_feature_candidate_head":F.linear(fo,candidate.fc.weight.detach().cpu(),candidate.fc.bias.detach().cpu())}
        summary[name]=dict(groups=report,feature_cosine_mean=float(cosine.mean()),
            feature_cosine_min=float(cosine.min()),linear_cka=linear_cka(fo[cka_ids],fc[cka_ids]),
            cka_images=count,head_feature_combinations={key:basic_metrics(value,labels) for key,value in heads.items()})
        arrays.update({f"{name}_indices":positions,f"{name}_labels":labels.numpy(),
                       f"{name}_original_features":fo.numpy(),f"{name}_candidate_features":fc.numpy(),
                       f"{name}_original_logits":zo.numpy(),f"{name}_candidate_logits":zc.numpy(),
                       f"{name}_feature_cosine":cosine.numpy()})
        for key in so:
            arrays[f"{name}_original_{key}"]=so[key].numpy();arrays[f"{name}_candidate_{key}"]=sc[key].numpy()
        for key in ("confidence","logodds","margin"):
            arrays[f"{name}_delta_{key}"]=(sc[key]-so[key]).numpy()
        arrays[f"{name}_correct_to_error"]=(~so["error"]&sc["error"]).numpy()
        arrays[f"{name}_error_to_correct"]=(so["error"]&~sc["error"]).numpy()
        if target is not None:
            for key,value in target.items():arrays[f"{name}_target_{key}"]=value.numpy()
        base.log("mechanism_group",set=name,n=len(labels),feature_cosine=summary[name]["feature_cosine_mean"],
                 cka=summary[name]["linear_cka"],original_error=report["all"]["original_error_pct"],candidate_error=report["all"]["candidate_error_pct"])
    np.savez_compressed(a.run_dir/"per_example.npz",**arrays)
    changes=parameter_changes(original_state,candidate_state,original,candidate)
    gradients={"original":gradient_probe(a,original,x,y,idx,targets),"candidate":gradient_probe(a,candidate,x,y,idx,targets)}
    base.dump(a.run_dir/"mechanism.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=SOURCE_SHA256,checkpoint_sha256=common.sha256_file(a.checkpoint),
        original_sha256=common.SHA256_ORIGINAL["cifar10"],train_sha256=common.SHA256_TRAIN["cifar10"],
        seed=a.seed,split_seed=a.split_seed,target_kind=a.target_kind,loss=a.loss,
        max_images=a.max_images,selection_split_seed=20261107,
        target_cache_sha256=a.target_cache_sha256,audit_predicted=False,test_used=False,retrain_used=False,
        groups=summary,changes=changes,gradients=gradients,
        warning="descriptive mechanistic diagnostics, not evidence of certified deletion"))


if __name__=="__main__":main()
