"""SalUn-inspired coordinate mask around the unchanged calibration harness.

The mask is computed from the absolute gradient of mean clean/eval CE over
the complete forget set at Original.  It is a separate mechanism control,
not an exact SalUn reproduction: the historical SalUn baseline augments its
mask pass and trains by random labels; this wrapper inherits the selected
fixed-target objective and schedule.  No selection/audit/test/Retrain output
is used to construct the mask.

Candidates mask backward gradients.  After ordinary SGD (including original
weight decay/momentum), only parameters tagged by this wrapper are restored
outside the mask to Original and their masked-out momentum is cleared.  The
allowed coordinates follow exactly ordinary masked-gradient SGD; unrelated
optimizers and the fully frozen teacher are untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "2")
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("saliency_calibration", HERE / "calibration_study.py")
study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(study)
base, common = study.base, study.common
SOURCE_SHA256 = common.sha256_file(__file__)
SCAFFOLD_SHA256 = common.sha256_file(HERE / "calibration_study.py")
_OWNER = object()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_mask(model, images, labels, fraction):
    """Clean/eval gradient of full-dataset mean CE; threshold ties are excluded."""
    model.eval()
    model.zero_grad(set_to_none=True)
    synchronize(images.device)
    start = time.monotonic()
    for i in range(0,len(labels),256):
        logits=model(images[i:i+256].float()/255.)
        (F.cross_entropy(logits,labels[i:i+256],reduction="sum")/len(labels)).backward()
    synchronize(images.device)
    forward_seconds=time.monotonic()-start
    begin_threshold=time.monotonic()
    params={name:p for name,p in model.named_parameters() if p.requires_grad}
    flat=torch.cat([p.grad.detach().abs().flatten() for p in params.values()])
    if not torch.isfinite(flat).all():raise ValueError("non-finite saliency gradient")
    if fraction==1.:
        threshold=None
        masks={name:torch.ones_like(p,dtype=torch.bool) for name,p in params.items()}
    else:
        # At fraction .5 this is a global median-order-statistic threshold.
        rank=max(1,min(len(flat),int(math.floor((1.-fraction)*len(flat)))))
        threshold=float(flat.kthvalue(rank).values)
        masks={name:p.grad.detach().abs()>threshold for name,p in params.items()}
    synchronize(images.device)
    threshold_seconds=time.monotonic()-begin_threshold
    counts={name:int(mask.sum()) for name,mask in masks.items()}
    total=sum(p.numel() for p in params.values())
    return masks,dict(mask_fraction_requested=fraction,actual_allowed_count=sum(counts.values()),
        total_parameter_count=total,actual_allowed_fraction=sum(counts.values())/total,
        allowed_counts_by_parameter=counts,threshold=threshold,
        threshold_rule="strictly greater than global order statistic; threshold ties excluded",
        mask_forward_pass_seconds=forward_seconds,mask_threshold_seconds=threshold_seconds,
        mask_total_seconds=forward_seconds+threshold_seconds,n_forget=len(labels),
        mask_loss="full forget mean cross entropy; clean images; model.eval(); FP32")


def attach_mask(model,masks):
    handles=[]
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        mask=masks[name].to(device=p.device,dtype=torch.bool)
        p._saliency_owner=_OWNER
        p._saliency_mask=mask
        p._saliency_original=p.detach().clone()
        handles.append(p.register_hook(lambda gradient,keep=mask:gradient*keep))
    model._saliency_hook_handles=handles
    return model


def make_guarded_step(original_step):
    def guarded_step(optimizer,closure=None):
        result=original_step(optimizer,closure=closure)
        with torch.no_grad():
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if getattr(p,"_saliency_owner",None) is not _OWNER:continue
                    excluded=~p._saliency_mask
                    p.copy_(torch.where(excluded,p._saliency_original,p))
                    momentum=optimizer.state.get(p,{}).get("momentum_buffer")
                    if momentum is not None:momentum.masked_fill_(excluded,0.)
        return result
    return guarded_step


def install(fraction):
    original_factory,original_dump,original_preregister=base.model_from,base.dump,study.preregister
    original_targets,original_step=study.study_targets,torch.optim.SGD.step
    masks=None
    mask_metadata={}
    extra=dict(study_kind="saliency_coordinate_ablation",network="saliency-masked parameter coordinates, FP32",
        wrapper_source_sha256=SOURCE_SHA256,inherited_scaffold_sha256=SCAFFOLD_SHA256,
        mask_fraction_requested=fraction,mask_provenance="SalUn-inspired, not an exact SalUn reproduction",
        optimizer_semantics="ordinary masked-gradient SGD, then restore excluded coordinates and zero their momentum",
        mask_rule="Original complete-forget mean CE, clean/eval, global absolute-gradient threshold",
        mask_uses_selection=False,teacher_trainable_parameter_count=0,teacher_has_mask_hooks=False)

    def factory(*args,**kwargs):
        model=original_factory(*args,**kwargs)
        return attach_mask(model,masks) if masks is not None else model

    def metadata_dump(path,value):
        if Path(path).name in ("last.json","best.json","summary.json"):
            value=dict(value,**extra,mask_artifact=mask_metadata)
            if isinstance(value.get("selection"),dict):
                value["selection"]=dict(value["selection"],**extra,mask_artifact=mask_metadata)
        return original_dump(path,value)

    def preregister(a):
        captured=[]
        with tempfile.TemporaryDirectory(prefix="unlearning-saliency-config-") as temporary:
            shadow=SimpleNamespace(**{**vars(a),"run_dir":Path(temporary)})
            saved=base.dump
            try:
                base.dump=lambda _path,value:captured.append(value)
                original_preregister(shadow)
            finally:base.dump=saved
        configuration=dict(captured[0]["configuration"],**extra)
        path=a.run_dir/"preregister.json"
        if path.exists():
            if json.loads(path.read_text()).get("configuration")!=configuration:
                raise ValueError("saliency preregistration mismatch; use a separate out-dir")
        else:original_dump(path,dict(created_utc=datetime.now(timezone.utc).isoformat(),configuration=configuration))
        snapshots=a.run_dir/"source_snapshot";snapshots.mkdir(exist_ok=True)
        for name,digest in (("saliency_study.py",SOURCE_SHA256),("calibration_study.py",SCAFFOLD_SHA256)):
            content=(HERE/name).read_bytes()
            if hashlib.sha256(content).hexdigest()!=digest:raise RuntimeError("source changed")
            destination=snapshots/name
            if destination.exists() and destination.read_bytes()!=content:raise ValueError("snapshot mismatch")
            destination.write_bytes(content)

    def study_targets(a,teacher,data):
        nonlocal masks,mask_metadata
        # The teacher was constructed before masks exist and has no hooks.
        if any(p.requires_grad for p in teacher.parameters()) or hasattr(teacher,"_saliency_hook_handles"):
            raise ValueError("teacher must be fully frozen and unmasked")
        mask_path=a.run_dir/"saliency_mask.npz";meta_path=a.run_dir/"saliency_mask.json"
        if mask_path.exists() and meta_path.exists():
            mask_metadata=json.loads(meta_path.read_text())
            if common.sha256_file(mask_path)!=mask_metadata["mask_file_sha256"]:
                raise ValueError("mask artifact hash mismatch")
            masks={}
            with np.load(mask_path,allow_pickle=False) as z:
                for name,shape in mask_metadata["parameter_shapes"].items():
                    size=int(np.prod(shape))
                    array=np.unpackbits(z[name],bitorder="little",count=size).reshape(shape).astype(bool)
                    masks[name]=torch.from_numpy(array).to(data["forget"][0].device)
        else:
            common.set_determinism(a.seed)
            probe=original_factory(base.original_state(a)).eval()
            masks,mask_metadata=build_mask(probe,*data["forget"],fraction)
            mask_metadata["parameter_shapes"]={name:list(mask.shape) for name,mask in masks.items()}
            np.savez_compressed(mask_path,**{name:np.packbits(mask.detach().cpu().numpy().reshape(-1),bitorder="little")
                                             for name,mask in masks.items()})
            mask_metadata.update(mask_file_sha256=common.sha256_file(mask_path),
                wrapper_source_sha256=SOURCE_SHA256,seed=a.seed,split_seed=a.split_seed,
                original_sha256=common.SHA256_ORIGINAL["cifar10"],train_sha256=common.SHA256_TRAIN["cifar10"])
            original_dump(meta_path,mask_metadata)
            del probe
        registration=json.loads((a.run_dir/"preregister.json").read_text())
        registration["mask_artifact"]=mask_metadata
        original_dump(a.run_dir/"preregister.json",registration)
        base.log("saliency_mask_ready",**mask_metadata)
        return original_targets(a,teacher,data)

    base.model_from,base.dump,study.preregister,study.study_targets=factory,metadata_dump,preregister,study_targets
    torch.optim.SGD.step=make_guarded_step(original_step)
    return original_factory,original_dump,original_preregister,original_targets,original_step


def self_test():
    torch.set_num_threads(1);torch.manual_seed(51)
    model=torch.nn.Linear(3,2)
    masks={name:(torch.arange(p.numel()).reshape(p.shape)%2==0) for name,p in model.named_parameters()}
    initial={name:p.detach().clone() for name,p in model.named_parameters()}
    attach_mask(model,masks)
    original_step=torch.optim.SGD.step
    torch.optim.SGD.step=make_guarded_step(original_step)
    try:
        opt=torch.optim.SGD(model.parameters(),lr=.03,momentum=.9,weight_decay=.2)
        reference={name:value.clone() for name,value in initial.items()};momentum={}
        for iteration in range(4):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(torch.randn(9,3)),torch.arange(9)%2).backward()
            for name,p in model.named_parameters():
                mask=masks[name]
                gradient=(p.grad+.2*reference[name])[mask]
                momentum[name]=gradient.clone() if iteration==0 else .9*momentum[name]+gradient
                reference[name][mask]-=.03*momentum[name]
            opt.step()
            for name,p in model.named_parameters():
                assert torch.equal(p[~masks[name]],initial[name][~masks[name]])
                assert torch.allclose(p,reference[name],atol=1e-7)
                assert torch.count_nonzero(opt.state[p]["momentum_buffer"][~masks[name]])==0
        # An unrelated optimizer must retain ordinary SGD decay behavior.
        p=torch.nn.Parameter(torch.ones(2));other=torch.optim.SGD([p],lr=.1,weight_decay=.2)
        p.grad=torch.zeros_like(p);other.step();assert torch.allclose(p,torch.full((2,),.98))
        teacher=torch.nn.Linear(3,2).requires_grad_(False);attach_mask(teacher,masks)
        assert not any(p.requires_grad for p in teacher.parameters()) and not teacher._saliency_hook_handles
        # Clean mean-gradient construction works without touching any eval set.
        tiny=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3*2*2,2))
        xx=torch.randint(0,256,(7,3,2,2),dtype=torch.uint8);yy=torch.arange(7)%2
        built,meta=build_mask(tiny,xx,yy,.5)
        assert meta["n_forget"]==7 and all(mask.dtype==torch.bool for mask in built.values())
    finally:torch.optim.SGD.step=original_step
    print("saliency CPU checks passed: exact excluded coordinates, cleared excluded momentum, allowed-coordinate masked-SGD recurrence, unrelated SGD unchanged, teacher frozen, clean mean-gradient mask")


def main():
    if "--self-test" in sys.argv:
        if len(sys.argv)!=2:raise ValueError("--self-test takes no other arguments")
        self_test();return
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mask-fraction",type=float,default=.5)
    options,remaining=parser.parse_known_args()
    if not 0.<options.mask_fraction<=1.:raise ValueError("mask fraction must be in (0,1]")
    install(options.mask_fraction)
    sys.argv=[sys.argv[0],*remaining]
    study.main()


if __name__=="__main__":main()
