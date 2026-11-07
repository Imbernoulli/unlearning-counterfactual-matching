"""Parameter-scope ablation around the unchanged calibration-only harness.

Use its normal CLI plus --scope all|fc|layer4_fc.  The wrapper sets only
requires_grad flags on candidate factories; teacher construction is followed
by the harness's existing requires_grad_(False), so it stays fully frozen.
The loss, batches, target construction, selection rule, and optimizer remain
unchanged.  Frozen parameters have grad=None, so SGD neither decays them nor
creates momentum state for them.  BatchNorm buffers follow --bn-mode; use
frozen mode to isolate parameter localization from running-statistic changes.

All preregistration/checkpoint metadata records the actual scope and both
source hashes.  Existing unwrapped outputs are rejected by preregistration.
This script has no evaluation of audit or official-test data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "2")
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("localized_calibration", HERE / "calibration_study.py")
study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(study)
base, common = study.base, study.common
SOURCE_SHA256 = common.sha256_file(__file__)
SCAFFOLD_SHA256 = common.sha256_file(HERE / "calibration_study.py")
SCOPE_PREFIXES = {"all": None, "fc": ("fc.",), "layer4_fc": ("layer4.", "fc.")}


def configure_scope(model, scope):
    """Change parameter trainability only; preserve parameter/buffer values."""
    prefixes = SCOPE_PREFIXES[scope]
    for name, parameter in model.named_parameters():
        allowed = prefixes is None or name.startswith(prefixes)
        parameter.requires_grad_(allowed)
        if not allowed:
            parameter.grad = None
    return model


def scope_metadata(scope, layout=None):
    labels = {"all": "all parameters", "fc": "fc parameters only",
              "layer4_fc": "layer4 and fc parameters only"}
    result = dict(scope=scope, network=f"{labels[scope]}, FP32",
        wrapper_source_sha256=SOURCE_SHA256, inherited_scaffold_sha256=SCAFFOLD_SHA256,
        localization="requires_grad controls optimizer updates; buffers follow bn_mode",
        allowed_parameter_prefixes=list(SCOPE_PREFIXES[scope]) if SCOPE_PREFIXES[scope] else ["*"])
    if layout is not None:
        result.update(layout)
    return result


def model_layout(model):
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    return dict(trainable_parameter_names=[name for name,_ in trainable],
                trainable_parameter_count=sum(parameter.numel() for _,parameter in trainable),
                total_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
                teacher_trainable_parameter_count=0)


def install(scope):
    """Install narrow hooks without copying the frozen training/selection loop."""
    factory, dump, preregister = base.model_from, base.dump, study.preregister
    declared_layout = model_layout(configure_scope(common.build_model("cifar10"), scope))
    extra = scope_metadata(scope, declared_layout)

    def localized_factory(*args, **kwargs):
        model = configure_scope(factory(*args, **kwargs), scope)
        if model_layout(model)["trainable_parameter_names"] != declared_layout["trainable_parameter_names"]:
            raise ValueError("factory parameter layout differs from preregistered scope")
        return model

    def metadata_dump(path, value):
        path = Path(path)
        if path.name in ("last.json", "best.json", "summary.json"):
            value = dict(value, **extra)
            if isinstance(value.get("selection"), dict):
                value["selection"] = dict(value["selection"], **extra)
        return dump(path, value)

    def localized_preregister(a):
        # Ask the unmodified harness to produce its exact configuration in an
        # empty scratch directory, capturing its write in memory.  Then replace
        # its all-parameter field before any persistent preregistration write.
        captured = []
        with tempfile.TemporaryDirectory(prefix="unlearning-scope-config-") as temporary:
            shadow = SimpleNamespace(**{**vars(a), "run_dir": Path(temporary)})
            saved_dump = base.dump
            try:
                base.dump = lambda _path, value: captured.append(value)
                preregister(shadow)
            finally:
                base.dump = saved_dump
        if len(captured) != 1:
            raise RuntimeError("unexpected preregistration write protocol")
        configuration = dict(captured[0]["configuration"], **extra)
        configuration["study_kind"] = "parameter_scope_ablation"
        path = a.run_dir / "preregister.json"
        if path.exists():
            if json.loads(path.read_text()).get("configuration") != configuration:
                raise ValueError("localized preregistration mismatch; use a separate out-dir")
        else:
            dump(path, dict(created_utc=datetime.now(timezone.utc).isoformat(), configuration=configuration))
        snapshots = a.run_dir / "source_snapshot"
        snapshots.mkdir(exist_ok=True)
        for name, digest in (("localized_study.py", SOURCE_SHA256), ("calibration_study.py", SCAFFOLD_SHA256)):
            contents = (HERE / name).read_bytes()
            if hashlib.sha256(contents).hexdigest() != digest:
                raise RuntimeError("source changed during localization setup")
            destination = snapshots / name
            if destination.exists() and destination.read_bytes() != contents:
                raise ValueError("existing source snapshot differs")
            destination.write_bytes(contents)

    base.model_from, base.dump, study.preregister = localized_factory, metadata_dump, localized_preregister
    return factory, dump, preregister


def self_test():
    torch.set_num_threads(1)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Linear(3,4)
            self.layer1 = torch.nn.Linear(4,4)
            self.layer4 = torch.nn.Linear(4,4)
            self.fc = torch.nn.Linear(4,2)
        def forward(self, x):
            return self.fc(self.layer4(self.layer1(self.conv1(x)).tanh()).tanh())

    for scope in SCOPE_PREFIXES:
        torch.manual_seed(31)
        model = configure_scope(Tiny(), scope)
        before = {name: parameter.detach().clone() for name,parameter in model.named_parameters()}
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=.2)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(torch.randn(8,3)),torch.arange(8)%2).backward()
            assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
            optimizer.step()
        assert all(torch.equal(parameter,before[name]) for name,parameter in model.named_parameters() if not parameter.requires_grad)
        assert all(parameter not in optimizer.state for parameter in model.parameters() if not parameter.requires_grad)
        teacher = configure_scope(Tiny(), scope).requires_grad_(False)
        assert not any(parameter.requires_grad for parameter in teacher.parameters())
    original_hooks = install("fc")
    try:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = SimpleNamespace(run_dir=root,seed=42,split_seed=42,schedules=["C"],lr=.003,
                forget_weight=.1,epochs=100,anchor_weight=.1,retain_subset=5000,
                target_kind="global_logodds",loss="logodds_mse",bn_mode="frozen")
            study.preregister(a)
            configuration = json.loads((root/"preregister.json").read_text())["configuration"]
            assert configuration["scope"]=="fc" and configuration["network"]=="fc parameters only, FP32"
            assert configuration["trainable_parameter_names"]==["fc.weight","fc.bias"]
            assert configuration["teacher_trainable_parameter_count"]==0
            for name in ("last.json","best.json","summary.json"):
                base.dump(root/name,{"network":"all parameters, FP32","selection":{"epoch":2}})
                value=json.loads((root/name).read_text())
                assert value["network"]=="fc parameters only, FP32" and value["scope"]=="fc"
                assert value["selection"]["scope"]=="fc"
    finally:
        base.model_from,base.dump,study.preregister=original_hooks
    print("localized CPU checks passed: frozen grad=None, unchanged frozen parameters under SGD decay/momentum, teacher fully frozen, accurate preregistration/checkpoint scope metadata")


def main():
    if "--self-test" in sys.argv:
        if len(sys.argv)!=2:raise ValueError("--self-test takes no other arguments")
        self_test();return
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scope",choices=list(SCOPE_PREFIXES),default="all")
    options,remaining=parser.parse_known_args()
    install(options.scope)
    sys.argv=[sys.argv[0],*remaining]
    study.main()


if __name__=="__main__":main()
