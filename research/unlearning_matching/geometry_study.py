"""Independent PCGrad mechanism ablation around the frozen calibration harness.

All calibration_study.py arguments are forwarded unchanged. --geometry none
is the matched control: it uses the same separate-gradient collection and
statistics as symmetric_pcgrad. Only paired steps can be projected; the
other retain steps are unchanged. Forget gradients already include the
harness's loss weight and norm clipping. Network, data, targets, diagnostics,
checkpoint selection, SGD momentum=.9 and weight_decay=5e-4 are inherited.

The two-gradient first-order statement concerns the combined raw gradient,
not the actual momentum/weight-decay step or finite-step/test performance.
PCGrad is an existing method (Yu et al., NeurIPS 2020), not a novelty claim.
Run --self-test for CPU-only synthetic checks without loading the harness.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "2")
import torch

HERE = Path(__file__).resolve().parent
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
GEOMETRIES = ("none", "symmetric_pcgrad")
LIMITATION = ("First-order noninterference applies only to the combined raw paired-step "
              "gradient at the current minibatches. Inherited SGD momentum=.9 and "
              "weight_decay=5e-4, finite steps, and unpaired retain steps can violate it.")


@torch.no_grad()
def combine_gradients(forget, retain, geometry):
    """Sum or globally project aligned parameter-gradient lists; None is zero.

    Projection coefficients use the ORIGINAL two gradients, never an already
    projected gradient. The forget vector is already weighted and clipped.
    No flattening, parameter-wise projection, renormalization, or extra
    gradient clipping is performed. An entirely empty/frozen list is valid.
    """
    if geometry not in GEOMETRIES or len(forget) != len(retain):
        raise ValueError("invalid geometry or mismatched gradient lists")
    first = next((g for pair in zip(forget, retain) for g in pair if g is not None), None)
    scalars = torch.zeros(3, dtype=torch.float64, device=first.device if first is not None else "cpu")
    for q, r in zip(forget, retain):
        if q is not None:
            scalars[1] += q.square().sum(dtype=torch.float64)
        if r is not None:
            scalars[2] += r.square().sum(dtype=torch.float64)
        if q is not None and r is not None:
            if q.shape != r.shape or q.device != r.device or q.dtype != r.dtype:
                raise ValueError("paired gradients must have identical shape, device and dtype")
            scalars[0] += (q * r).sum(dtype=torch.float64)
    dot, q2, r2 = scalars.tolist()
    if not all(math.isfinite(value) for value in (dot, q2, r2)):
        raise ValueError("nonfinite gradient geometry")
    nonzero = q2 > 0. and r2 > 0.
    conflict = nonzero and dot < 0.
    cosine = max(-1., min(1., dot / math.sqrt(q2 * r2))) if nonzero else None
    # q' + r' = (1-dot/||q||^2)q + (1-dot/||r||^2)r.
    q_scale = 1. - dot / q2 if geometry == "symmetric_pcgrad" and conflict else 1.
    r_scale = 1. - dot / r2 if geometry == "symmetric_pcgrad" and conflict else 1.
    combined = []
    for q, r in zip(forget, retain):
        if q is None and r is None:
            combined.append(None)
        elif q is None:
            combined.append(r.mul(r_scale))
        elif r is None:
            combined.append(q.mul(q_scale))
        else:
            combined.append(q.mul(q_scale).add_(r, alpha=r_scale))
    return combined, dict(dot=dot, forget_norm_sq=q2, retain_norm_sq=r2,
        cosine=cosine, conflict=conflict, projected=geometry == "symmetric_pcgrad" and conflict,
        raw_direction_dot_forget=q_scale * q2 + r_scale * dot,
        raw_direction_dot_retain=q_scale * dot + r_scale * r2)


class GeometryStatistics:
    def __init__(self):
        self.retain_steps = self.paired_steps = self.nonzero_pairs = self.conflicts = self.projections = 0
        self.cosine_sum, self.cosine_min, self.cosine_max = 0., None, None

    def record(self, paired, statistics=None):
        self.retain_steps += 1
        self.paired_steps += int(paired)
        if not paired:
            return
        self.conflicts += int(statistics["conflict"])
        self.projections += int(statistics["projected"])
        cosine = statistics["cosine"]
        if cosine is not None:
            self.nonzero_pairs += 1
            self.cosine_sum += cosine
            self.cosine_min = cosine if self.cosine_min is None else min(self.cosine_min, cosine)
            self.cosine_max = cosine if self.cosine_max is None else max(self.cosine_max, cosine)

    def snapshot(self):
        return dict(retain_steps=self.retain_steps, paired_steps=self.paired_steps,
            nonzero_pairs=self.nonzero_pairs, conflicts=self.conflicts, projections=self.projections,
            conflict_fraction_paired=self.conflicts / self.paired_steps if self.paired_steps else None,
            conflict_fraction_nonzero=self.conflicts / self.nonzero_pairs if self.nonzero_pairs else None,
            cosine_mean=self.cosine_sum / self.nonzero_pairs if self.nonzero_pairs else None,
            cosine_min=self.cosine_min, cosine_max=self.cosine_max,
            cosine_definition="global dot of weighted/clipped forget and retain gradient vectors")


class GradientHook:
    def __init__(self, geometry, retain_gradient):
        self.geometry, self.original_retain_gradient = geometry, retain_gradient
        self.reset()

    def reset(self):
        self.total, self.epoch = GeometryStatistics(), GeometryStatistics()

    def __call__(self, model, teacher, xx, yy, ids, anchor_weight, bn_mode):
        parameters = tuple(p for p in model.parameters() if p.requires_grad)
        forget = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
        paired = any(g is not None for g in forget)
        model.zero_grad(set_to_none=True)
        loss = self.original_retain_gradient(model, teacher, xx, yy, ids, anchor_weight, bn_mode)
        if paired:
            retain = [None if p.grad is None else p.grad.detach() for p in parameters]
            combined, statistics = combine_gradients(forget, retain, self.geometry)
            for parameter, gradient in zip(parameters, combined):
                parameter.grad = gradient
        else:
            statistics = None
        self.total.record(paired, statistics)
        self.epoch.record(paired, statistics)
        return loss


def load_harness():
    spec = importlib.util.spec_from_file_location("geometry_calibration_harness", HERE / "calibration_study.py")
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    return study


def install_hooks(study, geometry):
    """Patch only this imported module instance; never edit the frozen sources."""
    original_preregister, original_schedule = study.preregister, study.study_schedule
    original_save_last, original_dump = study.save_last, study.base.dump
    original_forget, original_retain = study.forget_gradient, study.retain_gradient
    harness_sha256 = hashlib.sha256(Path(study.__file__).read_bytes()).hexdigest()

    # Ordinary runs call the exact inherited loss functions. An entirely
    # frozen model has no backward graph: compute the same losses without
    # backward, allowing None gradients throughout instead of an exception.
    def frozen_safe_retain(model, teacher, xx, yy, ids, anchor_weight, bn_mode):
        if any(p.requires_grad for p in model.parameters()):
            return original_retain(model, teacher, xx, yy, ids, anchor_weight, bn_mode)
        with torch.no_grad():
            model.train(bn_mode != "frozen")
            xb = study.common.gpu_augment(xx[ids], True)
            z = model(xb)
            loss = torch.nn.functional.cross_entropy(z, yy[ids])
            if anchor_weight:
                q = torch.nn.functional.softmax(teacher(xb) / 2., dim=1)
                loss += anchor_weight * 4. * torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(z / 2., dim=1), q, reduction="batchmean")
            return loss.detach()

    def frozen_safe_forget(model, optimizer, xx, yy, ids, targets, weight, loss_kind):
        if any(p.requires_grad for p in model.parameters()):
            return original_forget(model, optimizer, xx, yy, ids, targets, weight, loss_kind)
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            return study.target_loss(model(xx[ids].float() / 255.), yy[ids], ids, targets, loss_kind).detach()

    hook = GradientHook(geometry, frozen_safe_retain)

    def provenance():
        return dict(geometry=geometry, source_sha256=SOURCE_SHA256,
            calibration_harness_sha256=harness_sha256,
            optimizer="torch.optim.SGD", momentum=.9, weight_decay=5e-4,
            geometry_first_order_limitation=LIMITATION,
            projection_scope="global trainable-parameter vector; paired steps only",
            pcgrad_reference="Yu et al., Gradient Surgery for Multi-Task Learning, NeurIPS 2020, Algorithm 1")

    def preregister(a):
        # Reuse the frozen harness's exact manifest construction, with an
        # isolated scratch destination. Then preregister the wrapper as an
        # independent method BEFORE any model/data loading in study.main().
        with tempfile.TemporaryDirectory(prefix="unlearning-geometry-prereg-") as directory:
            shadow = copy.copy(a)
            shadow.run_dir = Path(directory)
            original_preregister(shadow)
            config = json.loads((shadow.run_dir / "preregister.json").read_text())["configuration"]
        config.update(provenance())
        config.update(geometry_wrapper_schema=1,
            matched_control="--geometry none uses identical separate-gradient/statistics path",
            gradient_combination="project both original gradients if dot<0, then sum; no renormalization",
            network="inherited all parameters, FP32; frozen/unused parameters excluded from gradient vector")
        path = a.run_dir / "preregister.json"
        if path.exists():
            if json.loads(path.read_text()).get("configuration") != config:
                raise ValueError("geometry preregistration mismatch; use a separate out-dir")
        else:
            original_dump(path, dict(created_utc=datetime.now(timezone.utc).isoformat(), configuration=config))

    def save_last(destination, model, epoch, metadata):
        # Mutate the caller's metadata too, so logs and final summary record
        # the wrapper, actual optimizer and the same counters as last.json.
        metadata.update(provenance(), geometry_statistics=hook.total.snapshot(),
                        geometry_epoch_statistics=hook.epoch.snapshot())
        original_save_last(destination, model, epoch, metadata)
        hook.epoch = GeometryStatistics()

    def dump(path, value):
        if Path(path).name == "best.json":
            # This object is also used in summary['selection']; update both.
            value.update(provenance(), geometry_statistics=hook.total.snapshot())
        return original_dump(path, value)

    def schedule(*args, **kwargs):
        hook.reset()
        return original_schedule(*args, **kwargs)

    study.preregister, study.study_schedule = preregister, schedule
    study.retain_gradient, study.forget_gradient = hook, frozen_safe_forget
    study.save_last, study.base.dump = save_last, dump
    return hook


def self_test():
    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu").manual_seed(20261108)
    for geometry in GEOMETRIES:
        for _ in range(40):
            q, r = (torch.randn(29, generator=generator, dtype=torch.float64) for _ in range(2))
            # Mixed absent/unused parameter gradients participate as zeros.
            qparts, rparts = [q[:8], None, q[13:], None], [r[:8], r[8:13], r[13:], None]
            expected_q = torch.cat((q[:8], torch.zeros(5), q[13:])).double()
            out, statistics = combine_gradients(qparts, rparts, geometry)
            direction = torch.cat(out[:3])
            if geometry == "none" or expected_q.dot(r) >= 0.:
                assert torch.allclose(direction, expected_q + r, atol=1e-12)
            else:
                dot = expected_q.dot(r)
                expected = expected_q - dot / r.square().sum() * r + r - dot / expected_q.square().sum() * expected_q
                assert torch.allclose(direction, expected, atol=1e-12)
            if geometry == "symmetric_pcgrad":
                assert expected_q.dot(direction) >= -1e-11
                assert r.dot(direction) >= -1e-11
            assert out[-1] is None
            assert abs(statistics["raw_direction_dot_forget"] - float(expected_q.dot(direction))) < 1e-10
        zero, positive = torch.zeros(2), torch.tensor([1., 2.])
        for q, r in ((zero, positive), (positive, zero), (positive, positive), (positive, -positive)):
            out, statistics = combine_gradients([q], [r], geometry)
            expected = torch.zeros(2) if geometry == "symmetric_pcgrad" and q.dot(r) < 0 else q + r
            assert torch.equal(out[0], expected)
        assert combine_gradients([], [], geometry)[0] == []
        assert combine_gradients([None], [None], geometry)[0] == [None]
        assert combine_gradients([None], [positive], geometry)[0][0].equal(positive)

        model = torch.nn.Linear(2, 2, bias=True).double()
        model.bias.requires_grad_(False)
        frozen_bias = model.bias.detach().clone()
        params = [p for p in model.parameters() if p.requires_grad]
        q = torch.tensor([[1., 0.], [0., -2.]], dtype=torch.float64)
        r = torch.tensor([[-1., 1.], [0., 1.]], dtype=torch.float64)

        def fake_retain(model, *_):
            (model.weight * r).sum().backward()
            return torch.tensor(3.)

        model.weight.grad = q.clone()
        hook = GradientHook(geometry, fake_retain)
        hook(model, None, None, None, None, None, None)
        expected, _ = combine_gradients([q], [r], geometry)
        assert torch.allclose(params[0].grad, expected[0])
        assert model.bias.grad is None and torch.equal(model.bias, frozen_bias)
        model.zero_grad(set_to_none=True)
        hook(model, None, None, None, None, None, None)
        assert torch.equal(model.weight.grad, r)
        snapshot = hook.total.snapshot()
        assert snapshot["paired_steps"] == 1 and snapshot["retain_steps"] == 2

    # Exercise independent preregistration and all-frozen wrapper paths on a
    # CPU fake harness; no CIFAR data, CUDA, or calibration import is needed.
    with tempfile.TemporaryDirectory(prefix="unlearning-geometry-test-") as directory:
        def dump(path, value):
            Path(path).write_text(json.dumps(value))
        def register(a):
            dump(a.run_dir / "preregister.json", {"configuration": {"source_sha256": "inherited", "lr": .003}})
        def no_trainable(*_):
            raise AssertionError("frozen model must not enter inherited backward")
        fake = SimpleNamespace(__file__=__file__, preregister=register, study_schedule=lambda *args: None,
            save_last=lambda *args: None, base=SimpleNamespace(dump=dump),
            common=SimpleNamespace(gpu_augment=lambda x, train: x.float()),
            forget_gradient=no_trainable, retain_gradient=no_trainable,
            target_loss=lambda z, y, ids, targets, kind: z.square().mean())
        hook = install_hooks(fake, "none")
        args = SimpleNamespace(run_dir=Path(directory))
        fake.preregister(args)
        fake.preregister(args)
        config = json.loads((args.run_dir / "preregister.json").read_text())["configuration"]
        assert config["geometry"] == "none" and config["momentum"] == .9 and config["source_sha256"] == SOURCE_SHA256
        assert config["calibration_harness_sha256"] == SOURCE_SHA256
        changed = copy.deepcopy(config)
        changed["geometry"] = "symmetric_pcgrad"
        dump(args.run_dir / "preregister.json", {"configuration": changed})
        try:
            fake.preregister(args)
        except ValueError:
            pass
        else:
            raise AssertionError("method drift was accepted")
        model = torch.nn.Linear(2, 2).requires_grad_(False)
        optimizer = torch.optim.SGD(model.parameters(), lr=.003, momentum=.9, weight_decay=5e-4)
        xx, yy, ids = torch.ones(3, 2, dtype=torch.float64), torch.tensor([0, 1, 0]), torch.arange(3)
        fake.forget_gradient(model, optimizer, xx, yy, ids, {}, .1, "unused")
        fake.retain_gradient(model, model, xx, yy, ids, .1, "frozen")
        optimizer.step()
        assert all(p.grad is None for p in model.parameters())
        assert hook.total.snapshot()["paired_steps"] == 0
    print("geometry CPU checks passed: projection, directional dots, zero/nonconflict, "
          "unused/frozen parameters, hook accumulation, manifest identity/drift")


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--geometry", choices=GEOMETRIES, default="symmetric_pcgrad")
    parser.add_argument("--self-test", action="store_true")
    options, forwarded = parser.parse_known_args(argv)
    if options.self_test:
        if forwarded:
            parser.error("--self-test does not accept calibration harness arguments")
        self_test()
        return
    study = load_harness()
    install_hooks(study, options.geometry)
    sys.argv = [sys.argv[0], *forwarded]
    study.main()


if __name__ == "__main__":
    main()
