"""Fixed held-out donor targets; construction uses no model updates or test data.

Within each class, rank Original forget and held-out examples by log-odds
(default) or signed margin. Assign each source the held-out donor at its
mid-quantile, without averaging donor predictions. Keep the donor's true-class
logit and place its sorted other logits in the Original's other-class order.
This preserves the donor's confidence, margin, entropy, and sorted prediction
vector, while retaining the Original's ordering of competing class identities.

For a fixed donor probability multiset, this permutation minimizes KL(q || p0)
over permutations of non-true classes, by the rearrangement inequality. This
claim concerns output-space permutation only; the classwise donor assignment
is a heuristic and neither minimum parameter change nor deletion is proved.
All tensors are detached and processed on CPU. Ties retain input-row/class
order. Repeated donors are expected when a class has more forget than held-out
points; mid-quantiles do not exactly preserve every held-out statistic.
"""
from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path

import torch


def _scores(logits: torch.Tensor, labels: torch.Tensor):
    true = logits.gather(1, labels[:, None]).squeeze(1)
    other = logits.masked_fill(
        torch.nn.functional.one_hot(labels, logits.shape[1]).bool(), -torch.inf)
    return true - torch.logsumexp(other, dim=1), true - other.max(dim=1).values


def _inputs(logits, labels, name):
    z = torch.as_tensor(logits).detach().cpu()
    y = torch.as_tensor(labels).detach().cpu()
    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] < 2:
        raise ValueError(f"{name} logits must be nonempty N x K, with K >= 2")
    if not z.is_floating_point() or not bool(torch.isfinite(z).all()):
        raise ValueError(f"{name} logits must be finite floating-point values")
    if y.ndim != 1 or len(y) != len(z) or y.is_floating_point() or y.is_complex():
        raise ValueError(f"{name} labels must be a length-N integer vector")
    y = y.long()
    if bool(((y < 0) | (y >= z.shape[1])).any()):
        raise ValueError(f"{name} labels are outside the class range")
    return z.double(), y


@torch.no_grad()
def build_rank_donor_targets(forget_logits, forget_labels, validation_logits,
                             validation_labels, *, rank_by="logodds"):
    """Return CPU tensors in forget-row order; donor_ids index validation rows.

    Outputs are target_centered_logits [N,K], target_logodds [N],
    target_margin [N], target_log_probs [N,K], and donor_ids [N]. Floating
    outputs use float64 for float64 forget input, otherwise float32. No random
    generator, calibration predictions, Retrain model, or test set is used.
    """
    if rank_by not in ("logodds", "margin"):
        raise ValueError("rank_by must be 'logodds' or 'margin'")
    output_dtype = (torch.float64 if torch.as_tensor(forget_logits).dtype == torch.float64
                    else torch.float32)
    zf, yf = _inputs(forget_logits, forget_labels, "forget")
    zv, yv = _inputs(validation_logits, validation_labels, "validation")
    if zf.shape[1] != zv.shape[1]:
        raise ValueError("forget and validation class counts differ")
    which = 0 if rank_by == "logodds" else 1
    sf, sv = _scores(zf, yf)[which], _scores(zv, yv)[which]
    donors = torch.empty(len(yf), dtype=torch.long)
    for c in yf.unique(sorted=True).tolist():
        fi, vi = torch.where(yf == c)[0], torch.where(yv == c)[0]
        if len(vi) == 0:
            raise ValueError(f"validation has no donor for forget class {c}")
        fi = fi[torch.argsort(sf[fi], stable=True)]
        vi = vi[torch.argsort(sv[vi], stable=True)]
        # floor((i + 1/2) * m/n), with exact integer arithmetic.
        positions = ((2 * torch.arange(len(fi)) + 1) * len(vi)) // (2 * len(fi))
        donors[fi] = vi[positions]

    donor_logits = zv[donors]
    n, k = zf.shape
    other_ids = torch.arange(k).expand(n, k)[
        ~torch.nn.functional.one_hot(yf, k).bool()].reshape(n, k - 1)
    source_order = torch.argsort(zf.gather(1, other_ids), dim=1,
                                descending=True, stable=True)
    destination_ids = other_ids.gather(1, source_order)
    donor_other = donor_logits.gather(1, other_ids)
    donor_values = donor_other.sort(dim=1, descending=True, stable=True).values
    target = torch.empty_like(zf)
    target.scatter_(1, destination_ids, donor_values)
    target.scatter_(1, yf[:, None], donor_logits.gather(1, yf[:, None]))
    target -= target.mean(dim=1, keepdim=True)
    logodds, margin = _scores(target, yf)
    return {
        "target_centered_logits": target.to(output_dtype),
        "target_logodds": logodds.to(output_dtype),
        "target_margin": margin.to(output_dtype),
        "target_log_probs": torch.log_softmax(target, dim=1).to(output_dtype),
        "donor_ids": donors,
    }


def self_test():
    """CPU checks of target invariants, rank/tie rules, and permutation optimum."""
    torch.set_num_threads(1)
    gen = torch.Generator(device="cpu").manual_seed(20261107)
    for n, m in ((13, 8), (8, 13), (5, 5), (1, 4), (4, 1)):
        zf, zv = torch.randn(n, 4, generator=gen, dtype=torch.float64), torch.randn(
            m, 4, generator=gen, dtype=torch.float64)
        yf, yv = torch.zeros(n, dtype=torch.long), torch.zeros(m, dtype=torch.long)
        for rank_by in ("logodds", "margin"):
            out = build_rank_donor_targets(zf, yf, zv, yv, rank_by=rank_by)
            donor = zv[out["donor_ids"]]
            q = out["target_log_probs"].exp()
            donor_p = donor.softmax(1)
            assert torch.allclose(q.sum(1), torch.ones(n, dtype=q.dtype))
            assert torch.allclose(q[:, 0], donor_p[:, 0])
            assert torch.allclose(q[:, 1:].sort(1).values, donor_p[:, 1:].sort(1).values)
            ds, dm = _scores(donor, yf)
            assert torch.allclose(out["target_logodds"], ds)
            assert torch.allclose(out["target_margin"], dm)
            assert torch.allclose(-(q * q.log()).sum(1), -(donor_p * donor_p.log()).sum(1))
            assert torch.allclose(out["target_centered_logits"].mean(1), torch.zeros(n, dtype=q.dtype), atol=1e-14)
            # Exhaust all 3! other-class assignments for this fixed donor.
            logp = zf.log_softmax(1)
            chosen = (q * (q.log() - logp)).sum(1)
            for i in range(n):
                candidates = []
                for perm in permutations(donor_p[i, 1:].tolist()):
                    permq = torch.tensor([donor_p[i, 0].item(), *perm], dtype=q.dtype)
                    candidates.append((permq * (permq.log() - logp[i])).sum())
                assert torch.allclose(chosen[i], torch.stack(candidates).min(), atol=1e-12)
    # Stable score ties: mid-quantile indices select original donor row order.
    zf, zv = torch.zeros(6, 3), torch.zeros(4, 3)
    out = build_rank_donor_targets(zf, torch.zeros(6, dtype=torch.long),
                                   zv, torch.zeros(4, dtype=torch.long))
    assert out["donor_ids"].tolist() == [0, 1, 1, 2, 3, 3]
    # Multiple labels must never receive donors from a different class.
    yf, yv = torch.tensor([2, 0, 2, 0, 1]), torch.tensor([1, 2, 0, 1])
    out = build_rank_donor_targets(torch.zeros(5, 3), yf, torch.zeros(4, 3), yv)
    assert torch.equal(yv[out["donor_ids"]], yf)
    again = build_rank_donor_targets(torch.zeros(5, 3), yf, torch.zeros(4, 3), yv)
    assert all(torch.equal(out[key], again[key]) for key in out)
    try:
        build_rank_donor_targets(torch.zeros(1, 3), torch.tensor([2]),
                                  torch.zeros(1, 3), torch.tensor([1]))
    except ValueError:
        pass
    else:
        raise AssertionError("missing donor class was accepted")
    print("rank_donor CPU checks passed: donor statistics, KL optimum, unequal sizes, stable ties")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-by", choices=("logodds", "margin"), default="logodds")
    parser.add_argument("--input-npz", type=Path,
                        help="NPZ with forget_logits/labels and validation_logits/labels")
    parser.add_argument("--output-npz", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.input_npz is not None:
        if args.output_npz is None:
            parser.error("--input-npz requires --output-npz")
        import numpy as np
        with np.load(args.input_npz, allow_pickle=False) as arrays:
            out = build_rank_donor_targets(arrays["forget_logits"], arrays["forget_labels"],
                                           arrays["validation_logits"], arrays["validation_labels"],
                                           rank_by=args.rank_by)
        np.savez(args.output_npz, **{key: value.numpy() for key, value in out.items()})
    elif not args.self_test:
        parser.error("provide --input-npz or --self-test")


if __name__ == "__main__":
    main()
