"""CPU-only probes of cached Original/held-out donor geometry, without models.

Reads exactly four named arrays from each fixed_targets.npz: Original logits
and labels on forget and target-validation points. Does not import runners,
load checkpoints/results, or access selection, audit, test, or Retrain data.
Temperature probes change ranking computations only; donor target values stay
at their unscaled Original values, isolating assignment sensitivity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ALLOWED_KEYS = ("forget_original_logits", "forget_labels",
                "validation_original_logits", "validation_labels")


def scores(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    other = logits.copy()
    other[np.arange(len(logits)), labels] = -np.inf
    maximum = other.max(1)
    lse = maximum + np.log(np.exp(other - maximum[:, None]).sum(1))
    true = logits[np.arange(len(logits)), labels]
    return true - lse, true - maximum


def logsoftmax(logits):
    centered = logits - logits.max(1, keepdims=True)
    return centered - np.log(np.exp(centered).sum(1, keepdims=True))


def assign_donors(source_scores, source_labels, donor_scores, donor_labels):
    assignment = np.empty(len(source_labels), dtype=np.int64)
    for cls in np.unique(source_labels):
        fi, vi = np.flatnonzero(source_labels == cls), np.flatnonzero(donor_labels == cls)
        if not len(vi):
            raise ValueError(f"missing held-out donor class {cls}")
        fi = fi[np.argsort(source_scores[fi], kind="stable")]
        vi = vi[np.argsort(donor_scores[vi], kind="stable")]
        assignment[fi] = vi[((2 * np.arange(len(fi)) + 1) * len(vi)) // (2 * len(fi))]
    return assignment


def barycentric_targets(source, labels, reference, reference_labels):
    result = np.empty(len(source), dtype=np.float64)
    for cls in np.unique(labels):
        fi = np.flatnonzero(labels == cls)
        fi = fi[np.argsort(source[fi], kind="stable")]
        donor = np.sort(reference[reference_labels == cls])
        n, m = len(fi), len(donor)
        if m == 0:
            raise ValueError(f"missing reference class {cls}")
        left = np.maximum(np.arange(n)[:, None] / n, np.arange(m)[None, :] / m)
        right = np.minimum((np.arange(n)[:, None] + 1) / n, (np.arange(m)[None, :] + 1) / m)
        result[fi] = (n * np.maximum(right - left, 0)) @ donor
    return result


def rank_vector(values):
    result = np.empty(len(values), dtype=np.float64)
    result[np.argsort(values, kind="stable")] = np.arange(len(values))
    return result


def compatible_donor_logits(source_logits, labels, validation_logits, donor_ids):
    result = np.empty_like(source_logits)
    for i, label in enumerate(labels):
        other = np.flatnonzero(np.arange(source_logits.shape[1]) != label)
        ordered = other[np.argsort(-source_logits[i, other], kind="stable")]
        result[i, ordered] = np.sort(validation_logits[donor_ids[i], other])[::-1]
        result[i, label] = validation_logits[donor_ids[i], label]
    return result - result.mean(1, keepdims=True)


def run_cache(path):
    if path.name != "fixed_targets.npz":
        raise ValueError("accept only fixed_targets.npz Original/target caches")
    with np.load(path, allow_pickle=False) as archive:
        # Explicit named reads: no iteration over potentially unrelated arrays.
        z, yf, v, yv = [archive[key] for key in ALLOWED_KEYS]
    z, v = z.astype(np.float64), v.astype(np.float64)
    yf, yv = yf.astype(np.int64), yv.astype(np.int64)
    sf, mf = scores(z, yf)
    sv, mv = scores(v, yv)
    donor_ids = assign_donors(sf, yf, sv, yv)
    donor_error = mv[donor_ids] < 0
    fixed_s = barycentric_targets(sf, yf, sv, yv)
    # I-projection margin = target log-odds - log(source non-true r_max).
    iprojection_margin = fixed_s + mf - sf
    correlations = [float(np.corrcoef(rank_vector(sf[yf == cls]),
                                      rank_vector(mf[yf == cls]))[0, 1])
                    for cls in np.unique(yf)]
    geometry = dict(n_forget=len(z), n_target=len(v),
        forget_error_pct=float(100 * np.mean(mf < 0)),
        target_error_pct=float(100 * np.mean(mv < 0)),
        iprojection_target_error_pct=float(100 * np.mean(iprojection_margin < 0)),
        same_donor_target_error_pct=float(100 * np.mean(donor_error)),
        iprojection_vs_donor_error_identity_disagreement_pct=float(100 * np.mean(
            donor_error != (iprojection_margin < 0))),
        other_conditional_top_prob_median_forget=float(np.median(np.exp(sf - mf))),
        other_conditional_top_prob_median_target=float(np.median(np.exp(sv - mv))),
        logodds_margin_classwise_rank_correlation_mean=float(np.mean(correlations)))

    sensitivity = []
    for scale in (.5, .8, 1.25, 2.):
        source_rank, _ = scores(scale * z, yf)
        target_rank, _ = scores(scale * v, yv)
        changed = assign_donors(source_rank, yf, target_rank, yv)
        sensitivity.append(dict(scale=scale,
            donor_id_changed_pct=float(100 * np.mean(changed != donor_ids)),
            negative_target_identity_changed_pct=float(100 * np.mean((mv[changed] < 0) != donor_error)),
            target_logodds_mae=float(np.mean(abs(sv[changed] - sv[donor_ids]))),
            target_margin_mae=float(np.mean(abs(mv[changed] - mv[donor_ids])))))

    q_logits = compatible_donor_logits(z, yf, v, donor_ids)
    p, q = np.exp(logsoftmax(z)), np.exp(logsoftmax(q_logits))
    energy = {}
    for name, squared_gradient in (
            ("KL", ((p - q) ** 2).sum(1)),
            ("centered_logits_MSE", ((z - z.mean(1, keepdims=True) - q_logits) ** 2).sum(1))):
        energy[name] = dict(negative_donor_fraction=float(donor_error.mean()),
            negative_donor_output_gradient_energy_share=float(squared_gradient[donor_error].sum() / squared_gradient.sum()),
            top10pct_output_gradient_energy_share=float(np.sort(squared_gradient)[-int(np.ceil(.1 * len(z))):].sum() / squared_gradient.sum()))

    pseudo = []
    for name, statistic in (("logodds", sv), ("margin", mv)):
        rows = []
        for repeat in range(10):
            rng = np.random.default_rng(18000 + repeat)
            disagreements, count_floor, planned_n, intersection_n, union_n = 0, 0, 0, 0, 0
            for cls in np.unique(yv):
                indices = np.flatnonzero(yv == cls)
                rng.shuffle(indices)
                source, donors = indices[:len(indices) // 2], indices[len(indices) // 2:]
                assignment = assign_donors(statistic[source], yv[source], statistic[donors], yv[donors])
                actual, planned = mv[source] < 0, mv[donors[assignment]] < 0
                disagreements += int(np.sum(actual != planned))
                count_floor += int(abs(np.sum(actual) - np.sum(planned)))
                planned_n += int(planned.sum())
                intersection_n += int(np.sum(actual & planned))
                union_n += int(np.sum(actual | planned))
            source_n = sum(int(np.sum(yv == cls)) // 2 for cls in np.unique(yv))
            rows.append([disagreements, count_floor, disagreements - count_floor,
                         100 * disagreements / source_n, 100 * intersection_n / max(1, planned_n),
                         100 * intersection_n / max(1, union_n)])
        pseudo.append(dict(rank_by=name, repetitions=10, source_n=source_n,
            **dict(zip(("mean_error_disagreement_count", "mean_count_difference_floor",
                        "mean_excess_error_disagreement_count", "mean_error_disagreement_pct",
                        "mean_planned_error_precision_pct", "mean_error_set_jaccard_pct"),
                       np.mean(rows, axis=0).tolist()))))
    return dict(cache=str(path), cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        arrays_read=list(ALLOWED_KEYS), geometry=geometry, rank_sensitivity=sensitivity,
        output_gradient_energy=energy, target_only_pseudounlearning=pseudo,
        caveats=["Output-gradient energy is not parameter-gradient energy or optimizer displacement.",
                 "Margin-rank error-tail alignment is largely guaranteed by a monotone threshold.",
                 "Pseudo-unlearning does not independently identify member-to-nonmember counterfactual rank reliability."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = dict(schema=1, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  device="CPU; no checkpoints loaded", probes=[run_cache(path) for path in args.cache])
    serialized = json.dumps(result, indent=2)
    if args.output_json is not None:
        args.output_json.write_text(serialized + "\n")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
