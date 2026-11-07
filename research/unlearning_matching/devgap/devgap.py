"""Development replica of the benchmark's four metrics on never-trained
validation images, and its gap to a paired Retrain on the same split.

Benchmark (evaluate.evaluate_state): UA = forget error, RA = retain accuracy,
TA = test accuracy, MIA = 1 - SVC(C=3, gamma='auto', rbf) membership
prediction on forget confidences, SVC fit on retain[:n_test] (members) vs
test (nonmembers) true-class confidences.

Dev replica: test -> held-out H = target (2500) + selection (1250) under the
same model; retain[:n_test] -> first |H| entries of the fixed 5000 retain
subset.  The audit fold (1250) is never used here.
"""
from __future__ import annotations

import numpy as np
from sklearn.svm import SVC

HELD = ("target", "selection")


def mia(member_conf, nonmember_conf, query_conf):
    x = np.concatenate([member_conf, nonmember_conf]).reshape(-1, 1).astype(np.float64)
    y = np.concatenate([np.ones(len(member_conf)), np.zeros(len(nonmember_conf))])
    clf = SVC(C=3, gamma="auto", kernel="rbf").fit(x, y)
    return float(1.0 - clf.predict(np.asarray(query_conf, dtype=np.float64).reshape(-1, 1)).mean())


def metrics(d, held=HELD):
    """d: mapping with <subset>_confidence / <subset>_error arrays."""
    hc = np.concatenate([np.asarray(d[f"{h}_confidence"]) for h in held])
    he = np.concatenate([np.asarray(d[f"{h}_error"]) for h in held])
    rc = np.asarray(d["retain_subset_confidence"])[:len(hc)]
    fc, fe = np.asarray(d["forget_confidence"]), np.asarray(d["forget_error"])
    return {"ua": 100 * float(fe.mean()), "ra": 100 * (1 - float(np.asarray(d["retain_subset_error"]).mean())),
            "ta": 100 * (1 - float(he.mean())), "mia": 100 * mia(rc, hc, fc),
            # Retrain-free indistinguishability diagnostics under the same model
            "held_nonmember_rate": 100 * mia(rc, hc, hc)}


def gap(m, ref):
    g = {k: abs(m[k] - ref[k]) for k in ("ua", "ra", "ta", "mia")}
    g["avg"] = sum(g.values()) / 4
    return g
