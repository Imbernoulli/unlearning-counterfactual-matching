"""CPU-only descriptive analysis of recovered calibration studies.

Example:
  python analyze_calibration.py --directory /path/to/optimization \
      --out-dir /path/to/analysis

Directories are searched only for calibration_table.json. Companion reads
are limited to preregister.json, summary.json, history.jsonl and run.log;
no checkpoint, distribution NPZ, audit prediction or test result is opened.
Default planned grids are study_round{1,2,3,4,5}.json and round3_recovery.json
beside this script. --grid can explicitly replace that list.

This is exploratory discovery-seed analysis. Selection J is NOT AvgGap.
Family minima reflect adaptive hyperparameter search, not causal rankings.
Pairs must match complete training-update budgets, not just selected epoch;
same update budgets do not imply equal FLOPs for dynamic ranks/localization.
Missing, failed, corrupt and incomplete trials never receive a zero score.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import tempfile

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

HERE = Path(__file__).resolve().parent
DISCLAIMER = ("Discovery seed 42; adaptive selection-only exploration. J=max(error-gap probability, "
              "confidence KS), NOT Retrain AvgGap. No audit/test predictions were read. "
              "Single-run differences, especially below sampling variation, are not evidence of improvement.")
PAIR_FIELDS = ("schedule", "lr", "alpha", "bn_mode", "loss", "target_kind", "seed", "split_seed",
               "anchor_weight", "momentum", "weight_decay", "batch_size", "epochs",
               "forget_samples_per_epoch", "retain_samples_per_epoch", "updates_per_epoch",
               "retain_subset", "selection_size", "train_sha256", "original_sha256", "harness_sha256")


def load(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def planned_trials(grids):
    planned, failures, sources = {}, {}, []
    for path in grids:
        grid = load(path)
        sources.append(dict(path=str(path.resolve()), sha256=sha256(path)))
        for candidate in [*grid.get("candidates", []), *([grid["smoke"]] if "smoke" in grid else [])]:
            identity = candidate["id"]
            if identity in planned and planned[identity] != candidate:
                raise ValueError(f"conflicting preregistered configuration: {identity}")
            planned[identity] = candidate
        for identity in grid.get("infrastructure_failed_attempts", []):
            failures[identity] = grid.get("recovery_note", "grid records infrastructure failure")
    return planned, failures, sources


def artifact_directory(root, identity, schedule):
    if Path(identity).name != identity or schedule not in ("A", "B", "C", "D"):
        raise ValueError("unsafe trial id or unknown schedule")
    paths = list((root / "out" / "optimization" / identity).glob(f"seed*_split*/{schedule}"))
    if len(paths) != 1:
        raise ValueError(f"expected one recovered run directory; found {len(paths)}")
    return paths[0]


def normalize(row, root, plan):
    run = artifact_directory(root, row["id"], row.get("method", plan.get("schedule")))
    raw = load(run.parent / "preregister.json")["configuration"]
    configuration = dict(raw.get("arguments", {}), **raw)
    summary = load(run / "summary.json")
    schedule = summary["schedule"]
    history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines() if line.strip()]
    endpoint = "after_retain" if schedule == "A" else "epoch_end"
    epochs = int(summary["epochs"])
    ends = [entry for entry in history if entry.get("stage") == endpoint]
    if [entry["epoch"] for entry in ends] != list(range(1, epochs + 1)):
        raise ValueError("incomplete or duplicate epoch endpoints")
    stage = "initial" if row["epoch"] == 0 else endpoint
    selected = [entry for entry in history if entry.get("epoch") == row["epoch"] and entry.get("stage") == stage]
    if len(selected) != 1 or selected[0]["metrics"]["selection_J"] != row["J"]:
        raise ValueError("table J/epoch does not match exact history record")
    if summary["selection"]["epoch"] != row["epoch"] or summary["selection"]["selection_J"] != row["J"]:
        raise ValueError("table selection differs from final summary")
    metrics = selected[0]["metrics"]
    for key in ("J", "retain_drift_pp", "selection_drift_pp"):
        if not finite(row.get(key)):
            raise ValueError(f"missing/nonfinite selected metric: {key}")
    scope = summary.get("scope", configuration.get("scope", "all"))
    geometry = summary.get("geometry", configuration.get("geometry", "none"))
    dynamic = configuration.get("method") == "epoch_dynamic_global_quantile_logodds"
    masked = configuration.get("study_kind") == "saliency_coordinate_ablation"
    energy = "energy_statistic" in configuration
    kind = ("mask" if masked else "energy" if energy else "dynamic" if dynamic else
            "geometry" if "geometry" in configuration else "localized" if "scope" in configuration else "base")
    if masked:
        family = f"mask {configuration['mask_fraction_requested']:g}/" + row["loss"]
    elif energy:
        family = "energy/" + configuration["energy_statistic"]
    elif row["loss"] == "margin_ascent":
        family = "mean-logodds ascent control"
    elif dynamic:
        family = "dynamic ranks"
    elif kind == "geometry":
        family = "PCGrad" if geometry == "symmetric_pcgrad" else "geometry control"
    elif scope != "all":
        family = "layer4+fc" if scope == "layer4_fc" else scope
    else:
        family = ("rank donor" if row["target_kind"] == "rank_donor" else "fixed quantile") + "/" + row["loss"]
    batch = configuration.get("batch_size", 256)
    nf, nr = summary["forget_samples_per_epoch"], summary["retain_samples_per_epoch"]
    updates = math.ceil(nr / batch) + (math.ceil(nf / batch) if schedule == "A" else 0)
    selection_size = configuration.get("selection_size", selected[0]["distributions_summary"]["selection"]["all"]["n"])
    harness = configuration.get("calibration_harness_sha256", configuration.get("inherited_scaffold_sha256",
                               configuration.get("source_sha256")))
    result = dict(row, status="complete", family=family, kind=kind, scope=scope, geometry=geometry,
        rank_mode="distribution" if energy else "dynamic" if dynamic else "fixed", schedule=schedule,
        mask_fraction=configuration.get("mask_fraction_requested", 0.),
        seed=summary["seed"], split_seed=summary["split_seed"], epochs=epochs,
        anchor_weight=summary["anchor_weight"], momentum=configuration["momentum"],
        weight_decay=configuration["weight_decay"], batch_size=batch,
        forget_samples_per_epoch=nf, retain_samples_per_epoch=nr, updates_per_epoch=updates,
        total_forget_samples=epochs * nf, total_retain_samples=epochs * nr,
        total_optimizer_updates=epochs * updates, selected_optimizer_updates=row["epoch"] * updates,
        retain_subset=configuration.get("retain_subset"), selection_size=selection_size,
        train_sha256=configuration.get("train_sha256"), original_sha256=configuration.get("original_sha256"),
        harness_sha256=harness, directory=str(run.resolve()),
        full_registered_horizon_train_s=row.get("train_s"),
        full_registered_horizon_wall_s=row.get("wall_s"),
        selected_checkpoint_train_s=None,
        timing_scope="collector train_s/wall_s cover the full registered horizon, not best-checkpoint cost; setup overhead can be excluded by the inherited timer",
        geometry_statistics=summary.get("geometry_statistics"),
        selected_classwise_KS=metrics.get("classwise_confidence_ks"),
        trace=[dict(epoch=entry["epoch"], J=entry["metrics"]["selection_J"],
                    feasible=entry["metrics"]["feasible"],
                    retain_drift_pp=entry["metrics"]["retain_subset_accuracy_drift_pp"],
                    selection_drift_pp=entry["metrics"]["selection_accuracy_drift_pp"])
               for entry in history if entry.get("stage") in ("initial", endpoint)])
    # Assert the paired artifacts are genuinely the claimed trial, rather
    # than interpreting a missing configuration field as a matching zero.
    for key, meta in (("lr", "lr"), ("alpha", "forget_weight"), ("loss", "loss"),
                      ("target_kind", "target_kind"), ("bn_mode", "bn_mode")):
        if result[key] != summary[meta]:
            raise ValueError(f"table/summary mismatch: {key}")
    if nf <= 0 or nr <= 0 or selection_size <= 0:
        raise ValueError("nonpositive sample budget")
    return result


def collect(directories, grids):
    planned, failures, grid_sources = planned_trials(grids)
    tables = sorted({path.resolve() for root in directories for path in
                     ([root] if root.is_file() else root.rglob("calibration_table.json"))})
    if not tables:
        raise ValueError("no calibration_table.json found")
    by_id, table_sources = defaultdict(list), []
    for table in tables:
        if table.name != "calibration_table.json":
            raise ValueError("inputs must be named calibration_table.json")
        contents = load(table)
        table_sources.append(dict(path=str(table), sha256=sha256(table), updated_utc=contents.get("updated_utc")))
        for row in contents["rows"]:
            try:
                trial = normalize(row, table.parent, planned.get(row["id"], {}))
            except (OSError, ValueError, KeyError, TypeError) as error:
                trial = dict(id=row.get("id", "unknown"), status="invalid_or_incomplete_artifacts", reason=str(error))
            trial["source_tables"] = [str(table)]
            by_id[trial["id"]].append(trial)
    # An attempt can fail before the collector sees a summary/table row.
    # Include every local attempt directory, and distinguish explicit error
    # evidence from merely pending or not-yet-recovered output.
    attempts = defaultdict(list)
    for root in {table.parent for table in tables}:
        for path in (root / "out" / "optimization").glob("*"):
            if path.is_dir():
                attempts[path.name].append(path)
    records = []
    for identity in sorted(set(planned) | set(by_id) | set(failures) | set(attempts)):
        copies = by_id[identity]
        complete = [trial for trial in copies if trial["status"] == "complete"]
        if complete:
            comparable = [dict((key, trial.get(key)) for key in (*PAIR_FIELDS, "J", "epoch", "fallback", "source_sha256"))
                          for trial in complete]
            if any(value != comparable[0] for value in comparable[1:]):
                trial = dict(id=identity, status="conflicting_recovered_copies", reason="same id has inconsistent metrics/configuration")
            else:
                trial = dict(complete[-1])
        elif identity in failures:
            trial = dict(id=identity, status="failed_infrastructure", reason=failures[identity])
        elif copies:
            trial = dict(copies[-1])
        else:
            trial = dict(id=identity, status="incomplete_or_not_recovered", reason="planned trial has no validated completed calibration table row")
            for attempt in attempts[identity]:
                log = attempt / "run.log"
                if log.exists():
                    with log.open("rb") as stream:
                        stream.seek(max(0, log.stat().st_size - 12000))
                        tail = stream.read().decode("utf-8", errors="replace")
                    if "Traceback (most recent call last)" in tail:
                        trial.update(status="failure_log_without_complete_summary",
                                     reason="; ".join(tail.strip().splitlines()[-3:])[:500])
                        break
        trial["source_tables"] = sorted({source for copy in copies for source in copy["source_tables"]})
        trial["planned"] = planned.get(identity)
        records.append(trial)
    return records, dict(tables=table_sources, grids=grid_sources)


def matched_pairs(records):
    complete = [trial for trial in records if trial["status"] == "complete"]
    output, matched = [], set()
    for left, right in combinations(complete, 2):
        category = None
        allowed_differences = ()
        if left["kind"] == right["kind"] == "geometry" and left["geometry"] != right["geometry"] and left["scope"] == right["scope"]:
            category = "geometry none vs PCGrad"
            if left["geometry"] != "none":
                left, right = right, left
        elif {left["rank_mode"], right["rank_mode"]} == {"fixed", "dynamic"} and left["kind"] in ("base", "geometry", "dynamic") and right["kind"] in ("base", "geometry", "dynamic") and left["scope"] == right["scope"] == "all" and left["geometry"] == right["geometry"] == "none":
            category = "fixed vs dynamic ranks"
            if left["rank_mode"] != "fixed":
                left, right = right, left
        elif left["scope"] != right["scope"] and "all" in (left["scope"], right["scope"]) and left["kind"] in ("base", "geometry", "localized") and right["kind"] in ("base", "geometry", "localized") and left["rank_mode"] == right["rank_mode"] and left["geometry"] == right["geometry"]:
            category = "all vs localized parameters"
            if left["scope"] != "all":
                left, right = right, left
        elif (left["kind"] == "mask") != (right["kind"] == "mask") and left["kind"] in ("base", "geometry", "mask") and right["kind"] in ("base", "geometry", "mask") and left["scope"] == right["scope"] == "all" and left["geometry"] == right["geometry"] == "none":
            category = "all vs saliency mask"
            if left["kind"] == "mask":
                left, right = right, left
        elif {left["loss"], right["loss"]} == {"margin_ascent", "logodds_mse"} and left["kind"] in ("base", "geometry") and right["kind"] in ("base", "geometry") and left["scope"] == right["scope"] == "all" and left["geometry"] == right["geometry"] == "none" and left["rank_mode"] == right["rank_mode"] == "fixed":
            category = "target matching vs mean-logodds ascent"
            allowed_differences = ("loss",)
            if left["loss"] == "margin_ascent":
                left, right = right, left
        if category is None:
            continue
        required = tuple(key for key in PAIR_FIELDS if key not in allowed_differences)
        if any(left.get(key) is None or right.get(key) is None or left[key] != right[key] for key in required):
            continue
        # Classical iid two-sample KS reference, NOT a confidence interval
        # for adaptive/paired J differences or training-set predictions.
        reference = 1.36 * math.sqrt(1 / left["forget_samples_per_epoch"] + 1 / left["selection_size"])
        output.append(dict(category=category, left=left["id"], right=right["id"],
            schedule=left["schedule"], delta_J_right_minus_left=right["J"] - left["J"],
            delta_retain_drift_pp=right["retain_drift_pp"] - left["retain_drift_pp"],
            delta_selection_drift_pp=right["selection_drift_pp"] - left["selection_drift_pp"],
            left_selected_epoch=left["epoch"], right_selected_epoch=right["epoch"],
            matching_fields={key: left[key] for key in required},
            treatment_differences={key: [left[key], right[key]] for key in allowed_differences},
            iid_KS_reference_scale=reference,
            difference_below_iid_KS_reference=abs(right["J"] - left["J"]) < reference,
            interpretation="descriptive discovery difference only; no significance or improvement claim",
            budget_note="same complete training sample/update budget; dynamic refresh, mask construction, and parameter scope can change FLOPs/runtime; D is not paired to A/B/C; ascent changes objective/loss scale at fixed alpha"))
        matched.update((left["id"], right["id"]))
    return output, [trial["id"] for trial in complete if trial["id"] not in matched]


def family_summaries(records):
    groups = defaultdict(list)
    for trial in records:
        if trial["status"] == "complete":
            groups[(trial["family"], trial["schedule"], trial["bn_mode"])].append(trial)
    return [dict(family=key[0], schedule=key[1], bn_mode=key[2], completed_search_trials=len(trials),
                 best_id=best["id"], J=best["J"], epoch=best["epoch"], epochs=best["epochs"],
                 retain_drift_pp=best["retain_drift_pp"], selection_drift_pp=best["selection_drift_pp"],
                 fallback=best["fallback"])
            for key, trials in sorted(groups.items())
            for best in [min(trials, key=lambda trial: (trial["J"], trial["epoch"], trial["id"]))]]


def format_value(value):
    return f"{value:.5g}" if finite(value) else "missing"


def markdown(result):
    lines = ["# Calibration discovery analysis", "", DISCLAIMER, "",
        "Only completed, internally consistent runs enter plots/minima. Recovered copies are deduplicated, not treated as independent seeds. "
        "Different search counts and epoch horizons make family minima descriptive. Accuracy drifts use percentage points; positive means higher accuracy. "
        "J is in probability units. Original fallback is a valid recorded fallback, never an imputed score. "
        "Collector train_s/wall_s are full registered horizon costs, never best-checkpoint costs; no proportional best-time estimate is made.", "",
        "## Method-family minima", "", "| Family / schedule / BN | Completed search trials | Selected J | Retain drift pp | Selection drift pp | Epoch / budget | Trial |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for row in result["families"]:
        lines.append(f"| {row['family']} / {row['schedule']} / {row['bn_mode']} | {row['completed_search_trials']} | {row['J']:.5f} | "
                     f"{row['retain_drift_pp']:+.3f} | {row['selection_drift_pp']:+.3f} | {row['epoch']}/{row['epochs']} | {row['best_id']}"
                     f"{' (Original fallback)' if row['fallback'] else ''} |")
    lines.extend(["", "## Matched comparisons", "", "Each pair matches: " + ", ".join(PAIR_FIELDS) + ", "
        "except the explicitly changed loss for target-matching versus mean-logodds-ascent controls. "
        "Selected epochs may differ because the same selection rule chooses within the matched complete horizon. "
        "The full matching values and treatment differences are in analysis.json. Same update budgets do not imply equal FLOPs: dynamic ranking adds forward passes, mask construction adds a forget gradient pass, and localized backward is cheaper. "
        "Ascent is the inherited margin_ascent CLI objective (minimize mean true-class log-odds), not signed-margin loss or standard negative-CE GA. "
        "Only equal complete horizons are paired: a completed 30-epoch ascent can match geometry-none at 30 epochs, while an all-network 100-epoch run is not retrospectively reselected from its first 30 epochs. "
        "Schedule A has 177 updates/epoch, B/C 159, D 18 in the current 256-batch setup; D uses 4,500 retain samples rather than 40,500.", "",
        "The 1.36 sqrt(1/n_forget+1/n_selection) KS scale is an iid reference only, not a confidence interval for adaptive paired J changes. "
        "No difference is declared statistically significant; smaller differences require fresh splits/seeds.", "",
        "| Comparison | Left | Right | Right minus left J | Retain drift delta pp | Selection drift delta pp | Selected epochs | Below iid KS reference? |",
        "|---|---|---|---:|---:|---:|---:|---|"])
    for pair in result["pairs"]:
        lines.append(f"| {pair['category']} ({pair['schedule']}) | {pair['left']} | {pair['right']} | {pair['delta_J_right_minus_left']:+.5f} | "
                     f"{pair['delta_retain_drift_pp']:+.3f} | {pair['delta_selection_drift_pp']:+.3f} | "
                     f"{pair['left_selected_epoch']} / {pair['right_selected_epoch']} | {pair['difference_below_iid_KS_reference']} |")
    if not result["pairs"]:
        lines.append("| No fully matched completed pair yet | — | — | missing | missing | missing | — | — |")
    lines.extend(["", "## All attempts, including failures and incomplete trials", "",
        "| Trial | Status | Selected J | Selected epoch | Reason |", "|---|---|---:|---:|---|"])
    for row in result["trials"]:
        reason = row.get("reason", "").replace("|", "/").replace("\n", " ")
        lines.append(f"| {row['id']} | {row['status']} | {format_value(row.get('J'))} | {row.get('epoch', 'missing')} | {reason} |")
    lines.extend(["", "Completed trials without a fully matched comparator: " + ", ".join(result["unmatched_complete_trials"]), ""])
    return "\n".join(lines)


def plots(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 10, "axes.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    complete = [trial for trial in result["trials"] if trial["status"] == "complete"]
    families = sorted({trial["family"] for trial in complete})
    palette = {name: plt.get_cmap("tab20")(index % 20) for index, name in enumerate(families)}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    fig.suptitle("Calibration discovery: objectives, utility, training horizon and matched mechanisms", fontsize=14)
    chosen = sorted(result["families"], key=lambda row: row["J"])[:14]
    labels = [f"{row['family']} / {row['schedule']} / {row['bn_mode']} (n={row['completed_search_trials']})" for row in chosen]
    axis = axes[0, 0]
    axis.barh(range(len(chosen)), [row["J"] for row in chosen], color=[palette[row["family"]] for row in chosen])
    axis.set_yticks(range(len(chosen)), labels, fontsize=6.5)
    axis.invert_yaxis(); axis.set_xlabel("Selected J (probability units; lower observed value)")
    axis.set_title("Family minima after adaptive search (top 14; not causal)")
    axis = axes[0, 1]
    for family in families:
        for schedule in ("A", "B", "C", "D"):
            trials = [trial for trial in complete if trial["family"] == family and trial["schedule"] == schedule]
            if trials:
                axis.scatter([trial["retain_drift_pp"] for trial in trials], [trial["selection_drift_pp"] for trial in trials],
                             color=palette[family], marker="s" if schedule == "D" else "o", s=24, alpha=.7)
    axis.axvline(-.5, color="black", linestyle=":"); axis.axhline(-1., color="black", linestyle=":")
    axis.set(xlabel="Retain accuracy drift (pp; positive is better)", ylabel="Selection accuracy drift (pp)",
             title="Selected utility; square = separately budgeted D")
    axis = axes[0, 2]
    for family in families:
        trials = [trial for trial in complete if trial["family"] == family]
        axis.scatter([trial["epoch"] for trial in trials], [trial["J"] for trial in trials],
                     color=palette[family], s=25, label=family)
    axis.set(xlabel="Selected epoch within full registered horizon", ylabel="Selected J", title="Selected epoch; full registered horizons differ")
    if families:
        axis.legend(fontsize=5.5, ncol=2, loc="upper right")
    axis = axes[1, 0]
    pair_ids = {value for pair in result["pairs"] for value in (pair["left"], pair["right"])}
    traces = [trial for trial in complete if trial["id"] in pair_ids]
    for trial in traces:
        axis.plot([entry["epoch"] for entry in trial["trace"]], [entry["J"] for entry in trial["trace"]],
                  color=palette[trial["family"]], alpha=.55, linewidth=1,
                  linestyle="--" if trial["schedule"] == "D" else "-")
    axis.set(xlabel="Epoch (no cross-budget causal comparison)", ylabel="Endpoint J; includes infeasible points",
             title="Completed matched-candidate trajectories")
    if not traces:
        axis.text(.5, .5, "No completed matched pair yet", transform=axis.transAxes, ha="center")
    axis = axes[1, 1]
    pairs = result["pairs"][:12]
    axis.barh(range(len(pairs)), [pair["delta_J_right_minus_left"] for pair in pairs], color="#6b8eb5")
    axis.set_yticks(range(len(pairs)), [f"{pair['category']} / {pair['schedule']}\n{pair['left']} → {pair['right']}" for pair in pairs], fontsize=5.5)
    axis.axvline(0, color="black", linewidth=.6); axis.invert_yaxis()
    axis.set(xlabel="J(right) - J(left); descriptive only", title="Strictly matched update-budget pairs (first 12)")
    axis = axes[1, 2]
    axis.axis("off")
    counts = Counter(trial["status"] for trial in result["trials"])
    status = "\n".join(f"{name}: {count}" for name, count in sorted(counts.items()))
    axis.text(0, 1, status + "\n\nJ is NOT AvgGap.\nDiscovery seed 42; no audit/test read.\n"
              "Failures/incomplete attempts are listed, never zero.\n"
              "No statistical or improvement claim from these plots.\n"
              "Small differences need fresh seeds/splits.\n"
              "D uses 4,500 retain samples/epoch; C uses 40,500.\n"
              "Matching update budgets does not match FLOPs.\n"
              "train_s/wall_s: full registered horizon, not best time.\n"
              "See analysis.md/analysis.json for every trial and pair.",
              va="top", fontsize=9, linespacing=1.5, transform=axis.transAxes)
    fig.savefig(out / "calibration_overview.pdf")
    fig.savefig(out / "calibration_overview.png", dpi=140)
    plt.close(fig)
    # Full family panels complement the compact one-page overview.
    fig, axes = plt.subplots(1, 3, figsize=(16, max(4, len(result["families"]) * .24)), layout="constrained")
    rows = sorted(result["families"], key=lambda row: (row["family"], row["schedule"], row["bn_mode"]))
    y = np.arange(len(rows))
    names = [f"{row['family']} / {row['schedule']} / {row['bn_mode']}" for row in rows]
    axes[0].barh(y, [row["J"] for row in rows], color=[palette[row["family"]] for row in rows])
    axes[0].set_yticks(y, names, fontsize=7)
    axes[1].scatter([row["retain_drift_pp"] for row in rows], y, label="retain", s=22)
    axes[1].scatter([row["selection_drift_pp"] for row in rows], y, label="selection", s=22, marker="x")
    axes[1].legend(); axes[1].set_yticks(y, [])
    axes[2].scatter([row["epoch"] for row in rows], y, label="selected", s=20)
    axes[2].scatter([row["epochs"] for row in rows], y, label="full registered horizon", s=20, marker="|")
    axes[2].legend(); axes[2].set_yticks(y, [])
    for axis, title in zip(axes, ("Family minimum J (adaptive search)", "Accuracy drift at that checkpoint (pp)", "Selected epoch and full registered horizon")):
        axis.set_title(title); axis.invert_yaxis(); axis.grid(axis="x", alpha=.2)
    fig.savefig(out / "method_family_panels.pdf")
    fig.savefig(out / "method_family_panels.png", dpi=140)
    plt.close(fig)


def self_test():
    # Pairing must reject changed schedules/BN/horizons and missing fields.
    baseline = dict((key, 1) for key in PAIR_FIELDS)
    baseline.update(id="none", status="complete", family="control", kind="geometry", geometry="none",
        scope="all", rank_mode="fixed", schedule="C", J=.07, epoch=2,
        retain_drift_pp=0., selection_drift_pp=0., forget_samples_per_epoch=4500, selection_size=1250,
        loss="logodds_mse")
    projected = dict(baseline, id="pcgrad", geometry="symmetric_pcgrad", J=.06)
    pairs, _ = matched_pairs([baseline, projected])
    assert len(pairs) == 1 and abs(pairs[0]["delta_J_right_minus_left"] + .01) < 1e-12
    for key, value in (("schedule", "D"), ("epochs", 30), ("bn_mode", "different"), ("lr", None)):
        assert not matched_pairs([baseline, dict(projected, **{key: value})])[0]
    assert not matched_pairs([baseline, dict(projected, status="failed_infrastructure")])[0]
    dynamic = dict(baseline, id="dynamic", kind="dynamic", rank_mode="dynamic")
    assert matched_pairs([baseline, dynamic])[0][0]["category"] == "fixed vs dynamic ranks"
    local = dict(baseline, id="fc", kind="localized", scope="fc")
    assert matched_pairs([baseline, local])[0][0]["category"] == "all vs localized parameters"
    mask = dict(baseline, id="mask50", kind="mask", mask_fraction=.5)
    assert matched_pairs([baseline, mask])[0][0]["category"] == "all vs saliency mask"
    ascent = dict(baseline, id="ascent", kind="base", loss="margin_ascent")
    pair = matched_pairs([baseline, ascent])[0][0]
    assert pair["category"] == "target matching vs mean-logodds ascent" and "loss" not in pair["matching_fields"]
    assert pair["treatment_differences"] == {"loss": ["logodds_mse", "margin_ascent"]}
    assert not matched_pairs([dict(baseline, epochs=100), ascent])[0]
    assert not matched_pairs([mask, dynamic])[0]
    assert format_value(None) == "missing" and format_value(0.) == "0"
    with tempfile.TemporaryDirectory(prefix="calibration-analysis-check-") as directory:
        path = Path(directory) / "grid.json"
        save(path, dict(candidates=[dict(id="failed")], infrastructure_failed_attempts=["failed"]))
        planned, failures, _ = planned_trials([path])
        assert "failed" in failures and "failed" in planned
    print("analysis CPU checks passed: strict pair matching, incomplete exclusion, dynamic/scope pairs, missing vs zero, failure ledger")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, action="append", help="directory tree or calibration_table.json; repeatable")
    parser.add_argument("--grid", type=Path, action="append", help="planned grid(s), replacing default round1/2/3/recovery grids")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.directory or not args.out_dir:
        parser.error("provide --directory and --out-dir, or --self-test")
    grids = args.grid or [path for name in ("study_round1.json", "study_round2.json", "study_round3.json", "study_round3_recovery.json", "study_round4.json", "study_round5.json")
                         if (path := HERE / name).exists()]
    records, sources = collect(args.directory, grids)
    pairs, unmatched = matched_pairs(records)
    result = dict(schema=1, created_utc=datetime.now(timezone.utc).isoformat(),
        analyzer_sha256=sha256(__file__), disclaimer=DISCLAIMER, sources=sources,
        status_counts=dict(Counter(trial["status"] for trial in records)),
        families=family_summaries(records), pairs=pairs, unmatched_complete_trials=unmatched, trials=records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save(args.out_dir / "analysis.json", result)
    (args.out_dir / "analysis.md").write_text(markdown(result))
    plots(result, args.out_dir)
    print(json.dumps(dict(output=str(args.out_dir.resolve()), status_counts=result["status_counts"], matched_pairs=len(pairs))))


if __name__ == "__main__":
    main()
