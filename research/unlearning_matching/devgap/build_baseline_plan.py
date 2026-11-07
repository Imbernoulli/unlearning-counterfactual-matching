"""final_eval plan for the inherited FT/GA/RL/l1-sparse baselines on the 12 confirmation runs."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_plan import refs, FIXED
W = Path("/mnt/datablaze2/bohanlyu/unlearning-20261106")
runs = []
for name in "s43,s44,s45,s46,s47,s48,s49,o142,o242,o342,o442,o542".split(","):
    splits, retrain, salun, orig = refs(name)
    d = W / ("baselines_confirm/out" if name.startswith("s") else f"baselines_confirm/out_o{name[1:]}") / (
        f"seed{name[1:]}_split{name[1:]}" if name.startswith("s") else f"seed{name[1:]}_split42")
    cands = {m: str(d / f"{m}.pt") for m in ("ft", "ga", "rl", "l1_sparse") if (d / f"{m}.pt").exists()}
    if cands:
        runs.append({"id": name, "splits": str(splits), "retrain": str(retrain), "candidates": cands})
json.dump({"tag": "baselines", "fixed_reference": FIXED, "runs": runs}, open(sys.argv[1], "w"), indent=1)
print(sum(len(r["candidates"]) for r in runs), "candidates in", len(runs), "runs")
