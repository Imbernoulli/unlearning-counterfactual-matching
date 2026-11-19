# Unlearning lane log (Claude, from 2026-11-06 21:10 PST)

## State inherited
- Daytona box 34d0d83c was deleted 05:08Z after a verified 2.2 GB snapshot:
  `.local-saves/unlearning-project-20261106/optimization/daytona_recovery/final_snapshot.tgz`
  extracted to `/mnt/datablaze2/bohanlyu/unlearning-20261106/snapshot/out` (all confirmation
  checkpoints, Retrain/SalUn refs for seeds 42/43/44 and independent Originals 142..542).
- Frozen joint-score energy method (23 ep): dev AvgGap on seed 42 = 1.69 vs SalUn 1.15
  -> under-forgets (UA 3.7 / MIA 7.7 vs Retrain 5.0 / 12.0). Not evaluated on test.
- SalUn-residual job crashed at launch (passed original.pt where a JSON manifest was expected).

## Dev protocol (new)
- `devgap.py`: replica of benchmark UA/RA/TA/MIA with validation target+selection (3750) as the
  "test" set, retain-subset[:3750] as SVC members.  Audit fold and official test untouched.
- Dev seed 42 only, gap to its paired Retrain.  Confirmation = seeds 43/44 + Originals 142..542
  on the official test evaluator (`final_eval.py`), config + epoch frozen first.
- Insight: the SVC MIA is a single cut at confidence ~0.965 -> MIA = share of forget below it.
  Retrain held-out is less confident than Original held-out (12.1% vs 10.2% below the cut).

## Results (dev seed 42, AvgGap vs Retrain-42; SalUn = 1.147)
- grid1 (15 ep): dense GQ log-odds MSE, forget batch every retain step: ~1.0-1.1;
  KL I-projection 1.2; **target shift -1: 0.389 @ep15** (UA 5.20, TA 94.51, MIA 10.64).
- grid2 (25 ep dense, OVER BUDGET): shift -1.5 best 0.22-0.25 (UA 5.8, MIA 12.0); -2.5 overshoots;
  scale .7/.8 keeps UA too low (0.77-0.82).  Model's own held-out tail rises 10.2 -> 12-13 (coupling).
- Self-referential targets (current model's held-out): runaway, TA 94.8 -> 92.4 in 1 epoch.
- Review (Opus, 11-06 22:40): benchmark budget is 10 retain + 11 forget passes, no held-out images;
  dense runs are ~25/220 passes -> diagnostic only.  coco pool 503 all night (retry).
- Within budget (10 ep): sparse k=9 clip5 shift -1.5: 0.65 (sel 0.97); mixed both-pass 1.26-1.53
  (26-sample forget slices = noisy clipped steps, RA/TA damage); unclipped MSE/Huber diverge (frozen BN);
  front-loaded forget then retain-only: retain epochs RELEARN (MIA 10 -> 5-7, held tail below Original);
  tail-focused loss ~same as MSE (0.74/0.66).  Endpoint noise +-0.4.
- FROZEN A (frozen_A_20261107.json) = sparse k=9 shift -1.5, 10 ep; confirmation on s43..s48,
  o142..o542 with official test eval (final_eval.py) -> devgap/confirm/A_k9_sh-1.5_e10/final_test.json
- Fresh splits 45-49: Retrain+SalUn via run_fresh_refs.sh -> fresh_splits/out (all five done by 22:26).

## Paths
- code: `research/unlearning_matching/devgap/` (run_grid.sh, unlearn_dev.py, final_eval.py)
- runs/logs: `/mnt/datablaze2/bohanlyu/unlearning-20261106/devgap/{runs,logs}`
- venv: `/srv/home/bohanlyu/envs/unlearn-venv` (torch 2.5.1 cu121, numpy 1.26.4, sklearn 1.5.2)
- data copy: `/dev/shm/unlearn_cifar10.npz` (delete when done)
- GPUs: own cards only (local 3,4,5,7 run my AgentVLA jobs too); never peterchen's 0,1,2,6.
- 22:38 official test, frozen A (10 runs): 0.732 +- 0.259 vs SalUn 1.229 +- 0.197, 9/10 wins
  (t 4.55, sign p .011); fixed-ref 0.655 vs 1.238.  Residual: MIA 11.0 vs 12.7, TA 93.9 vs 94.4.
- grid8 (dev 42, 2 train seeds): endpoint phase noise confirmed.  A replicates 1.30/0.92; weight
  averaging last epoch 0.55; linear lr decay over last 2 epochs 0.43 (sel 0.69), RA->100, TA ~94.5.
- grid10 running: decay-2 + shift -1/-2, measured data-size map (x1, x2, +shift -1), decay-3.
- B (dense 20 ep, over budget) + A on s48/s49: official eval queued (bsylx0y14 waiter).
- (times above are wall-clock PST from file mtimes/git; earlier hand-written estimates were off ~1h)
- 23:05 official test, 12 runs: A 0.778 +- 0.258 vs SalUn 1.226 +- 0.184 (11/12, p .003); fresh n=7
  0.875 vs 1.169 (6/7); B (dense 20 ep, over budget) 0.403 +- 0.150 (12/12), fresh 0.332.
- Opus verification of A: pairing/args/aggregates correct, s43 CPU re-eval matches; caveats = not
  harness-legal (held-out images, clean forget forwards), o-runs reuse dev forget set.
- A2 (decay2 + measured map + shift -1) and A3 (decay2 + shift -2) frozen ~23:00, confirming.
