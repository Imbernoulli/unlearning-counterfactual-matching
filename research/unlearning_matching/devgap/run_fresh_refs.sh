#!/bin/bash
# Fresh deletion splits (canonical Original): paired Retrain (182 ep, BF16 as the earlier references) + SalUn baseline.
GPU=$1; shift
cd /srv/home/bohanlyu/MLS-Bench
PY=/srv/home/bohanlyu/envs/unlearn-venv/bin/python
for s in "$@"; do
  for ph in reference experiments; do
    CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 $PY -u research/unlearning_matching/run_experiments.py --phase $ph \
      --data-dir .local-saves/unlearning-project-20261106/data --out-dir /mnt/datablaze2/bohanlyu/unlearning-20261106/fresh_splits/out \
      --seed $s --split-seed $s --threads 3 --amp --methods original,salun || echo "FAIL seed $s phase $ph"
  done
done
echo REFS_DONE
