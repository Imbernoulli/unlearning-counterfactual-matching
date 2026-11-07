#!/bin/bash
# Confirmation: one frozen config, fixed epoch count, returned checkpoint = last.pt.
#   s42      : the development split (reported separately, never pooled)
#   s43,s44  : fresh deletion splits, canonical Original (codex references)
#   s45..s49 : fresh deletion splits, canonical Original (references trained 2026-11-06 night)
#   o142..o542: independent Originals, deletion split 42 (same forget set as dev; varies Original only)
# usage: confirm_runs.sh TAG GPU_LIST PER_GPU [--] <frozen unlearn_dev.py args incl. --epochs>
# Retrain refs are passed only for dev-metric logging; training and last.pt never depend on them.
TAG=$1; GPULIST=$2; PER=$3; shift 3
[ "$1" = "--" ] && shift
ARGS="$*"
W=/mnt/datablaze2/bohanlyu/unlearning-20261106
S=$W/snapshot/out
SEL=$S/optimization/r4_energy_joint_C_a1/seed42_split42/selection_audit_split.json  # calibration folds do not depend on the forget seed
PY=${PY:-/srv/home/bohanlyu/envs/unlearn-venv/bin/python}
CANON=/srv/home/bohanlyu/data_mlsbench/unlearn-lab/original/cifar10_resnet18_original.pt
cd /srv/home/bohanlyu/MLS-Bench/research/unlearning_matching/devgap
IFS=, read -ra GPUS <<< "$GPULIST"
slots=(); for g in "${GPUS[@]}"; do for i in $(seq 1 $PER); do slots+=($g); done; done
n=${#slots[@]}
runs=("s42|42|$S/optimization/r4_energy_joint_C_a1/seed42_split42/splits.json|$CANON|$W/devgap/seed42/retrain.npz")
for s in 43 44; do runs+=("s$s|$s|$S/eval-only/canonical/seed${s}_split${s}/splits.json|$CANON|$W/devgap/seed$s/retrain.npz"); done
for s in 45 46 47 48 49; do
  f=$W/fresh_splits/out/seed${s}_split${s}/splits.json
  [ -f "$f" ] && runs+=("s$s|$s|$f|$CANON|")
done
for o in 142 242 342 442 542; do
  B=$S/final_replicates/original_seed$o/baselines/seed${o}_split42
  runs+=("o$o|$o|$B/splits.json|$S/independent_originals/seed$o/original.pt|$W/devgap/orig$o/retrain.npz")
done
ONLY=${ONLY:-}
i=0
for r in "${runs[@]}"; do
  IFS='|' read -r name seed splits orig ref <<< "$r"
  if [ -n "$ONLY" ] && [[ ",$ONLY," != *",$name,"* ]]; then continue; fi
  [ -f $W/devgap/confirm/$TAG/$name/summary.json ] && continue
  g=${slots[$((i % n))]}
  refarg=""; [ -n "$ref" ] && [ -f "$ref" ] && refarg="--retrain-ref $ref"
  ( CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 $PY unlearn_dev.py --data ${DATA:-/dev/shm/unlearn_cifar10.npz} \
      --original $orig --splits $splits --selection-audit $SEL $refarg --seed $seed \
      --out $W/devgap/confirm/$TAG/$name --threads 3 $ARGS > $W/devgap/logs/confirm_${TAG}_$name.log 2>&1 ) &
  i=$((i+1))
  if [ $((i % n)) -eq 0 ]; then wait; fi
done
wait
echo CONFIRM_DONE
