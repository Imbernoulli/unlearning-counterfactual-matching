#!/bin/bash
# usage: run_grid.sh GRIDFILE GPU_LIST PER_GPU
# GRIDFILE lines: name|extra args
W=/mnt/datablaze2/bohanlyu/unlearning-20261106
SEED=${SEED:-42}
S=${SPLITDIR:-$W/snapshot/out/optimization/r4_energy_joint_C_a1/seed42_split42}
REF=${REF:-$W/devgap/seed42/retrain.npz}
ORIG=${ORIG:-/srv/home/bohanlyu/data_mlsbench/unlearn-lab/original/cifar10_resnet18_original.pt}
cd /srv/home/bohanlyu/MLS-Bench/research/unlearning_matching/devgap
IFS=, read -ra GPUS <<< "$2"
slots=()
for g in "${GPUS[@]}"; do for i in $(seq 1 $3); do slots+=($g); done; done
n=${#slots[@]}
i=0
while IFS='|' read -r name extra; do
  [ -z "$name" ] && continue
  g=${slots[$((i % n))]}
  ( CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 ${PY:-/srv/home/bohanlyu/envs/unlearn-venv/bin/python} unlearn_dev.py --data ${DATA:-/dev/shm/unlearn_cifar10.npz} --original $ORIG --splits $S/splits.json --selection-audit $S/selection_audit_split.json --retrain-ref $REF --seed $SEED --out $W/devgap/runs/$name --threads 3 $extra > $W/devgap/logs/$name.log 2>&1 ) &
  i=$((i+1))
  if [ $((i % n)) -eq 0 ]; then wait; fi
done < "$1"
wait
