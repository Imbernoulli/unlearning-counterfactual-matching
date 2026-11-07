# Unlearning as Counterfactual Matching

Course project for random-deletion machine unlearning on CIFAR-10 with ResNet-18.
Goal: after unlearning, deleted examples should be indistinguishable from examples the model never saw,
while knowledge shared with the remaining data is kept. Retrain without the deleted data is the evaluation reference.

## Layout
- `research/unlearning_matching/`: first-round runner, target constructions, calibration studies, probes, Daytona tooling, protocols.
- `research/unlearning_matching/devgap/`: development replica of the benchmark metrics, the budget-matched trainer
  `unlearn_dev.py`, confirmation launchers, official-test evaluation `final_eval.py`, frozen configurations.
- `tasks/security-machine-unlearning/`: benchmark metric code and the inherited baseline ports.
- `docs/`: derivations, mechanism probes, diagnosis notes and figures.
- `results/`: official-test results of the confirmation runs and the first-round summary.

## Main result
On twelve confirmation runs, seven fresh deletion splits and five independently trained Originals, the frozen
configuration A2 reaches an average gap to Retrain of 0.336 against 1.226 for SalUn and 1.141 for random labelling,
winning all twelve runs against SalUn, within the benchmark pass budget. See `results/` and the progress report.
