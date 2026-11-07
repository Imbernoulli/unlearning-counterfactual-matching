"""Baseline `l1_sparse` for security-machine-unlearning: replaces the whole editable file.

Generated from the ported method source.
"""

_FILE = "unlearn-lab/unlearn/custom_unlearning.py"

_CONTENT = r'''"""l1-sparse: FT with a decaying l1 penalty (Jia et al., NeurIPS 2023; SalUn baseline).

Port of Classification/unlearn/FT.py:FT_iter(with_l1=True) (FT_l1 / FT_prune):\nloss = CE + alpha_e * ||theta||_1 over ALL parameters, alpha_e = gamma * (1 - e / E)\n(no_l1_epochs = 0), 10 passes over D_r, SGD(momentum 0.9, wd 5e-4).  The l1 term's\ngradient alpha_e * sign(theta) is added to .grad before the step (the same\ngradient autograd gives for torch.linalg.norm(theta, 1)).  gamma from the\npaper's interval [1e-6, 1e-4], lr from [1e-3, 1e-1], per setting.
"""

import torch
import torch.nn.functional as F

HP = {'cifar10_random10': {'lr': 0.03, 'gamma': 1e-06, 'epochs': 10}, 'cifar100_random10': {'lr': 0.03, 'gamma': 3e-05, 'epochs': 10}, 'cifar10_random50': {'lr': 0.03, 'gamma': 1e-05, 'epochs': 10}}

class UnlearningMethod:
    def plan(self, ctx):
        self.hp = HP[ctx.setting]
        return [{"set": "retain"} for _ in range(self.hp["epochs"])]

    def begin(self, params, ctx):
        self.opt = torch.optim.SGD(list(params.values()), lr=self.hp["lr"], momentum=0.9,
                                   weight_decay=5e-4)

    def loss(self, logits, labels, is_forget, ids, info):
        return F.cross_entropy(logits, labels)

    def update(self, params, info):
        alpha = self.hp["gamma"] * (1 - info.pass_index / self.hp["epochs"])
        with torch.no_grad():
            for p in params.values():
                p.grad.add_(torch.sign(p), alpha=alpha)
        self.opt.step()
'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 1, "end_line": 51, "content": _CONTENT}
]
