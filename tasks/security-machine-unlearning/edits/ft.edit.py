"""Baseline `ft` for security-machine-unlearning: replaces the whole editable file.

Generated from the ported method source.
"""

_FILE = "unlearn-lab/unlearn/custom_unlearning.py"

_CONTENT = r'''"""FT: fine-tune on the retain set (Warnecke et al., NDSS 2023; SalUn baseline).

Port of Classification/unlearn/FT.py:FT_iter (with_l1=False) + impl.py:\n10 passes over D_r, CE, SGD(momentum 0.9, weight decay 5e-4), constant lr (the\nMultiStepLR milestones 91/136 are never reached).  lr per setting from the\npaper's search interval [1e-3, 1e-1] (App. C.1).
"""

import torch
import torch.nn.functional as F

HP = {'cifar10_random10': {'lr': 0.03, 'epochs': 10}, 'cifar100_random10': {'lr': 0.042, 'epochs': 10}, 'cifar10_random50': {'lr': 0.042, 'epochs': 10}}

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
        self.opt.step()
'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 1, "end_line": 51, "content": _CONTENT}
]
