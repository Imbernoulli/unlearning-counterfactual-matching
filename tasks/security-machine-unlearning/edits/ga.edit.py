"""Baseline `ga` for security-machine-unlearning: replaces the whole editable file.

Generated from the ported method source.
"""

_FILE = "unlearn-lab/unlearn/custom_unlearning.py"

_CONTENT = r'''"""GA: gradient ascent on the forget set (Thudi et al., EuroS&P 2022; SalUn baseline).

Port of Classification/unlearn/GA.py:GA (non-imagenet branch) + impl.py:\n5 passes over D_f with loss -CE, SGD(momentum 0.9, weight decay 5e-4), constant\nlr.  lr per setting from the paper's interval [1e-5, 1e-3].
"""

import torch
import torch.nn.functional as F

HP = {'cifar10_random10': {'lr': 0.001, 'epochs': 5}, 'cifar100_random10': {'lr': 0.001, 'epochs': 5}, 'cifar10_random50': {'lr': 0.001, 'epochs': 5}}

class UnlearningMethod:
    def plan(self, ctx):
        self.hp = HP[ctx.setting]
        return [{"set": "forget"} for _ in range(self.hp["epochs"])]

    def begin(self, params, ctx):
        self.opt = torch.optim.SGD(list(params.values()), lr=self.hp["lr"], momentum=0.9,
                                   weight_decay=5e-4)

    def loss(self, logits, labels, is_forget, ids, info):
        return -F.cross_entropy(logits, labels)

    def update(self, params, info):
        self.opt.step()
'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 1, "end_line": 51, "content": _CONTENT}
]
