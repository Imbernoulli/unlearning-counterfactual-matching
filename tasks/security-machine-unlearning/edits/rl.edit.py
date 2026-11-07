"""Baseline `rl` for security-machine-unlearning: replaces the whole editable file.

Generated from the ported method source.
"""

_FILE = "unlearn-lab/unlearn/custom_unlearning.py"

_CONTENT = r'''"""RL: random labelling (Golatkar et al., CVPR 2020; SalUn baseline).

Port of Classification/unlearn/RL.py:RL with mask=None + impl.py:\nSGD(momentum 0.9, weight decay 5e-4), constant lr, 10 epochs.  lr per\ndataset from the paper's interval [1e-3, 1e-1] (App. C.1).
"""

import torch
import torch.nn.functional as F

HP = {'cifar10_random10': {'lr': 0.01, 'epochs': 10}, 'cifar100_random10': {'lr': 0.0014, 'epochs': 10}, 'cifar10_random50': {'lr': 0.01, 'epochs': 10}}

class UnlearningMethod:
    # Classification/unlearn/RL.py:RL.  CIFAR-10 / SVHN branch (RL.py:633-700): per
    # epoch a pass over D_f with fresh uniform random labels per batch, then a pass
    # over D_r.  CIFAR-100 / TinyImageNet branch (RL.py:575-631): D_f relabelled
    # uniformly at random once per epoch, concatenated with D_r, shuffled together.
    def _rl_plan(self, ctx):
        self.concat = ctx.dataset in ("cifar100", "tinyimagenet")
        self.num_classes = ctx.num_classes
        self.n_forget = ctx.n_forget
        self.table, self.table_pass = None, -1
        if self.concat:
            return [{"set": "both"} for _ in range(self.hp["epochs"])]
        out = []
        for _ in range(self.hp["epochs"]):
            out += [{"set": "forget"}, {"set": "retain"}]
        return out

    def _rl_loss(self, logits, labels, is_forget, ids, info):
        if self.concat:
            if self.table_pass != info.pass_index:
                self.table = torch.randint(0, self.num_classes, (self.n_forget,),
                                           device=logits.device)
                self.table_pass = info.pass_index
            rand = self.table[ids.clamp(max=self.n_forget - 1)]
        else:
            rand = torch.randint(0, self.num_classes, labels.shape, device=labels.device)
        return F.cross_entropy(logits, torch.where(is_forget, rand, labels))

    def plan(self, ctx):
        self.hp = HP[ctx.setting]
        return self._rl_plan(ctx)

    def begin(self, params, ctx):
        self.opt = torch.optim.SGD(list(params.values()), lr=self.hp["lr"], momentum=0.9,
                                   weight_decay=5e-4)

    def loss(self, logits, labels, is_forget, ids, info):
        return self._rl_loss(logits, labels, is_forget, ids, info)

    def update(self, params, info):
        self.opt.step()
'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 1, "end_line": 51, "content": _CONTENT}
]
