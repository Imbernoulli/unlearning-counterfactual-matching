"""Baseline `salun` for security-machine-unlearning: replaces the whole editable file.

Generated from the ported method source.
"""

_FILE = "unlearn-lab/unlearn/custom_unlearning.py"

_CONTENT = r'''"""SalUn: saliency-masked random labelling (Fan et al., ICLR 2024, arXiv:2310.12508).

Weight saliency (Classification/generate_mask.py:save_gradient_ratio): one pass\nover D_f in eval mode accumulating the gradient of -CE; keep the globally top\n`sparsity` fraction of ALL parameter entries by |g| (mask 1), freeze the rest.\nThen RL (unlearn/RL.py) with the gradient masked, the frozen entries restored\nafter each step (weight decay would otherwise move them) and their momentum\nzeroed (RL.py:_apply_mask_to_grads / _restore_masked_params); SGD(momentum 0.9,\nwd 5e-4), constant lr, 10 epochs.  sparsity 0.5 (paper default, with_0.5.pt);\nlr per setting from the paper's interval [5e-4, 5e-2] (CIFAR-10: 0.013, README).
"""

import torch
import torch.nn.functional as F

HP = {'cifar10_random10': {'lr': 0.013, 'sparsity': 0.5, 'epochs': 10}, 'cifar100_random10': {'lr': 0.0014, 'sparsity': 0.5, 'epochs': 10}, 'cifar10_random50': {'lr': 0.013, 'sparsity': 0.5, 'epochs': 10}}

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
        self.mask = None
        return [{"set": "forget", "mode": "accumulate", "eval": True}] + self._rl_plan(ctx)

    def loss(self, logits, labels, is_forget, ids, info):
        if info.pass_index == 0:                  # saliency pass: gradient of -CE
            return -F.cross_entropy(logits, labels)
        return self._rl_loss(logits, labels, is_forget, ids, info)

    def end_pass(self, params, info):
        if info.pass_index != 0:
            return
        with torch.no_grad():
            flat = torch.cat([p.grad.abs().flatten() for p in params.values()])
            k = int(len(flat) * self.hp["sparsity"])
            keep = torch.zeros_like(flat)
            keep[torch.argsort(-flat)[:k]] = 1.0        # largest |g| first (generate_mask.py)
            self.mask, start = {}, 0
            for n, p in params.items():
                self.mask[n] = keep[start:start + p.numel()].view_as(p)
                start += p.numel()
            self.theta0 = {n: p.detach().clone() for n, p in params.items()}
        self.names = list(params)
        self.opt = torch.optim.SGD(list(params.values()), lr=self.hp["lr"], momentum=0.9,
                                   weight_decay=5e-4)

    def update(self, params, info):
        with torch.no_grad():
            for n, p in params.items():
                p.grad.mul_(self.mask[n])
        self.opt.step()
        with torch.no_grad():
            for n, p in params.items():
                m = self.mask[n]
                p.mul_(m).add_(self.theta0[n] * (1 - m))
                st = self.opt.state.get(p)
                if st is not None and "momentum_buffer" in st:
                    st["momentum_buffer"].mul_(m)
'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 1, "end_line": 51, "content": _CONTENT}
]
