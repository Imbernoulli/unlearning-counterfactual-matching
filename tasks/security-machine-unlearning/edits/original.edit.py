"""Baseline `original` for security-machine-unlearning: replaces the whole editable file.

Generated from the ported method source.
"""

_FILE = "unlearn-lab/unlearn/custom_unlearning.py"

_CONTENT = r'''"""Declared null: no unlearning (the Original model is returned unchanged).

An empty plan: the harness runs no pass.
"""

import torch
import torch.nn.functional as F

HP = {'cifar10_random10': {}, 'cifar100_random10': {}, 'cifar10_random50': {}}

class UnlearningMethod:
    def plan(self, ctx):
        return []

    def loss(self, logits, labels, is_forget, ids, info):
        return logits.sum() * 0.0

    def update(self, params, info):
        pass
'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 1, "end_line": 51, "content": _CONTENT}
]
