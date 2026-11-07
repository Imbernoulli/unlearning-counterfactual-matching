"""Fixed components for security-machine-unlearning (READ-ONLY).

Port of the image-classification unlearning protocol of SalUn
(Fan et al., ICLR 2024, arXiv:2310.12508; code: OPTML-Group/Unlearn-Saliency
@a6e89fd, Classification/), which inherits it from l1-sparse unlearning
(Jia et al., NeurIPS 2023, arXiv:2304.04934; OPTML-Group/Unlearn-Sparse):

* the fixed network (3x3-conv stem, no max-pool) with the input normalisation
  inside the network, so every image tensor is in [0, 1]
  (Classification/models/ResNet.py:7-28, 180-325).
* 10% of each class of the official training split is held out as a
  validation set; the remaining images are the training set D of the
  original model (Classification/dataset.py:576-593).
* Random data forgetting: a uniformly random fraction of D (SalUn Table 1's
  scenarios) is the forget set D_f, the rest is the retain set D_r
  (dataset.py:674-705, replace_class(-1)).  The fraction is fixed per setting
  by the trusted run script.  In a scored run the forget set is drawn from
  FRESH entropy by the trusted parent (run_unlearning.py), so it cannot be
  known before the run; the Retrain reference is the mean over Retrain models
  of several random splits (a random Retrain varies by < 1 point across splits,
  SalUn Table 1 std).
* Augmentation: RandomCrop(32, padding=4) + RandomHorizontalFlip for the
  CIFAR datasets, none for SVHN (dataset.py:85-95, 549-555).

Everything here is shared by the trusted parent, the sandboxed worker and the
separate evaluation process.
"""

import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_ROOT = os.environ.get("UNLEARN_DATA", "/opt/unlearn-lab/data")
CKPT_ROOT = os.environ.get("UNLEARN_CKPT", "/opt/unlearn-lab/original")

# Per-dataset constants (SalUn Classification/utils.py:113-198 normalisation,
# dataset.py augmentation).
DATASETS = {
    "cifar10": dict(num_classes=10, augment=True,
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    "cifar100": dict(num_classes=100, augment=True,
                     mean=(0.5071, 0.4866, 0.4409), std=(0.2673, 0.2564, 0.2762)),
    "svhn": dict(num_classes=10, augment=False,
                 mean=(0.4377, 0.4438, 0.4728), std=(0.1201, 0.1231, 0.1052)),
}

VALID_FRACTION = 0.10   # per class, held out of the official train split
# Random data forgetting scenarios (SalUn Table 1); a setting
# picks one through run_unlearning.py / evaluate.py --forget-fraction.
FORGET_FRACTIONS = (0.10, 0.50)
SPLIT_SEED = 1          # validation split (fixed: defines D, the Original's training set)
BATCH_SIZE = 256        # SalUn arg_parser.py default


# ---------------------------------------------------------------------------
# Model: ResNet-18 exactly as in Classification/models/ResNet.py
# ---------------------------------------------------------------------------

class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x):
        return x.sub(self.mean[None, :, None, None]).div(self.std[None, :, None, None])


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet18(nn.Module):
    def __init__(self, num_classes, mean, std):
        super().__init__()
        self.normalize = NormalizeByChannelMeanStd(mean, std)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.normalize(x)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


def build_model(dataset):
    spec = DATASETS[dataset]
    return ResNet18(spec["num_classes"], spec["mean"], spec["std"])


def original_checkpoint_path(dataset):
    return os.path.join(CKPT_ROOT, f"{dataset}_resnet18_original.pt")


# ---------------------------------------------------------------------------
# Data and splits
# ---------------------------------------------------------------------------

# Under Harbor the test split is withheld from the agent-visible image: the
# image's <dataset>.npz holds only x_train / y_train, and the test split lives
# verifier-side only (holdout/security-machine-unlearning/data/withheld/, staged
# to the bundle's tests/meta/data/withheld/, mounted only at verification and
# read only by the root evaluator through its task directory).  Natively the
# npz still carries all four arrays and nothing here changes.
WITHHELD_TEST_FILES = {"cifar10": "cifar10_test.npz", "cifar100": "cifar100_test.npz"}


def load_train(dataset):
    """uint8 train images (N, 32, 32, 3) and int64 labels (all the unlearning phase needs)."""
    with np.load(os.path.join(DATA_ROOT, f"{dataset}.npz"), allow_pickle=False) as z:
        return z["x_train"], z["y_train"].astype(np.int64)


def load_arrays(dataset, test_dir=None):
    """uint8 images (N, 32, 32, 3) and int64 labels for train and test.

    The test split comes from the image's npz when it carries one (native),
    otherwise from ``test_dir``/WITHHELD_TEST_FILES[dataset] (Harbor verifier).
    """
    with np.load(os.path.join(DATA_ROOT, f"{dataset}.npz"), allow_pickle=False) as z:
        x_tr, y_tr = z["x_train"], z["y_train"].astype(np.int64)
        if "x_test" in z.files:
            return x_tr, y_tr, z["x_test"], z["y_test"].astype(np.int64)
    if test_dir is None or dataset not in WITHHELD_TEST_FILES:
        raise RuntimeError(f"the {dataset} test split is withheld from this image "
                           "(it is mounted only for the final evaluation)")
    with np.load(os.path.join(str(test_dir), WITHHELD_TEST_FILES[dataset]),
                 allow_pickle=False) as t:
        return x_tr, y_tr, t["x_test"], t["y_test"].astype(np.int64)


def setting_name(dataset, forget_fraction):
    """'cifar10_random10', 'cifar10_random50', ... (the key methods see as ctx.setting)."""
    return f"{dataset}_random{int(round(100 * float(forget_fraction)))}"


def check_forget_fraction(forget_fraction):
    f = float(forget_fraction)
    if not any(abs(f - g) < 1e-9 for g in FORGET_FRACTIONS):
        raise ValueError(f"forget fraction {forget_fraction} is not one of {FORGET_FRACTIONS}")
    return f


def split_indices(y_train, num_classes, forget_seed, forget_fraction=0.10):
    """Return (train_idx, valid_idx, forget_idx, retain_idx) into the official train split.

    train_idx is D (what the original model is trained on); forget_idx and
    retain_idx partition it, |forget_idx| = round(forget_fraction * |D|).  All
    index arrays are sorted ascending.
    """
    forget_fraction = check_forget_fraction(forget_fraction)
    rng = np.random.RandomState(SPLIT_SEED)
    valid = []
    for c in range(num_classes):
        cls = np.where(y_train == c)[0]
        valid.append(rng.choice(cls, int(VALID_FRACTION * len(cls)), replace=False))
    valid_idx = np.sort(np.hstack(valid))
    keep = np.ones(len(y_train), dtype=bool)
    keep[valid_idx] = False
    train_idx = np.flatnonzero(keep)
    n_forget = int(round(forget_fraction * len(train_idx)))
    rng_f = np.random.RandomState(int(forget_seed) % (2 ** 32))
    pos = rng_f.choice(len(train_idx), size=n_forget, replace=False)
    is_forget = np.zeros(len(train_idx), dtype=bool)
    is_forget[pos] = True
    return train_idx, valid_idx, train_idx[is_forget], train_idx[~is_forget]


RETAIN, FORGET = 0, 1

# SalUn's per-method budgets (Classification/arg_parser.py `--unlearn_epochs`
# default 10; README commands `--unlearn_epochs 10` for SalUn/RL, FT and
# l1-sparse, GA uses 5): at most 10 passes over the retain set; the forget set
# gets 10 passes (RL / SalUn visit D_f once per epoch) plus the single saliency
# pass of generate_mask.py.  A pass over the union counts once for each set.
RETAIN_PASSES = 10
FORGET_PASSES = 11
MAX_CUSTOM_BYTES = 64 * 1024   # size cap on the editable file (no embedded weights)

# sha256 of the baked Original checkpoints (file bytes) and of the decoded
# dataset arrays (x_train, y_train, x_test, y_test bytes in that order; the
# npz container itself is not byte-stable across builds).  Checked by the
# trusted parent and by the evaluator.
SHA256_ORIGINAL = {
    "cifar10": "a136ae6dcb21ddcf4fa2adede89c2d7f3dd22c00217c1913ed3929aef95d9e58",
    "cifar100": "1a87165d82d3831ff6111dfbbe389999bc41048b0e4c157d54ea39b52918f054",
    "svhn": "5c3761d04a276adde63e6e086205ea5df2dc37420123e9c6d36a0ddc9ce9db17",
}
SHA256_DATA = {
    "cifar10": "0760cfcaf7f8c679d9086f18f9aac9dc8a04de5349016f8d3e975cc3164229ff",
    "cifar100": "ce046832bab115ddc5dec09ddaeb8339c7a9fc00cb9d3f6aebdfd619f125d129",
    "svhn": "6e610f74ea7556886370c4de823226b6e3c458ee4efd15cdc7e5c32ae1731294",
}
# sha256 of (x_train, y_train) alone: what the unlearning phase checks (it never
# reads the test split, which the Harbor agent image does not carry).
SHA256_TRAIN = {
    "cifar10": "af4f586337eda0c26f82c4fd1864403d1ee0385e2ea10df41f8d40c8a70c8b89",
    "cifar100": "f062ca493c4d44a3d0921f223fbc3b1e88b46fd03322e97c635a0ac2a5f8baab",
    "svhn": "6c105b16b8fbeed59fec6a93f466e84bc38cbabdfbc5e198fc6ec9449bf282c4",
}


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_arrays(*arrays):
    import hashlib
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def load_verified(dataset, test_dir=None, with_test=True):
    """load_arrays + Original state_dict, both checked against the pinned sha256.

    ``with_test=False`` (the unlearning phase) loads and checks only the train
    split and returns ``(x_train, y_train, None, None)``.
    """
    if with_test:
        arrays = load_arrays(dataset, test_dir)
        got = sha256_arrays(*arrays)
        if got != SHA256_DATA[dataset]:
            raise RuntimeError(f"dataset {dataset} does not match its pinned sha256 ({got})")
    else:
        x_tr, y_tr = load_train(dataset)
        got = sha256_arrays(x_tr, y_tr)
        if got != SHA256_TRAIN[dataset]:
            raise RuntimeError(f"dataset {dataset} train split does not match its pinned "
                               f"sha256 ({got})")
        arrays = (x_tr, y_tr, None, None)
    path = original_checkpoint_path(dataset)
    got = sha256_file(path)
    if got != SHA256_ORIGINAL[dataset]:
        raise RuntimeError(f"Original checkpoint {path} does not match its pinned sha256 ({got})")
    import torch as _t
    state = _t.load(path, map_location="cpu", weights_only=True)
    return arrays, state


def gpu_augment(x_u8, augment):
    """uint8 (B, 3, 32, 32) -> float in [0, 1]; RandomCrop(32, 4) + HFlip when augment."""
    x = x_u8.float().div_(255.0)
    if not augment:
        return x
    b = x.shape[0]
    dev = x.device
    padded = F.pad(x, (4, 4, 4, 4))
    oy = torch.randint(0, 9, (b,), device=dev)
    ox = torch.randint(0, 9, (b,), device=dev)
    ar = torch.arange(32, device=dev)
    rows = (oy[:, None] + ar[None, :])[:, None, :, None].expand(b, 3, 32, 32)
    cols = (ox[:, None] + ar[None, :])[:, None, None, :].expand(b, 3, 32, 32)
    x = padded[torch.arange(b, device=dev)[:, None, None, None],
               torch.arange(3, device=dev)[None, :, None, None], rows, cols]
    flip = torch.rand(b, device=dev) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(3), x)


def to_gpu_u8(x_nhwc, device):
    return torch.from_numpy(np.ascontiguousarray(x_nhwc.transpose(0, 3, 1, 2))).to(device)


def set_determinism(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


PARAM_KEYS_EXCLUDE = ("normalize.",)


@torch.no_grad()
def descent_cosine(state, original_state):
    """Global cosine between the unlearned and the Original conv / linear weights.

    Concatenates the weight tensors of every Conv2d and Linear layer (11.2M of
    the 11.2M parameters).  BN affine parameters are left out on purpose: a
    re-initialised BN weight is all ones and the Original's are all positive,
    so they would add a large spurious positive dot product (measured: full
    re-initialisation scored 0.26 with them, ~0.00 without).  An update of the
    Original keeps this cosine far above 0 (every SalUn-family baseline,
    measured); a re-initialised network is independent of the Original.
    """
    model = ResNet18(int(original_state["fc.weight"].shape[0]), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    names = [n + ".weight" for n, m in model.named_modules()
             if isinstance(m, (nn.Conv2d, nn.Linear))]
    dot = na = nb = 0.0
    for n in names:
        a = state[n].double().flatten()
        b = original_state[n].double().flatten()
        dot += float(torch.dot(a, b))
        na += float(torch.dot(a, a))
        nb += float(torch.dot(b, b))
    cos = dot / (math.sqrt(na * nb) + 1e-12)
    return cos if math.isfinite(cos) else -1.0


# The unlearned model must remain an update of the Original: global conv/linear
# weight cosine to it >= this. Full re-initialisation measures 0.000; the lowest
# honest baseline measures 0.27 (l1_sparse, SVHN).
MIN_DESCENT_COSINE = 0.1


@torch.no_grad()
def predict_probs(model, images_u8, batch_size=1024):
    model.eval()
    out = []
    for s in range(0, images_u8.shape[0], batch_size):
        out.append(F.softmax(model(images_u8[s:s + batch_size].float().div_(255.0)), dim=1))
    return torch.cat(out)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self):
        return time.time() - self.t0
