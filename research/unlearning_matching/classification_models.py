"""Research model factories for the CIFAR-10 architecture comparison."""

import torch
from torch import nn
from torch.nn import functional as F


ARCHITECTURES = ("resnet18", "vit-tiny")


class EncoderBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 2 * width),
                                 nn.GELU(approximate="tanh"),
                                 nn.Linear(2 * width, width))

    def forward(self, x):
        batch, tokens, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(
            batch, tokens, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4).unbind(0)
        attn = F.scaled_dot_product_attention(q, k, v)
        x = x + self.projection(attn.transpose(1, 2).reshape(batch, tokens, width))
        return x + self.mlp(self.norm2(x))


class CifarVisionTransformer(nn.Module):
    patch_size = 4
    width = 192
    depth = 4
    heads = 3

    def __init__(self, normalize, classes):
        super().__init__()
        self.normalize = normalize
        self.patches = nn.Conv2d(3, self.width, self.patch_size, stride=self.patch_size)
        tokens = (32 // self.patch_size) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, self.width))
        self.position = nn.Parameter(torch.empty(1, tokens + 1, self.width))
        self.blocks = nn.Sequential(*(EncoderBlock(self.width, self.heads)
                                      for _ in range(self.depth)))
        self.norm = nn.LayerNorm(self.width)
        self.head = nn.Linear(self.width, classes)
        nn.init.trunc_normal_(self.position, std=0.02)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)

    def forward(self, images):
        patches = self.patches(self.normalize(images)).flatten(2).transpose(1, 2)
        dtype = patches.dtype
        tokens = torch.cat((self.cls.to(dtype).expand(len(images), -1, -1), patches), dim=1)
        tokens = self.blocks(tokens + self.position.to(dtype))
        return self.head(self.norm(tokens[:, 0]))


def build_model(common, architecture, dataset="cifar10"):
    if architecture == "resnet18":
        return common.build_model(dataset)
    if architecture == "vit-tiny" and dataset == "cifar10":
        dataset_spec = common.DATASETS[dataset]
        normalize = common.NormalizeByChannelMeanStd(dataset_spec["mean"], dataset_spec["std"])
        return CifarVisionTransformer(normalize, dataset_spec["num_classes"])
    raise ValueError(f"Unsupported architecture/dataset pair: {architecture}/{dataset}")
