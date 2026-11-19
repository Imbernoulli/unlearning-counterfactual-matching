"""Build the verified CIFAR-10 NPZ from the Hugging Face mirror's Parquet files."""

import argparse
import io
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq

from run_experiments import common


def decode(path):
    parquet = pq.ParquetFile(path)
    images = np.empty((parquet.metadata.num_rows, 32, 32, 3), dtype=np.uint8)
    labels = np.empty(len(images), dtype=np.int64)
    offset = 0
    for batch in parquet.iter_batches(batch_size=1024, columns=["img", "label"]):
        for image, label in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            with Image.open(io.BytesIO(image["bytes"])) as png:
                images[offset] = np.asarray(png.convert("RGB"))
            labels[offset] = label
            offset += 1
    return images, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    x_train, y_train = decode(args.train)
    x_test, y_test = decode(args.test)
    train_hash = common.sha256_arrays(x_train, y_train)
    full_hash = common.sha256_arrays(x_train, y_train, x_test, y_test)
    if train_hash != common.SHA256_TRAIN["cifar10"] or full_hash != common.SHA256_DATA["cifar10"]:
        raise ValueError(f"CIFAR-10 ordering/content mismatch: train={train_hash}, full={full_hash}")
    temp = args.out.with_suffix(".tmp")
    with temp.open("wb") as handle:
        np.savez(handle, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    temp.replace(args.out)
    print(f"{args.out}: train={train_hash}, full={full_hash}", flush=True)


if __name__ == "__main__":
    main()
