"""
Make a Deep Net Converge in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - fashion_loaders
import os
import gzip
import tempfile
import urllib.request

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def fashion_loaders(n_train=5000, n_val=1000, batch_size=64, seed=42):
    base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    cache_dir = tempfile.gettempdir()

    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    paths = {}
    for key, filename in files.items():
        cached_name = f"fashion_{filename}"
        path = os.path.join(cache_dir, cached_name)

        if not os.path.exists(path):
            urllib.request.urlretrieve(base_url + filename, path)

        paths[key] = path

    # Parse training images.
    with gzip.open(paths["train_images"], "rb") as f:
        image_bytes = f.read()

    train_images = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
        offset=16
    ).reshape(-1, 28 * 28)

    # Parse training labels.
    with gzip.open(paths["train_labels"], "rb") as f:
        label_bytes = f.read()

    train_labels = np.frombuffer(
        label_bytes,
        dtype=np.uint8,
        offset=8
    )

    # Parse validation images from the test set.
    with gzip.open(paths["test_images"], "rb") as f:
        image_bytes = f.read()

    test_images = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
        offset=16
    ).reshape(-1, 28 * 28)

    # Parse validation labels from the test set.
    with gzip.open(paths["test_labels"], "rb") as f:
        label_bytes = f.read()

    test_labels = np.frombuffer(
        label_bytes,
        dtype=np.uint8,
        offset=8
    )

    # Training data: first n_train examples.
    x_train = train_images[:n_train]
    y_train = train_labels[:n_train]

    # Validation data: next n_val examples after the training portion.
    x_val = train_images[n_train:n_train + n_val]
    y_val = train_labels[n_train:n_train + n_val]

    # Standardize after scaling pixels to [0, 1].
    x_train = x_train.astype(np.float32) / 255.0
    x_val = x_val.astype(np.float32) / 255.0

    x_train = (x_train - 0.2860) / 0.3530
    x_val = (x_val - 0.2860) / 0.3530

    # Convert to PyTorch tensors.
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train.astype(np.int64))

    x_val = torch.from_numpy(x_val)
    y_val = torch.from_numpy(y_val.astype(np.int64))

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    # Seeded generator for reproducible training shuffling.
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "n_features": 784
    }

# Step 2 - DeepNet
class DeepNet(nn.Module):
    def __init__(
        self,
        n_features=784,
        n_hidden=100,
        n_layers=10,
        activation="sigmoid",
        batchnorm=False,
        dropout=0.0,
        n_classes=10
    ):
        super().__init__()

        if activation not in ("sigmoid", "relu"):
            raise ValueError("activation must be either 'sigmoid' or 'relu'")

        layers = []
        in_features = n_features

        for _ in range(n_layers):
            # Linear
            layers.append(nn.Linear(in_features, n_hidden))

            # Optional BatchNorm1d
            if batchnorm:
                layers.append(nn.BatchNorm1d(n_hidden))

            # Activation
            if activation == "sigmoid":
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

            # Optional Dropout
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            in_features = n_hidden

        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        x = self.body(x)
        return self.head(x)

# Step 3 - train_epochs (not yet solved)
# TODO: implement

# Step 4 - gradient_norms (not yet solved)
# TODO: implement

# Step 5 - apply_he_init (not yet solved)
# TODO: implement

# Step 6 - compare_configs (not yet solved)
# TODO: implement

# Step 7 - dropout_effect (not yet solved)
# TODO: implement

# Step 8 - make_optimizer (not yet solved)
# TODO: implement

# Step 9 - compare_optimizers (not yet solved)
# TODO: implement

# Step 10 - one_cycle (not yet solved)
# TODO: implement

# Step 11 - clipping_effect (not yet solved)
# TODO: implement

# Step 12 - split_by_class (not yet solved)
# TODO: implement

# Step 13 - transfer_head (not yet solved)
# TODO: implement

# Step 14 - transfer_experiment (not yet solved)
# TODO: implement

# Step 15 - save_deepnet (not yet solved)
# TODO: implement

