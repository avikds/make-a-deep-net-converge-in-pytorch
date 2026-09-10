"""
Make a Deep Net Converge in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - fashion_loaders
import os
import tempfile
import urllib.request
import gzip
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def fashion_loaders(n_train=5000, n_val=1000, batch_size=64, seed=42):
    base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    tmp_dir = tempfile.gettempdir()

    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    paths = {}

    # Download each file once and cache it with a fashion_ prefix.
    for key, filename in files.items():
        path = os.path.join(tmp_dir, "fashion_" + filename)
        paths[key] = path

        if not os.path.exists(path):
            urllib.request.urlretrieve(base_url + filename, path)

    # Read and parse the IDX files.
    with gzip.open(paths["train_images"], "rb") as f:
        train_images_raw = f.read()

    with gzip.open(paths["train_labels"], "rb") as f:
        train_labels_raw = f.read()

    with gzip.open(paths["test_images"], "rb") as f:
        test_images_raw = f.read()

    with gzip.open(paths["test_labels"], "rb") as f:
        test_labels_raw = f.read()

    train_images = np.frombuffer(
        train_images_raw, dtype=np.uint8, offset=16
    ).reshape(-1, 28 * 28)

    train_labels = np.frombuffer(
        train_labels_raw, dtype=np.uint8, offset=8
    )

    test_images = np.frombuffer(
        test_images_raw, dtype=np.uint8, offset=16
    ).reshape(-1, 28 * 28)

    test_labels = np.frombuffer(
        test_labels_raw, dtype=np.uint8, offset=8
    )

    # Use the first n_train training examples and the next n_val examples
    # for validation.
    val_start = n_train
    val_end = n_train + n_val

    x_train = train_images[:n_train]
    y_train = train_labels[:n_train]

    x_val = train_images[val_start:val_end]
    y_val = train_labels[val_start:val_end]

    # Scale pixels to [0, 1], then standardize using the specified
    # Fashion-MNIST mean and standard deviation.
    mean = 0.2860
    std = 0.3530

    x_train = x_train.astype(np.float32) / 255.0
    x_val = x_val.astype(np.float32) / 255.0

    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    # Convert to PyTorch tensors.
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train.astype(np.int64))

    x_val = torch.from_numpy(x_val)
    y_val = torch.from_numpy(y_val.astype(np.int64))

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    # Seeded generator for deterministic training-data shuffling.
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "n_features": 784,
    }

# Step 2 - DeepNet
import torch.nn as nn

class DeepNet(nn.Module):
    def __init__(
        self,
        n_features=784,
        n_hidden=100,
        n_layers=10,
        activation="sigmoid",
        batchnorm=False,
        dropout=0.0,
        n_classes=10,
    ):
        super().__init__()

        if activation not in ("sigmoid", "relu"):
            raise ValueError("activation must be 'sigmoid' or 'relu'")

        layers = []

        for i in range(n_layers):
            in_features = n_features if i == 0 else n_hidden

            layers.append(nn.Linear(in_features, n_hidden))

            if batchnorm:
                layers.append(nn.BatchNorm1d(n_hidden))

            if activation == "sigmoid":
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        x = self.body(x)
        return self.head(x)

# Step 3 - train_epochs
def train_epochs(model, loaders, optimizer, epochs=2, scheduler=None, clip=None):
    history = {
        "train_loss": [],
        "val_acc": [],
    }

    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for xb, yb in loaders["train"]:
            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()

            if clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip
                )

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            batch_size = xb.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        mean_train_loss = total_loss / total_samples

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in loaders["val"]:
                logits = model(xb)
                predictions = logits.argmax(dim=1)

                correct += (predictions == yb).sum().item()
                total += yb.size(0)

        val_accuracy = correct / total

        history["train_loss"].append(mean_train_loss)
        history["val_acc"].append(val_accuracy)

    return history

# Step 4 - gradient_norms
def gradient_norms(model, xb, yb):
    model.zero_grad()

    criterion = nn.CrossEntropyLoss()

    logits = model(xb)
    loss = criterion(logits, yb)

    loss.backward()

    norms = []

    for layer in model.body:
        if isinstance(layer, nn.Linear):
            if layer.weight.grad is None:
                norms.append(0.0)
            else:
                norms.append(layer.weight.grad.norm(p=2).item())

    return norms

# Step 5 - apply_he_init
def apply_he_init(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(
                layer.weight,
                nonlinearity="relu"
            )
            nn.init.zeros_(layer.bias)

    return model

# Step 6 - compare_configs
def compare_configs(loaders, configs, epochs=2, lr=0.1, seed=42):
    results = {}

    for name, config in configs.items():
        torch.manual_seed(seed)

        model_config = dict(config)
        he_init = model_config.pop("he_init", False)

        model = DeepNet(**model_config)

        if he_init:
            apply_he_init(model)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr
        )

        history = train_epochs(
            model,
            loaders,
            optimizer,
            epochs=epochs
        )

        results[name] = float(history["val_acc"][-1])

    return results

# Step 7 - dropout_effect
def dropout_effect(loaders, rate=0.5, epochs=3, seed=42):
    results = {}

    for name, dropout_rate in (
        ("no_dropout", 0.0),
        ("dropout", rate),
    ):
        torch.manual_seed(seed)

        model = DeepNet(
            activation="relu",
            batchnorm=True,
            dropout=dropout_rate,
            n_layers=3,
        )

        apply_he_init(model)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.1,
        )

        train_epochs(
            model,
            loaders,
            optimizer,
            epochs=epochs,
        )

        # Measure training accuracy with the model in training mode.
        # This keeps dropout active, showing its regularizing effect
        # on the training fit.
        model.train()

        train_correct = 0
        train_total = 0

        with torch.no_grad():
            for xb, yb in loaders["train"]:
                logits = model(xb)
                predictions = logits.argmax(dim=1)

                train_correct += (predictions == yb).sum().item()
                train_total += yb.size(0)

        train_acc = train_correct / train_total

        # Validation accuracy must be measured with dropout disabled.
        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in loaders["val"]:
                logits = model(xb)
                predictions = logits.argmax(dim=1)

                val_correct += (predictions == yb).sum().item()
                val_total += yb.size(0)

        val_acc = val_correct / val_total

        results[name] = {
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
        }

    return results

# Step 8 - make_optimizer
def make_optimizer(model, name, lr=0.01):
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
        )

    if name == "momentum":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
        )

    if name == "nesterov":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            nesterov=True,
        )

    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
        )

    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.01,
        )

    raise ValueError(f"Unknown optimizer: {name}")

# Step 9 - compare_optimizers
def compare_optimizers(loaders, names, epochs=2, lr=0.01, seed=42):
    results = {}

    for name in names:
        torch.manual_seed(seed)

        model = DeepNet(
            activation="relu",
            batchnorm=True,
            n_layers=3,
        )

        apply_he_init(model)

        optimizer = make_optimizer(
            model,
            name,
            lr=lr,
        )

        history = train_epochs(
            model,
            loaders,
            optimizer,
            epochs=epochs,
        )

        results[name] = float(history["val_acc"][-1])

    return results

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

