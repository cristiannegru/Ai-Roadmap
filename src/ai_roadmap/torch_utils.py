"""Phase 4 — PyTorch utilities: seeds, devices, MLP/CNN, training loop.

Maps to ``docs/deep-learning-roadmap.md > Phase 4`` and powers Milestone 4:

    "Train a custom CNN classifier using PyTorch."

Design goals:
- CPU-fast: synthetic digits train to >80% in <2 min on a laptop CPU.
- Deterministic given ``seed`` (for tests and reproducible notebooks).
- Size-agnostic CNN via ``AdaptiveAvgPool2d`` (works for 16×16 and 28×28).
- No downloads: :func:`make_synthetic_digits` replaces MNIST in tests/CI.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

__all__ = [
    "set_seed",
    "get_device",
    "SimpleMLP",
    "SimpleCNN",
    "make_synthetic_digits",
    "make_loaders",
    "train_classifier",
    "evaluate_accuracy",
    "count_parameters",
    "save_model",
    "load_model",
]


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "auto") -> torch.device:
    """Resolve a torch device: ``auto`` → cuda > mps > cpu.

    Args:
        prefer: ``"auto"`` | ``"cpu"`` | ``"cuda"`` | ``"mps"``.
    """
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if prefer == "mps":
        available = getattr(torch.backends, "mps", None) is not None
        use_mps = available and torch.backends.mps.is_available()
        return torch.device("mps" if use_mps else "cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SimpleMLP(nn.Module):
    """MLP baseline: Flatten → Linear+ReLU(+Dropout) ×N → Linear logits.

    Args:
        input_dim: flattened input size (e.g. 256 for 16×16 images).
        hidden_dims: hidden layer widths.
        num_classes: output logits.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: tuple[int, ...] = (128, 64),
        num_classes: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return unnormalized logits of shape ``(batch, num_classes)``."""
        return self.net(x)


class SimpleCNN(nn.Module):
    """Tiny CNN: Conv→ReLU→Pool ×2 → AdaptivePool → FC → logits.

    Works for any square input ≥ 8×8 (tested 16×16 and 28×28).
    ~25k params at default width — trains in seconds on CPU.
    """

    def __init__(self, num_classes: int = 4, in_channels: int = 1, width: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(width * 2 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return unnormalized logits of shape ``(batch, num_classes)``."""
        return self.classifier(self.features(x))


def make_synthetic_digits(
    n_per_class: int = 200,
    *,
    img_size: int = 16,
    num_classes: int = 4,
    noise: float = 0.25,
    seed: int = 42,
) -> TensorDataset:
    """Generate learnable quadrant-pattern images (no download, deterministic).

    Each class lights up one quadrant plus a class-specific bar:
      class 0 → top-left, 1 → top-right, 2 → bottom-left, 3 → bottom-right.
    Only the first 4 classes are supported (quadrant design); ``num_classes``
    must be in 1..4. Pixel values are clipped to [0, 1], shape ``(N,1,H,W)``.

    Args:
        n_per_class: samples per class.
        img_size: square image edge (≥8).
        noise: Gaussian noise std added to every pixel.
    """
    if not 1 <= num_classes <= 4:
        raise ValueError("synthetic quadrants support num_classes in 1..4")
    if img_size < 8:
        raise ValueError("img_size must be >= 8")
    rng = np.random.default_rng(seed)
    half = img_size // 2
    quadrants = [(0, 0), (0, 1), (1, 0), (1, 1)][:num_classes]
    images: list[np.ndarray] = []
    labels: list[int] = []
    for cls, (qr, qc) in enumerate(quadrants):
        for _ in range(n_per_class):
            img = np.zeros((img_size, img_size), dtype=np.float32)
            img[qr * half : (qr + 1) * half, qc * half : (qc + 1) * half] = 1.0
            # Class bar: a bright row whose position encodes the class.
            img[cls * (img_size // max(num_classes, 1)) % img_size, :] = 0.8
            img += rng.normal(0, noise, size=img.shape).astype(np.float32)
            images.append(np.clip(img, 0, 1))
            labels.append(cls)
    x = torch.from_numpy(np.stack(images)[:, None, :, :])
    y = torch.tensor(labels, dtype=torch.long)
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed))
    return TensorDataset(x[perm], y[perm])


def make_loaders(
    dataset: TensorDataset,
    *,
    batch_size: int = 64,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Split a dataset into deterministic train/val DataLoaders."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    n_val = max(1, int(len(dataset) * val_fraction))
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val], generator=gen)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen),
        DataLoader(val_ds, batch_size=batch_size),
    )


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Mean accuracy of ``model`` over ``loader`` (no grad).

    Moves the model to ``device`` first, so a checkpoint loaded on CPU
    still evaluates correctly on cuda/mps.
    """
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(dim=1)
            correct += int((pred == yb).sum())
            total += len(yb)
    return correct / total if total else 0.0


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    *,
    epochs: int = 5,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> dict[str, list[float]]:
    """Train with Adam + CrossEntropyLoss. Returns history dict.

    History keys: ``train_loss`` (per epoch), ``train_acc``, and ``val_acc``
    when ``val_loader`` is given. Runs fully on ``device`` (default auto).
    """
    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    device = device or get_device()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history: dict[str, list[float]] = {"train_loss": [], "train_acc": []}
    if val_loader is not None:
        history["val_acc"] = []

    for _ in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach()) * len(yb)
            correct += int(logits.argmax(dim=1).eq(yb).sum())
            total += len(yb)
        history["train_loss"].append(total_loss / max(total, 1))
        history["train_acc"].append(correct / max(total, 1))
        if val_loader is not None:
            history["val_acc"].append(evaluate_accuracy(model, val_loader, device))
    return history


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model: nn.Module, path: str | Path) -> Path:
    """Save ``state_dict`` (creating parent dirs). Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)
    return p


def load_model(
    model: nn.Module, path: str | Path, *, device: torch.device | None = None
) -> nn.Module:
    """Load ``state_dict`` into ``model`` (same architecture). Returns the model.

    The model is moved to ``device`` (default CPU) after loading, so
    callers can immediately evaluate on cuda/mps without a manual ``.to()``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    target = device or torch.device("cpu")
    state = torch.load(p, map_location=target, weights_only=True)
    model.load_state_dict(state)
    model.to(target)
    return model
