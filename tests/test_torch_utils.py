"""Tests for torch_utils.py (Phase 4, CPU-only, no downloads)."""

import torch

from ai_roadmap.torch_utils import (
    SimpleCNN,
    SimpleMLP,
    count_parameters,
    evaluate_accuracy,
    get_device,
    load_model,
    make_loaders,
    make_synthetic_digits,
    save_model,
    set_seed,
    train_classifier,
)


def test_seed_deterministic_synthetic():
    a = make_synthetic_digits(20, seed=7)
    b = make_synthetic_digits(20, seed=7)
    assert torch.equal(a.tensors[0], b.tensors[0])


def test_synthetic_shape_and_labels():
    ds = make_synthetic_digits(25, img_size=16, num_classes=4, seed=0)
    assert len(ds) == 100
    x, y = ds[0]
    assert x.shape == (1, 16, 16)
    assert set(ds.tensors[1].tolist()) == {0, 1, 2, 3}


def test_device_cpu_request():
    assert get_device("cpu").type == "cpu"
    assert get_device("auto").type in {"cpu", "cuda", "mps"}


def test_forward_shapes():
    mlp = SimpleMLP(input_dim=256, num_classes=4)
    assert mlp(torch.randn(8, 1, 16, 16)).shape == (8, 4)
    cnn = SimpleCNN(num_classes=4)
    assert cnn(torch.randn(8, 1, 16, 16)).shape == (8, 4)
    assert cnn(torch.randn(2, 1, 28, 28)).shape == (2, 4)  # size-agnostic
    assert count_parameters(cnn) > 1000


def test_training_loss_decreases_and_acc_rises():
    set_seed(0)
    ds = make_synthetic_digits(60, img_size=16, num_classes=4, seed=0)
    train_loader, val_loader = make_loaders(ds, batch_size=32, seed=0)
    model = SimpleCNN(num_classes=4)
    hist = train_classifier(
        model, train_loader, val_loader, epochs=5, lr=5e-3, device=torch.device("cpu")
    )
    assert hist["train_loss"][-1] < hist["train_loss"][0]
    assert hist["val_acc"][-1] > 0.7
    assert evaluate_accuracy(model, val_loader, torch.device("cpu")) > 0.7


def test_save_load_roundtrip(tmp_path):
    set_seed(1)
    model = SimpleMLP(input_dim=256, num_classes=4)
    path = save_model(model, tmp_path / "mlp.pt")
    fresh = SimpleMLP(input_dim=256, num_classes=4)
    load_model(fresh, path)
    for p1, p2 in zip(model.parameters(), fresh.parameters(), strict=True):
        assert torch.equal(p1, p2)
    # Device placement: loaded model must live on the requested device so
    # evaluate_accuracy() never hits CPU-weight vs MPS-input mismatch.
    assert next(fresh.parameters()).device.type == "cpu"
    ds = make_synthetic_digits(8, img_size=16, num_classes=4, seed=3)
    loader, _ = make_loaders(ds, batch_size=8, seed=3)
    assert 0.0 <= evaluate_accuracy(fresh, loader, torch.device("cpu")) <= 1.0
