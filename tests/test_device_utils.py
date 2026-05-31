from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_get_device_returns_cpu_when_requested():
    from utils.device import describe_device, get_device

    device = get_device("cpu")

    assert device.type == "cpu"
    assert "cpu" in describe_device(device).lower()


def test_get_device_cuda_fallback_is_explicit(monkeypatch):
    from utils.device import get_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        get_device("cuda", allow_cpu_fallback=False)

    assert get_device("cuda", allow_cpu_fallback=True).type == "cpu"


def test_move_batch_to_device_handles_tensor_dict_and_tuple():
    from utils.device import move_batch_to_device

    device = torch.device("cpu")
    batch = {
        "x": torch.zeros(1),
        "nested": (torch.ones(1), {"y": torch.full((1,), 2.0)}),
    }
    moved = move_batch_to_device(batch, device)

    assert moved["x"].device.type == "cpu"
    assert moved["nested"][0].device.type == "cpu"
    assert moved["nested"][1]["y"].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_assert_cuda_available_returns_cuda_when_present():
    from utils.device import assert_cuda_available

    assert assert_cuda_available().type == "cuda"
