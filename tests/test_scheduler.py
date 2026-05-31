from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_create_constant_scheduler_keeps_lr():
    from train.scheduler import create_scheduler

    param = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([param], lr=0.1)
    scheduler = create_scheduler(optimizer, "constant", total_epochs=3)
    lrs = []
    for _ in range(3):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    assert lrs == [0.1, 0.1, 0.1]


def test_create_cosine_and_step_schedulers_change_lr():
    from train.scheduler import create_scheduler

    param = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([param], lr=0.1)
    cosine = create_scheduler(optimizer, "cosine", total_epochs=4)
    optimizer.step(); cosine.step()
    assert optimizer.param_groups[0]["lr"] <= 0.1

    optimizer = torch.optim.SGD([param], lr=0.1)
    step = create_scheduler(optimizer, "step", total_epochs=4, step_size=1, gamma=0.5)
    optimizer.step(); step.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


def test_unknown_scheduler_raises():
    from train.scheduler import create_scheduler

    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=0.1)
    with pytest.raises(ValueError, match="scheduler"):
        create_scheduler(optimizer, "unknown", total_epochs=1)
