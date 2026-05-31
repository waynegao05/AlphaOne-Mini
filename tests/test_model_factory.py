from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_create_model_supports_cnn_resnet_advanced():
    from model.advanced_policy_value_net import AdvancedPolicyValueNet
    from model.model_factory import create_model
    from model.policy_value_net import PolicyValueNet
    from model.resnet_policy_value_net import SmallResNetPolicyValueNet

    assert isinstance(create_model("cnn"), PolicyValueNet)
    assert isinstance(create_model("resnet", blocks=1, channels=16), SmallResNetPolicyValueNet)
    assert isinstance(create_model("advanced", blocks=1, channels=16), AdvancedPolicyValueNet)
    assert isinstance(create_model(), AdvancedPolicyValueNet)


def test_create_model_rejects_unknown_type():
    from model.model_factory import create_model

    with pytest.raises(ValueError, match="model_type"):
        create_model("transformer")


def test_create_model_from_metadata_uses_model_type():
    from model.model_factory import create_model_from_metadata
    from model.resnet_policy_value_net import SmallResNetPolicyValueNet

    model = create_model_from_metadata({"model_type": "resnet", "model_kwargs": {"blocks": 1, "channels": 16}})

    assert isinstance(model, SmallResNetPolicyValueNet)
