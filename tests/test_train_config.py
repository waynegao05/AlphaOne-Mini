from __future__ import annotations


def test_load_yaml_config_and_cli_override(tmp_path):
    from train.config import load_config, merge_overrides, save_resolved_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "experiment_name: demo\n"
        "device: cuda\n"
        "allow_cpu_fallback: false\n"
        "model_type: advanced\n"
        "batch_size: 32\n"
        "loss_weights:\n"
        "  threat: 0.3\n",
        encoding="utf-8",
    )

    config = load_config(str(config_path))
    merged = merge_overrides(config, {"batch_size": 8, "allow_cpu_fallback": True, "missing": None})
    out_path = tmp_path / "resolved.yaml"
    save_resolved_config(merged, str(out_path))

    assert config["experiment_name"] == "demo"
    assert merged["batch_size"] == 8
    assert merged["allow_cpu_fallback"] is True
    assert merged["loss_weights"]["threat"] == 0.3
    assert out_path.exists()


def test_default_configs_exist_and_have_required_keys():
    from train.config import REQUIRED_CONFIG_KEYS, load_config

    for path in (
        "configs/deep_train_advanced_cuda.yaml",
        "configs/deep_train_resnet_cuda.yaml",
        "configs/deep_train_cnn_cuda.yaml",
        "configs/smoke_test_cpu.yaml",
        "configs/train_stage0_smoke.yaml",
        "configs/train_stage1_tactical_pretrain.yaml",
        "configs/train_stage2_selfplay_finetune.yaml",
        "configs/train_stage3_benchmark.yaml",
        "configs/train_stage_full_cuda.yaml",
    ):
        config = load_config(path)
        missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
        assert not missing, f"{path} missing {missing}"
