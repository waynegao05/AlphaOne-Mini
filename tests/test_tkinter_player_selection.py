from __future__ import annotations

from pathlib import Path

from game.board import BLACK, WHITE


def test_create_player_human_returns_human_slot():
    from ui.player_factory import create_player

    slot = create_player("human", BLACK, rule_mode="basic", device="cpu")

    assert slot.is_human is True
    assert slot.player is None
    assert slot.color == BLACK


def test_visible_player_choices_are_simplified():
    from ui.player_factory import PLAYER_TYPE_LABELS

    assert list(PLAYER_TYPE_LABELS.values()) == [
        "Human",
        "AlphaOne-Mini",
        "External AI.py",
    ]


def test_create_player_strong_has_select_action():
    from engine.strong_player import StrongPlayer
    from ui.player_factory import create_player

    slot = create_player("alphaone_mini", BLACK, rule_mode="basic", device="cpu", num_simulations=1)

    assert isinstance(slot.player, StrongPlayer)
    assert hasattr(slot.player, "select_action")


def test_legacy_player_names_map_to_alphaone_mini():
    from ui.player_factory import normalize_player_type

    assert normalize_player_type("StrongPlayer") == "alphaone_mini"
    assert normalize_player_type("NeuralGuarded no_hybrid_fallback") == "alphaone_mini"
    assert normalize_player_type("TacticalPlayer") == "alphaone_mini"
    assert normalize_player_type("HybridPlayer") == "alphaone_mini"
    assert normalize_player_type("Latest Advanced") == "alphaone_mini"


def test_create_player_external_ai(tmp_path: Path):
    from engine.external_ai_adapter import ExternalAIAdapter
    from ui.player_factory import create_player

    ai_path = tmp_path / "AI.py"
    ai_path.write_text("def select_action(board):\n    return 112\n", encoding="utf-8")

    slot = create_player(
        "external_ai",
        BLACK,
        rule_mode="basic",
        device="cpu",
        external_ai_path=str(ai_path),
    )

    assert isinstance(slot.player, ExternalAIAdapter)
    assert slot.is_external is True
