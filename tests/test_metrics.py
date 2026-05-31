"""evaluate/metrics.py 的单元测试。

只用手工构造的 :class:`GameResult` 对象，不依赖 PyTorch / 真实模型。
"""

from __future__ import annotations

import pytest

from evaluate.arena import GameResult
from evaluate.metrics import (
    compute_draw_rate,
    compute_win_rate,
    format_match_summary,
    should_promote,
    summarize_results,
)
from game.board import BLACK, WHITE


def _gr(winner: int, black: str, white: str, num_moves: int = 10, reason: str = "draw") -> GameResult:
    if reason == "draw":
        if winner == BLACK:
            reason = "black_win"
        elif winner == WHITE:
            reason = "white_win"
    return GameResult(
        winner=winner,
        black_player_name=black,
        white_player_name=white,
        num_moves=num_moves,
        moves=[],
        reason=reason,
    )


# ---------------------------------------------------------------------------
# summarize_results
# ---------------------------------------------------------------------------
class TestSummarize:
    def test_total_and_color_counts(self):
        results = [
            _gr(BLACK, "A", "B"),
            _gr(WHITE, "A", "B"),
            _gr(0, "A", "B", reason="draw"),
        ]
        s = summarize_results(results, player_a_name="A", player_b_name="B")
        assert s["total_games"] == 3
        assert s["black_wins"] == 1
        assert s["white_wins"] == 1
        assert s["draws"] == 1

    def test_player_wins_across_seats(self):
        # A 执黑赢 1 盘 + A 执白赢 1 盘 -> A 共 2 胜
        # B 执黑赢 1 盘 -> B 共 1 胜
        # 1 平局
        results = [
            _gr(BLACK, "A", "B"),                    # A 执黑胜
            _gr(WHITE, "B", "A"),                    # A 执白胜
            _gr(BLACK, "B", "A"),                    # B 执黑胜
            _gr(0, "A", "B", reason="draw"),         # 平局
        ]
        s = summarize_results(results, player_a_name="A", player_b_name="B")
        assert s["total_games"] == 4
        assert s["player_a_wins"] == 2
        assert s["player_b_wins"] == 1
        assert s["draws"] == 1
        assert s["player_a_win_rate"] == pytest.approx(2 / 4)
        assert s["player_b_win_rate"] == pytest.approx(1 / 4)
        assert s["draw_rate"] == pytest.approx(1 / 4)

    def test_draw_does_not_count_as_either_win(self):
        results = [_gr(0, "A", "B", reason="draw") for _ in range(3)]
        s = summarize_results(results, player_a_name="A", player_b_name="B")
        assert s["player_a_wins"] == 0
        assert s["player_b_wins"] == 0
        assert s["draws"] == 3
        assert s["draw_rate"] == pytest.approx(1.0)

    def test_avg_moves(self):
        results = [
            _gr(BLACK, "A", "B", num_moves=20),
            _gr(WHITE, "A", "B", num_moves=40),
        ]
        s = summarize_results(results, player_a_name="A", player_b_name="B")
        assert s["avg_moves"] == pytest.approx(30.0)

    def test_empty_results(self):
        s = summarize_results([], player_a_name="A", player_b_name="B")
        assert s["total_games"] == 0
        assert s["draw_rate"] == 0.0
        assert s["player_a_win_rate"] == 0.0
        assert s["avg_moves"] == 0.0


# ---------------------------------------------------------------------------
# compute_win_rate / compute_draw_rate
# ---------------------------------------------------------------------------
class TestRates:
    def test_compute_win_rate_across_seats(self):
        results = [
            _gr(BLACK, "A", "B"),
            _gr(WHITE, "B", "A"),
            _gr(BLACK, "B", "A"),
            _gr(0, "A", "B", reason="draw"),
        ]
        assert compute_win_rate(results, "A") == pytest.approx(0.5)
        assert compute_win_rate(results, "B") == pytest.approx(0.25)

    def test_compute_draw_rate(self):
        results = [
            _gr(0, "A", "B", reason="draw"),
            _gr(BLACK, "A", "B"),
            _gr(0, "A", "B", reason="max_moves"),
        ]
        assert compute_draw_rate(results) == pytest.approx(2 / 3)

    def test_empty_rates(self):
        assert compute_win_rate([], "A") == 0.0
        assert compute_draw_rate([]) == 0.0


# ---------------------------------------------------------------------------
# should_promote
# ---------------------------------------------------------------------------
class TestShouldPromote:
    def test_above_threshold(self):
        assert should_promote(0.55, threshold=0.55) is True
        assert should_promote(0.60, threshold=0.55) is True
        assert should_promote(1.0, threshold=0.55) is True

    def test_below_threshold(self):
        assert should_promote(0.54, threshold=0.55) is False
        assert should_promote(0.0, threshold=0.55) is False

    def test_custom_threshold(self):
        assert should_promote(0.61, threshold=0.6) is True
        assert should_promote(0.59, threshold=0.6) is False


def _make_candidate_checkpoint(tmp_path):
    model = object()
    candidate_path = tmp_path / "latest.pt"
    candidate_path.write_text("candidate", encoding="utf-8")
    return model, candidate_path


class TestPromotionGuards:
    def test_random_opponent_never_promotes(self, tmp_path):
        from main_evaluate import _maybe_promote

        model, candidate_path = _make_candidate_checkpoint(tmp_path)
        best_path = tmp_path / "best.pt"

        promoted = _maybe_promote(
            opponent="random",
            candidate_win_rate=1.0,
            threshold=0.55,
            promote_flag=True,
            candidate_path=str(candidate_path),
            best_path=str(best_path),
            candidate_model=model,
            num_games=4,
            save_fn=lambda *args, **kwargs: pytest.fail("unexpected promotion"),
        )

        assert promoted is False
        assert not best_path.exists()
        assert candidate_path.exists()

    def test_threshold_without_promote_flag_does_not_replace_best(self, tmp_path):
        from main_evaluate import _maybe_promote

        model, candidate_path = _make_candidate_checkpoint(tmp_path)
        best_path = tmp_path / "best.pt"

        promoted = _maybe_promote(
            opponent="best",
            candidate_win_rate=0.60,
            threshold=0.55,
            promote_flag=False,
            candidate_path=str(candidate_path),
            best_path=str(best_path),
            candidate_model=model,
            num_games=10,
            save_fn=lambda *args, **kwargs: pytest.fail("unexpected promotion"),
        )

        assert promoted is False
        assert not best_path.exists()
        assert candidate_path.exists()

    def test_promote_writes_best_checkpoint_with_metadata(self, tmp_path):
        from main_evaluate import _maybe_promote

        model, candidate_path = _make_candidate_checkpoint(tmp_path)
        best_path = tmp_path / "best.pt"
        saved = {}

        def fake_save(model_arg, path, metadata=None):
            saved["model"] = model_arg
            saved["path"] = path
            saved["metadata"] = metadata
            best_path.write_text("best", encoding="utf-8")

        promoted = _maybe_promote(
            opponent="best",
            candidate_win_rate=0.60,
            threshold=0.55,
            promote_flag=True,
            candidate_path=str(candidate_path),
            best_path=str(best_path),
            candidate_model=model,
            num_games=10,
            save_fn=fake_save,
        )

        assert promoted is True
        assert best_path.exists()
        assert candidate_path.exists()

        assert saved["model"] is model
        assert saved["path"] == str(best_path)
        metadata = saved["metadata"]
        assert metadata["promoted_from"] == str(candidate_path)
        assert metadata["candidate_win_rate"] == pytest.approx(0.60)
        assert metadata["threshold"] == pytest.approx(0.55)
        assert metadata["num_games"] == 10


# ---------------------------------------------------------------------------
# format_match_summary
# ---------------------------------------------------------------------------
class TestFormat:
    def test_returns_string_with_key_fields(self):
        results = [
            _gr(BLACK, "candidate", "best"),
            _gr(WHITE, "best", "candidate"),
            _gr(0, "candidate", "best", reason="draw"),
        ]
        s = summarize_results(results, player_a_name="candidate", player_b_name="best")
        text = format_match_summary(s)
        assert isinstance(text, str)
        # 关键字段应出现
        assert "candidate" in text
        assert "best" in text
        assert "对局总数" in text or "total" in text.lower()
