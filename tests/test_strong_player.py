"""Tests for ``engine.strong_player.StrongPlayer``.

We use a ``FakeMCTSPlayer`` so tests don't need PyTorch or a checkpoint; the
StrongPlayer's tactical and VCF tiers are the focus, and the MCTS tier is
exercised only as a fall-through path.
"""

from __future__ import annotations

from typing import Optional

import pytest

from engine.simulation import temporary_stone
from engine.strong_player import StrongPlayer
from game.board import BLACK, BOARD_SIZE, EMPTY, WHITE, Board
from game.encoder import action_to_index, index_to_action
from game.rules_basic import check_winner


def _set_stones(board: Board, black, white):
    for x, y in black:
        board.grid[x][y] = BLACK
    for x, y in white:
        board.grid[x][y] = WHITE
    board.move_count = len(black) + len(white)
    board.current_player = BLACK if len(black) == len(white) else WHITE


class _FakeMCTSPlayer:
    """Records that it was asked, and returns a configurable action."""

    def __init__(self, action: Optional[int] = None, name: str = "FakeMCTS"):
        self.name = name
        self._action = action
        self.calls = 0

    def select_action(self, board: Board) -> Optional[int]:
        self.calls += 1
        return self._action


def _build_player(rule_mode="basic", mcts=None) -> StrongPlayer:
    if mcts is None:
        mcts = _FakeMCTSPlayer(action=None)
    return StrongPlayer(
        mcts_player=mcts,
        rule_mode=rule_mode,
        vcf_depth=7,
        vcf_defense_depth=5,
        vcf_node_budget=5000,
        name="StrongPlayer_test",
    )


# ---------------------------------------------------------------------------
# Tier 1 / 2: immediate win and immediate block dominate everything
# ---------------------------------------------------------------------------
class TestImmediateTiers:
    def test_immediate_win_is_taken_over_mcts(self):
        board = Board()
        _set_stones(board, black=[(3, 7), (4, 7), (5, 7), (6, 7)], white=[])
        # Fake MCTS would prefer a useless cell, but Strong must take the win.
        useless = action_to_index(0, 0, BOARD_SIZE)
        mcts = _FakeMCTSPlayer(action=useless)
        player = _build_player(mcts=mcts)
        action = player.select_action(board)
        assert action is not None
        x, y = index_to_action(int(action), BOARD_SIZE)
        with temporary_stone(board, x, y, BLACK):
            assert check_winner(board, board.last_move) == BLACK
        # MCTS must not have been the deciding factor.
        assert mcts.calls == 0

    def test_immediate_block_dominates_mcts(self):
        board = Board()
        # White is one move from five; black to move must block.
        _set_stones(board, black=[], white=[(3, 7), (4, 7), (5, 7), (6, 7)])
        useless = action_to_index(14, 14, BOARD_SIZE)
        mcts = _FakeMCTSPlayer(action=useless)
        player = _build_player(mcts=mcts)
        action = player.select_action(board)
        assert action is not None
        x, y = index_to_action(int(action), BOARD_SIZE)
        # The chosen cell must be one of white's two extension ends.
        assert (x, y) in {(2, 7), (7, 7)}
        assert mcts.calls == 0


# ---------------------------------------------------------------------------
# Tier 3: VCF mate beats MCTS suggestion
# ---------------------------------------------------------------------------
class TestVcfTier:
    def test_vcf_mate_overrides_mcts(self):
        board = Board()
        # Black: 3 in row (4,7),(5,7),(6,7) — open four with one move.
        _set_stones(board, black=[(4, 7), (5, 7), (6, 7)], white=[])
        useless = action_to_index(0, 0, BOARD_SIZE)
        mcts = _FakeMCTSPlayer(action=useless)
        player = _build_player(mcts=mcts)
        action = player.select_action(board)
        assert action is not None
        # The chosen move must create a winning continuation. After the move,
        # opponent must not be able to prevent black from winning next ply.
        x, y = index_to_action(int(action), BOARD_SIZE)
        with temporary_stone(board, x, y, BLACK):
            survives = False
            for ox in range(BOARD_SIZE):
                if survives:
                    break
                for oy in range(BOARD_SIZE):
                    if board.grid[ox][oy] != EMPTY:
                        continue
                    with temporary_stone(board, ox, oy, WHITE):
                        any_win = False
                        for wx in range(BOARD_SIZE):
                            if any_win:
                                break
                            for wy in range(BOARD_SIZE):
                                if board.grid[wx][wy] != EMPTY:
                                    continue
                                with temporary_stone(board, wx, wy, BLACK):
                                    if check_winner(
                                        board, board.last_move
                                    ) == BLACK:
                                        any_win = True
                                        break
                        if not any_win:
                            survives = True
        assert not survives, "after StrongPlayer's move there should be no escape for white"


# ---------------------------------------------------------------------------
# VCT-lite tier: double open-three threat beats MCTS suggestion
# ---------------------------------------------------------------------------
class TestVctTier:
    def test_vct_double_open_three_overrides_mcts(self):
        board = Board()
        _set_stones(
            board,
            black=[(6, 7), (8, 7), (7, 6), (7, 8)],
            white=[],
        )
        board.current_player = BLACK
        center = action_to_index(7, 7, BOARD_SIZE)
        useless = action_to_index(0, 0, BOARD_SIZE)
        mcts = _FakeMCTSPlayer(action=useless)
        player = _build_player(mcts=mcts)

        action = player.select_action(board)

        assert action == center
        assert mcts.calls == 0

    def test_opponent_vct_is_preempted_before_mcts(self):
        board = Board()
        _set_stones(
            board,
            black=[],
            white=[(6, 7), (8, 7), (7, 6), (7, 8)],
        )
        board.current_player = BLACK
        center = action_to_index(7, 7, BOARD_SIZE)
        useless = action_to_index(0, 0, BOARD_SIZE)
        mcts = _FakeMCTSPlayer(action=useless)
        player = _build_player(mcts=mcts)

        action = player.select_action(board)

        assert action == center
        assert mcts.calls == 0


# ---------------------------------------------------------------------------
# Tier 7 / 8: fall-through to MCTS, then to TacticalPlayer
# ---------------------------------------------------------------------------
class TestFallThroughTiers:
    def test_quiet_position_uses_mcts_when_available(self):
        board = Board()
        # Quiet position: just one stone at the centre.
        _set_stones(board, black=[(7, 7)], white=[])
        # MCTS suggests a legal cell.
        suggestion = action_to_index(8, 8, BOARD_SIZE)
        mcts = _FakeMCTSPlayer(action=suggestion)
        player = _build_player(mcts=mcts)
        action = player.select_action(board)
        assert action == suggestion
        assert mcts.calls == 1

    def test_mcts_returning_illegal_falls_back_to_tactical(self):
        board = Board()
        _set_stones(board, black=[(7, 7)], white=[])
        # MCTS suggests an OCCUPIED cell (illegal).
        bad = action_to_index(7, 7, BOARD_SIZE)
        mcts = _FakeMCTSPlayer(action=bad)
        player = _build_player(mcts=mcts)
        action = player.select_action(board)
        assert action is not None
        x, y = index_to_action(int(action), BOARD_SIZE)
        assert board.is_legal_move(x, y), "StrongPlayer must never return an illegal cell"

    def test_mcts_returning_none_falls_back_to_tactical(self):
        board = Board()
        _set_stones(board, black=[(7, 7)], white=[])
        mcts = _FakeMCTSPlayer(action=None)
        player = _build_player(mcts=mcts)
        action = player.select_action(board)
        assert action is not None
        x, y = index_to_action(int(action), BOARD_SIZE)
        assert board.is_legal_move(x, y)


# ---------------------------------------------------------------------------
# Smoke: never returns illegal action in random-ish opening
# ---------------------------------------------------------------------------
def test_strong_player_smoke_returns_legal_on_opening():
    board = Board()
    _set_stones(
        board,
        black=[(7, 7), (8, 8)],
        white=[(7, 8), (8, 7)],
    )
    mcts = _FakeMCTSPlayer(action=action_to_index(6, 6, BOARD_SIZE))
    player = _build_player(mcts=mcts)
    action = player.select_action(board)
    assert action is not None
    x, y = index_to_action(int(action), BOARD_SIZE)
    assert board.is_legal_move(x, y)


def test_strong_player_does_not_mutate_input_board():
    board = Board()
    _set_stones(board, black=[(3, 7), (4, 7), (5, 7)], white=[(7, 7)])
    snapshot_grid = [row[:] for row in board.grid]
    snapshot_count = board.move_count
    snapshot_player = board.current_player
    snapshot_last = board.last_move
    player = _build_player(mcts=_FakeMCTSPlayer(action=action_to_index(0, 0, BOARD_SIZE)))
    _ = player.select_action(board)
    assert board.grid == snapshot_grid
    assert board.move_count == snapshot_count
    assert board.current_player == snapshot_player
    assert board.last_move == snapshot_last


def test_alphaone_lookahead_preempts_likely_opponent_threat_before_mcts():
    board = Board()
    _set_stones(board, black=[(7, 7)], white=[(5, 9), (6, 9), (7, 9)])
    board.current_player = BLACK
    useless = action_to_index(14, 14, BOARD_SIZE)
    mcts = _FakeMCTSPlayer(action=useless)
    player = StrongPlayer(
        mcts_player=mcts,
        rule_mode="basic",
        vcf_depth=3,
        vcf_defense_depth=3,
        vcf_node_budget=1000,
        vct_depth=3,
        vct_node_budget=1000,
        lookahead_depth=4,
        lookahead_branch_factor=5,
        name="AlphaOne-Mini_test",
    )

    action = player.select_action(board)

    assert action in {action_to_index(4, 9, BOARD_SIZE), action_to_index(8, 9, BOARD_SIZE)}
    assert player.last_decision_reason in {"vcf_defense"} or player.last_decision_reason.startswith("lookahead")
    assert mcts.calls == 0


def test_alphaone_threat_safety_preempts_speculative_lookahead_in_external_ai_trap():
    board = Board()
    # External-AI trap prefix from the 100-game fixed-side benchmark:
    # 1. H8 H9 2. I9.  White must first remove black's next open-three
    # escalation at G7 instead of following the speculative I8 lookahead line.
    _set_stones(
        board,
        black=[(7, 7), (8, 8)],
        white=[(7, 8)],
    )
    board.current_player = WHITE
    mcts = _FakeMCTSPlayer(action=action_to_index(14, 14, BOARD_SIZE))
    player = StrongPlayer(
        mcts_player=mcts,
        rule_mode="basic",
        vcf_depth=3,
        vcf_defense_depth=3,
        vcf_node_budget=1000,
        vct_depth=3,
        vct_node_budget=1000,
        lookahead_depth=4,
        lookahead_branch_factor=3,
        name="AlphaOne-Mini_test",
    )

    action = player.select_action(board)

    assert action == action_to_index(6, 6, BOARD_SIZE)  # G7
    assert player.last_decision_reason.startswith("threat_safety")
    assert mcts.calls == 0
