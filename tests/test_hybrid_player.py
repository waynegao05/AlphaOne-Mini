"""Tests for HybridPlayer tactical fast path and MCTS fallback."""

from __future__ import annotations

from game.board import BLACK, EMPTY, WHITE, Board
from game.encoder import action_to_index


class StubMCTSPlayer:
    def __init__(self, action: int | None):
        self.action = action
        self.calls = 0
        self.name = "stub_mcts"

    def select_action(self, board: Board):
        self.calls += 1
        return self.action


def _set_stones(board: Board, stones: list[tuple[int, int, int]]) -> None:
    for x, y, color in stones:
        board.grid[x][y] = color
    board.move_count = sum(
        1
        for x in range(board.BOARD_SIZE)
        for y in range(board.BOARD_SIZE)
        if board.grid[x][y] != EMPTY
    )


def test_hybrid_uses_tactical_win_without_calling_mcts():
    from engine.hybrid_player import HybridPlayer

    board = Board()
    _set_stones(board, [(5, 7, BLACK), (6, 7, BLACK), (7, 7, BLACK), (8, 7, BLACK)])
    board.current_player = BLACK
    stub = StubMCTSPlayer(action_to_index(0, 0))

    action = HybridPlayer(mcts_player=stub).select_action(board)

    assert action in {action_to_index(4, 7), action_to_index(9, 7)}
    assert stub.calls == 0


def test_hybrid_blocks_immediate_loss_without_calling_mcts():
    from engine.hybrid_player import HybridPlayer

    board = Board()
    _set_stones(board, [(5, 7, WHITE), (6, 7, WHITE), (7, 7, WHITE), (8, 7, WHITE)])
    board.current_player = BLACK
    stub = StubMCTSPlayer(action_to_index(0, 0))

    action = HybridPlayer(mcts_player=stub).select_action(board)

    assert action in {action_to_index(4, 7), action_to_index(9, 7)}
    assert stub.calls == 0


def test_hybrid_uses_open_four_without_calling_mcts():
    from engine.hybrid_player import HybridPlayer

    board = Board()
    _set_stones(board, [(6, 7, BLACK), (7, 7, BLACK), (8, 7, BLACK)])
    board.current_player = BLACK
    stub = StubMCTSPlayer(action_to_index(0, 0))

    action = HybridPlayer(mcts_player=stub).select_action(board)

    assert action in {action_to_index(5, 7), action_to_index(9, 7)}
    assert stub.calls == 0


def test_hybrid_calls_mcts_when_no_forcing_tactic():
    from engine.hybrid_player import HybridPlayer

    board = Board()
    _set_stones(board, [(7, 7, BLACK)])
    board.current_player = WHITE
    stub = StubMCTSPlayer(action_to_index(8, 8))

    action = HybridPlayer(mcts_player=stub).select_action(board)

    assert action == action_to_index(8, 8)
    assert stub.calls == 1


def test_hybrid_falls_back_when_mcts_returns_illegal_action():
    from engine.hybrid_player import HybridPlayer

    board = Board()
    _set_stones(board, [(7, 7, BLACK)])
    board.current_player = WHITE
    stub = StubMCTSPlayer(action_to_index(7, 7))

    action = HybridPlayer(mcts_player=stub).select_action(board)

    assert action != action_to_index(7, 7)
    assert stub.calls == 1
