"""Tests for TacticalPlayer decision priorities."""

from __future__ import annotations

from evaluate.arena import Arena
from evaluate.players import RandomPlayer
from game.board import BLACK, EMPTY, WHITE, Board
from game.encoder import action_to_index, index_to_action


def _set_stones(board: Board, stones: list[tuple[int, int, int]]) -> None:
    for x, y, color in stones:
        board.grid[x][y] = color
    board.move_count = sum(
        1
        for x in range(board.BOARD_SIZE)
        for y in range(board.BOARD_SIZE)
        if board.grid[x][y] != EMPTY
    )


def test_tactical_player_takes_immediate_win():
    from engine.tactical_player import TacticalPlayer

    board = Board()
    _set_stones(board, [(5, 7, BLACK), (6, 7, BLACK), (7, 7, BLACK), (8, 7, BLACK)])
    board.current_player = BLACK

    action = TacticalPlayer(name="tactical").select_action(board)

    assert action in {action_to_index(4, 7), action_to_index(9, 7)}


def test_tactical_player_blocks_opponent_immediate_win():
    from engine.tactical_player import TacticalPlayer

    board = Board()
    _set_stones(board, [(5, 7, WHITE), (6, 7, WHITE), (7, 7, WHITE), (8, 7, WHITE)])
    board.current_player = BLACK

    action = TacticalPlayer(name="tactical").select_action(board)

    assert action in {action_to_index(4, 7), action_to_index(9, 7)}


def test_tactical_player_prefers_own_open_four_then_blocks_open_four():
    from engine.tactical_player import TacticalPlayer

    own_board = Board()
    _set_stones(own_board, [(5, 7, BLACK), (6, 7, BLACK), (7, 7, BLACK)])
    own_board.current_player = BLACK
    own_action = TacticalPlayer(name="tactical").select_action(own_board)
    assert own_action == action_to_index(8, 7)

    block_board = Board()
    _set_stones(block_board, [(5, 7, WHITE), (6, 7, WHITE), (7, 7, WHITE)])
    block_board.current_player = BLACK
    block_action = TacticalPlayer(name="tactical").select_action(block_board)
    assert block_action == action_to_index(8, 7)


def test_tactical_player_never_returns_occupied_or_forbidden_point():
    from engine.tactical_player import TacticalPlayer

    occupied_board = Board()
    _set_stones(occupied_board, [(7, 7, BLACK)])
    occupied_board.current_player = WHITE
    action = TacticalPlayer(name="tactical").select_action(occupied_board)
    assert action != action_to_index(7, 7)
    x, y = index_to_action(action)
    assert occupied_board.is_legal_move(x, y)

    forbidden_board = Board()
    _set_stones(
        forbidden_board,
        [(5, 7, BLACK), (6, 7, BLACK), (7, 7, BLACK), (8, 7, BLACK), (9, 7, BLACK)],
    )
    forbidden_board.current_player = BLACK
    action = TacticalPlayer(rule_mode="forbidden", name="tactical").select_action(
        forbidden_board
    )
    assert action != action_to_index(10, 7)


def test_tactical_player_empty_board_selects_center():
    from engine.tactical_player import TacticalPlayer

    board = Board()

    assert TacticalPlayer(name="tactical").select_action(board) == action_to_index(7, 7)


def test_tactical_player_vs_random_completes_game():
    from engine.tactical_player import TacticalPlayer

    result = Arena(
        TacticalPlayer(name="tactical"),
        RandomPlayer(seed=1, name="random"),
        max_moves=80,
    ).play_one_game()

    assert result.winner in (BLACK, WHITE, 0)
    assert 0 <= result.num_moves <= 80
    assert all(isinstance(action, int) for action in result.moves)


def test_tactical_player_limits_threat_search_to_candidates(monkeypatch):
    import engine.tactical_player as tactical_module
    from engine.tactical_player import TacticalPlayer

    board = Board()
    _set_stones(board, [(7, 7, BLACK), (8, 7, WHITE)])
    board.current_player = BLACK

    seen_lengths: list[int] = []

    def fake_wins(board_arg, color, rule_mode="basic", actions=None):
        seen_lengths.append(len(list(actions)))
        return []

    monkeypatch.setattr(tactical_module, "find_immediate_winning_moves", fake_wins)
    monkeypatch.setattr(tactical_module, "find_immediate_blocking_moves", fake_wins)
    monkeypatch.setattr(tactical_module, "find_open_four_moves", fake_wins)

    action = TacticalPlayer(candidate_radius=1, max_candidates=8).select_action(board)

    assert action is not None
    assert seen_lengths
    assert max(seen_lengths) <= 8
