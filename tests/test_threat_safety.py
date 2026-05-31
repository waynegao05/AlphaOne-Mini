from __future__ import annotations

from engine.threat_safety import (
    RISK_SCORES,
    evaluate_opponent_reply_risk,
    select_threat_safe_move,
)
from game.board import BLACK, WHITE, Board
from game.encoder import action_to_index, index_to_action


def _set_stones(board: Board, black=(), white=()):
    for x, y in black:
        board.grid[x][y] = BLACK
    for x, y in white:
        board.grid[x][y] = WHITE
    board.move_count = len(tuple(black)) + len(tuple(white))
    board.current_player = BLACK if board.move_count % 2 == 0 else WHITE


def test_opponent_reply_risk_detects_open_four_escalation():
    board = Board()
    _set_stones(board, white=[(5, 7), (6, 7), (7, 7)])
    board.current_player = BLACK

    risk, reply, threats = evaluate_opponent_reply_risk(board, BLACK)

    assert risk >= RISK_SCORES["open_four"]
    assert reply in {action_to_index(4, 7), action_to_index(8, 7)}
    assert "open_four" in threats


def test_select_threat_safe_move_blocks_single_open_three_before_mcts():
    board = Board()
    _set_stones(board, white=[(5, 7), (6, 7), (7, 7)])
    board.current_player = BLACK

    safety = select_threat_safe_move(board, BLACK)

    assert safety is not None
    x, y = index_to_action(safety.action)
    assert (x, y) in {(4, 7), (8, 7)}
    assert safety.reply_risk < RISK_SCORES["open_four"]


def test_select_threat_safe_move_returns_none_in_quiet_opening():
    board = Board()
    _set_stones(board, black=[(7, 7)], white=[(7, 8)])
    board.current_player = BLACK

    assert select_threat_safe_move(board, BLACK) is None


def test_threat_safety_does_not_mutate_board():
    board = Board()
    _set_stones(board, white=[(5, 7), (6, 7), (7, 7)])
    board.current_player = BLACK
    snapshot = [row[:] for row in board.grid]
    current = board.current_player
    count = board.move_count
    last = board.last_move

    _ = select_threat_safe_move(board, BLACK)

    assert board.grid == snapshot
    assert board.current_player == current
    assert board.move_count == count
    assert board.last_move == last
