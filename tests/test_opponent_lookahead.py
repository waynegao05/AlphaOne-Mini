from __future__ import annotations

from engine.opponent_lookahead import predict_likely_moves, select_lookahead_move
from game.board import BLACK, WHITE, Board
from game.encoder import action_to_index, index_to_action


def _set_stones(board: Board, black=(), white=()):
    black = tuple(black)
    white = tuple(white)
    for x, y in black:
        board.grid[x][y] = BLACK
    for x, y in white:
        board.grid[x][y] = WHITE
    board.move_count = len(black) + len(white)
    board.current_player = BLACK if board.move_count % 2 == 0 else WHITE


def test_predict_likely_moves_ranks_forcing_reply_highest():
    board = Board()
    _set_stones(board, white=[(5, 7), (6, 7), (7, 7)])
    board.current_player = BLACK

    predictions = predict_likely_moves(board, WHITE, top_k=4)

    assert predictions
    top_action = predictions[0].action
    assert top_action in {action_to_index(4, 7), action_to_index(8, 7)}
    assert predictions[0].probability >= predictions[-1].probability
    assert "open_four" in predictions[0].threats


def test_select_lookahead_move_prevents_opponent_open_four_plan():
    board = Board()
    _set_stones(board, white=[(5, 7), (6, 7), (7, 7)])
    board.current_player = BLACK

    result = select_lookahead_move(board, BLACK, depth=4, branch_factor=3)

    assert result is not None
    x, y = index_to_action(result.action)
    assert (x, y) in {(4, 7), (8, 7)}
    assert result.principal_variation
    assert result.reason.startswith("lookahead")


def test_lookahead_returns_none_in_quiet_position():
    board = Board()
    _set_stones(board, black=[(7, 7)], white=[(8, 7)])
    board.current_player = BLACK

    assert select_lookahead_move(board, BLACK, depth=4, branch_factor=3) is None


def test_lookahead_does_not_mutate_board():
    board = Board()
    _set_stones(board, white=[(5, 7), (6, 7), (7, 7)])
    board.current_player = BLACK
    snapshot = [row[:] for row in board.grid]
    current = board.current_player
    count = board.move_count
    last = board.last_move

    _ = select_lookahead_move(board, BLACK, depth=4, branch_factor=3)

    assert board.grid == snapshot
    assert board.current_player == current
    assert board.move_count == count
    assert board.last_move == last
