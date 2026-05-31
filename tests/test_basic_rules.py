"""基础胜负规则与随机对战测试。"""

import pytest

from evaluate.players import RandomPlayer
from game.board import BLACK, BOARD_SIZE, EMPTY, WHITE, Board
from game.rules_basic import (
    check_five_or_more,
    check_winner,
    count_continuous_stones,
    is_draw,
    is_game_over,
)


# ---- 工具函数 ---------------------------------------------------------------
def _set_line(board: Board, start_x: int, start_y: int, dx: int, dy: int, color: int, length: int):
    for i in range(length):
        board.grid[start_x + i * dx][start_y + i * dy] = color


# ---- count_continuous_stones -----------------------------------------------
def test_count_continuous_stones_basic():
    board = Board()
    _set_line(board, 3, 7, 1, 0, BLACK, 4)  # (3,7)..(6,7) 黑
    # 从 (3,7) 向右数同色，应有 3 个(不含自身)
    assert count_continuous_stones(board, 3, 7, 1, 0, BLACK) == 3
    # 从 (3,7) 向左空，0
    assert count_continuous_stones(board, 3, 7, -1, 0, BLACK) == 0


# ---- 五连判胜：四个方向 ----------------------------------------------------
def test_horizontal_five_wins():
    board = Board()
    _set_line(board, 3, 7, 1, 0, BLACK, 5)  # (3..7, 7) 黑
    last_move = (5, 7, BLACK)
    assert check_five_or_more(board, 5, 7, BLACK)
    assert check_winner(board, last_move) == BLACK


def test_vertical_five_wins():
    board = Board()
    _set_line(board, 7, 3, 0, 1, WHITE, 5)  # (7, 3..7) 白
    last_move = (7, 5, WHITE)
    assert check_winner(board, last_move) == WHITE


def test_main_diagonal_five_wins():
    # ↘ 方向：(0,0),(1,1),(2,2),(3,3),(4,4)
    board = Board()
    _set_line(board, 0, 0, 1, 1, BLACK, 5)
    last_move = (2, 2, BLACK)
    assert check_winner(board, last_move) == BLACK


def test_anti_diagonal_five_wins():
    # ↗ 方向：(0,4),(1,3),(2,2),(3,1),(4,0)
    board = Board()
    _set_line(board, 0, 4, 1, -1, WHITE, 5)
    last_move = (2, 2, WHITE)
    assert check_winner(board, last_move) == WHITE


def test_six_in_a_row_also_wins():
    # 当前阶段长连(>=5)也判胜，暂不实现长连禁手。
    board = Board()
    _set_line(board, 0, 0, 1, 0, BLACK, 6)
    last_move = (3, 0, BLACK)
    assert check_winner(board, last_move) == BLACK


# ---- 不应误判 --------------------------------------------------------------
def test_four_in_a_row_is_not_a_win():
    board = Board()
    _set_line(board, 0, 0, 1, 0, BLACK, 4)
    last_move = (3, 0, BLACK)
    assert check_winner(board, last_move) == 0


def test_blocked_segment_is_not_a_win():
    # 黑3 + 白阻断 + 黑2，不能误判
    board = Board()
    _set_line(board, 0, 0, 1, 0, BLACK, 3)
    board.grid[3][0] = WHITE
    _set_line(board, 4, 0, 1, 0, BLACK, 2)
    last_move = (5, 0, BLACK)
    assert check_winner(board, last_move) == 0


def test_empty_board_no_winner():
    board = Board()
    assert check_winner(board, board.last_move) == 0
    assert not is_game_over(board, board.last_move)


# ---- 平局 ------------------------------------------------------------------
def _make_full_board_no_winner() -> Board:
    """构造一张被填满但无 5 连的棋盘。

    模式：每两列同色，每行向右整体偏移 2 列，保证横纵和两条对角都不会出现 5 连。
    """
    board = Board()
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            block_index = (x + 2 * y) // 2
            board.grid[x][y] = BLACK if block_index % 2 == 0 else WHITE
    board.move_count = BOARD_SIZE * BOARD_SIZE
    last_color = board.grid[BOARD_SIZE - 1][BOARD_SIZE - 1]
    board.last_move = (BOARD_SIZE - 1, BOARD_SIZE - 1, last_color)
    board.current_player = -last_color
    return board


def test_full_board_pattern_has_no_five():
    board = _make_full_board_no_winner()
    # 全棋盘扫描，确认任何方向都不存在 5 连
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            color = board.grid[x][y]
            assert color != EMPTY
            assert not check_five_or_more(board, x, y, color)


def test_draw_when_full_no_winner():
    board = _make_full_board_no_winner()
    assert check_winner(board, board.last_move) == 0
    assert is_draw(board)
    assert is_game_over(board, board.last_move)


def test_full_board_with_winner_is_not_draw_even_if_last_move_elsewhere():
    board = _make_full_board_no_winner()
    for x in range(5):
        board.grid[x][0] = BLACK
    board.last_move = (BOARD_SIZE - 1, BOARD_SIZE - 1, board.grid[BOARD_SIZE - 1][BOARD_SIZE - 1])

    assert check_winner(board, board.last_move) == 0
    assert not is_draw(board)
    assert is_game_over(board, board.last_move)


def test_not_draw_when_board_not_full():
    board = Board()
    board.place_stone(7, 7)
    assert not is_draw(board)


# ---- 随机对战集成测试 -------------------------------------------------------
def test_random_game_completes_without_illegal_move():
    board = Board()
    bp = RandomPlayer(seed=42)
    wp = RandomPlayer(seed=43)
    max_steps = BOARD_SIZE * BOARD_SIZE + 1  # 防御性上限
    steps = 0
    while not is_game_over(board, board.last_move):
        steps += 1
        assert steps <= max_steps, "随机对战出现死循环"
        player = bp if board.current_player == BLACK else wp
        move = player.select_move(board)
        if move is None:
            break
        x, y = move
        assert board.is_legal_move(x, y), f"非法落子: {(x, y)}"
        board.place_stone(x, y)
    winner = check_winner(board, board.last_move)
    assert winner in (BLACK, WHITE, 0)


@pytest.mark.parametrize("seed", [0, 1, 7, 123, 2024])
def test_multiple_random_games_terminate(seed):
    board = Board()
    bp = RandomPlayer(seed=seed)
    wp = RandomPlayer(seed=seed + 100)
    steps = 0
    max_steps = BOARD_SIZE * BOARD_SIZE + 1
    while not is_game_over(board, board.last_move):
        steps += 1
        assert steps <= max_steps
        player = bp if board.current_player == BLACK else wp
        move = player.select_move(board)
        if move is None:
            break
        board.place_stone(*move)
    assert is_game_over(board, board.last_move) or board.move_count == BOARD_SIZE * BOARD_SIZE
