"""命令行人机对弈。

主要 API：

- :func:`parse_human_move` : ``"H8"`` / ``"j10"`` / ``"O15"`` -> action_index。
- :func:`is_quit_command`  : 识别 ``q`` / ``quit`` / ``exit``。
- :func:`is_help_command`  : 识别 ``help`` / ``?`` / ``h``。
- :func:`get_help_text`    : 返回帮助文字(供 ``run_cli_game`` 与测试共用)。
- :func:`run_cli_game`     : 跑一盘人机对局，``input_fn`` / ``output_fn`` 可注入
  以便测试不需要真实 stdin。
"""

from __future__ import annotations

from typing import Callable, Optional

from game.board import BLACK, BOARD_SIZE, WHITE, Board
from game.coordinates import index_to_coord
from game.encoder import index_to_action
from game.rules_basic import _find_any_winner, check_winner, is_game_over
from ui.board_renderer import render_board


QUIT_TOKENS = frozenset({"q", "quit", "exit"})
HELP_TOKENS = frozenset({"help", "?", "h"})


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
def parse_human_move(text: str, board_size: int = BOARD_SIZE) -> int:
    """``"H8"`` / ``"j10"`` 等坐标 -> action_index。"""
    if not isinstance(text, str):
        raise ValueError(f"输入必须是字符串: {text!r}")
    s = text.strip().upper()
    if not s:
        raise ValueError("输入为空")
    if len(s) < 2 or len(s) > 3:
        raise ValueError(f"格式错误，请使用 H8 / J10 这样的坐标: {text!r}")
    letter = s[0]
    max_letter = chr(ord("A") + board_size - 1)
    if not ("A" <= letter <= max_letter):
        raise ValueError(
            f"列字母 {letter!r} 超出范围 (合法: A..{max_letter}): {text!r}"
        )
    number_part = s[1:]
    if not number_part.isdigit():
        raise ValueError(f"行号必须是数字: {text!r}")
    number = int(number_part)
    if not (1 <= number <= board_size):
        raise ValueError(
            f"行号 {number} 超出范围 (合法: 1..{board_size}): {text!r}"
        )
    x = ord(letter) - ord("A")
    y = number - 1
    return y * board_size + x


def is_quit_command(text: object) -> bool:
    return isinstance(text, str) and text.strip().lower() in QUIT_TOKENS


def is_help_command(text: object) -> bool:
    return isinstance(text, str) and text.strip().lower() in HELP_TOKENS


def get_help_text() -> str:
    return (
        "命令说明:\n"
        "  - 输入坐标如 H8 / J10 / A1 / O15 落子\n"
        "  - 输入 q / quit / exit 退出当前对局\n"
        "  - 输入 help / ? / h 查看本说明\n"
    )


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
def run_cli_game(
    ai_player,
    human_color: int = BLACK,
    board_size: int = BOARD_SIZE,
    max_moves: Optional[int] = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> Optional[int]:
    """跑一盘人机对局。

    Parameters
    ----------
    ai_player : 任意有 ``select_action(board) -> int`` 的玩家(例如 ``ModelMCTSPlayer``)。
    human_color : ``BLACK(1)`` / ``WHITE(-1)``，决定人类持哪一方。
    board_size : 棋盘边长。
    max_moves  : 防死循环上限，默认 ``board_size**2``。
    input_fn / output_fn : 为测试方便注入；默认走 ``input`` / ``print``。

    Returns
    -------
    winner : ``BLACK(1)`` / ``WHITE(-1)`` / ``0``(平局)。用户主动 quit 返回 ``None``。
    """
    if max_moves is None:
        max_moves = board_size * board_size

    board = Board()
    output_fn(get_help_text())
    output_fn(f"\n你执 {'黑(X)' if human_color == BLACK else '白(O)'}\n")
    output_fn(render_board(board))

    move_number = 0
    while move_number < max_moves and not is_game_over(board, board.last_move):
        if board.current_player == human_color:
            text = input_fn("> ")
            if is_quit_command(text):
                output_fn("已退出对局")
                return None
            if is_help_command(text):
                output_fn(get_help_text())
                continue
            try:
                action = parse_human_move(text, board_size)
            except ValueError as exc:
                output_fn(f"非法输入: {exc}\n请重试 (输入 help 查看说明)")
                continue
            x, y = index_to_action(action, board_size)
            if not board.is_legal_move(x, y):
                output_fn(
                    f"位置 {index_to_coord(x, y)} 不合法或已被占据，请重试"
                )
                continue
            move_label = "你"
        else:
            output_fn("AI 思考中...")
            action = ai_player.select_action(board)
            if action is None:
                output_fn("AI 无合法落子，对局结束")
                break
            action = int(action)
            x, y = index_to_action(action, board_size)
            if not board.is_legal_move(x, y):
                output_fn(f"AI 返回非法动作 ({x}, {y})，强制结束")
                break
            move_label = "AI"

        board.place_stone(x, y)
        move_number += 1
        output_fn(f"{move_label} 落子: {index_to_coord(x, y)}")
        output_fn(render_board(board))

    winner = check_winner(board, board.last_move)
    if winner == 0:
        winner = _find_any_winner(board)

    if winner == BLACK:
        output_fn("\n>>> 黑方胜 <<<")
    elif winner == WHITE:
        output_fn("\n>>> 白方胜 <<<")
    else:
        output_fn("\n>>> 平局 <<<")
    return int(winner)


__all__ = [
    "QUIT_TOKENS",
    "HELP_TOKENS",
    "parse_human_move",
    "is_quit_command",
    "is_help_command",
    "get_help_text",
    "run_cli_game",
]
