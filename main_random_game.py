"""随机对战入口。

运行 100 盘 ``RandomPlayer`` vs ``RandomPlayer`` 对局，
统计黑胜、白胜、平局并打印总结。

CLI 可选参数::

    python main_random_game.py [num_games] [seed]
"""

from __future__ import annotations

import sys
from typing import Optional

from evaluate.players import RandomPlayer
from game.board import BLACK, WHITE, Board
from game.rules_basic import check_winner, is_game_over


def play_one_game(black_player: RandomPlayer, white_player: RandomPlayer) -> int:
    """跑完一盘对局，返回胜者(``BLACK`` / ``WHITE`` / ``0`` 表示平局)。"""
    board = Board()
    while not is_game_over(board, board.last_move):
        player = black_player if board.current_player == BLACK else white_player
        move = player.select_move(board)
        if move is None:
            # 没有合法落子，平局结束。
            break
        x, y = move
        if not board.is_legal_move(x, y):
            raise RuntimeError(f"Player 选出了非法落子: {(x, y)}")
        board.place_stone(x, y)
    return check_winner(board, board.last_move)


def run_random_games(
    num_games: int = 100, seed: Optional[int] = None
) -> dict:
    """运行 ``num_games`` 盘随机对局并返回统计结果。"""
    if num_games <= 0:
        raise ValueError("num_games 必须 >= 1")

    # 给两个玩家不同的种子，避免黑白完全镜像。
    if seed is None:
        black_player = RandomPlayer()
        white_player = RandomPlayer()
    else:
        black_player = RandomPlayer(seed=seed)
        white_player = RandomPlayer(seed=seed + 1)

    black_wins = 0
    white_wins = 0
    draws = 0

    for _ in range(num_games):
        winner = play_one_game(black_player, white_player)
        if winner == BLACK:
            black_wins += 1
        elif winner == WHITE:
            white_wins += 1
        else:
            draws += 1

    return {
        "total": num_games,
        "black_wins": black_wins,
        "white_wins": white_wins,
        "draws": draws,
    }


def _parse_args(argv: list) -> tuple:
    num_games = 100
    seed: Optional[int] = None
    if len(argv) >= 2:
        num_games = int(argv[1])
    if len(argv) >= 3:
        seed = int(argv[2])
    return num_games, seed


def main(argv: Optional[list] = None) -> None:
    if argv is None:
        argv = sys.argv
    num_games, seed = _parse_args(argv)
    stats = run_random_games(num_games=num_games, seed=seed)
    print(f"总局数         : {stats['total']}")
    print(f"黑棋(BLACK)胜  : {stats['black_wins']}")
    print(f"白棋(WHITE)胜  : {stats['white_wins']}")
    print(f"平局           : {stats['draws']}")


if __name__ == "__main__":
    main()
