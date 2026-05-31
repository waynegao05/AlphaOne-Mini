"""模型 / 玩家对战 Arena。

本模块只做"两个玩家下一盘对局"这一件事，不涉及训练 / self-play 数据生成。
对外接口：

- :class:`GameResult`        : 单盘对局结果。
- :class:`Arena`             : 持有黑 / 白玩家，提供 ``play_one_game`` / ``play_many_games``。
- :func:`run_match`           : 跑两名玩家的 Match(可交换执黑)，返回 ``(summary, results)``。

玩家约定：必须实现 ``select_action(board) -> int | None`` 且暴露 ``name`` 属性。
``select_action`` 返回 None 表示无合法动作，会被 Arena 当作平局结束。
非法 action_index 会触发 ``RuntimeError`` —— Arena 不静默吞异常。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from game.board import BLACK, BOARD_SIZE, WHITE, Board
from game.encoder import action_to_index, index_to_action
from game.rules_basic import _find_any_winner, check_winner, is_game_over


# ---------------------------------------------------------------------------
# 结果数据结构
# ---------------------------------------------------------------------------
@dataclass
class GameResult:
    """单盘对局结果。

    - ``winner``           : ``BLACK(1)`` / ``WHITE(-1)`` / ``0`` (平局)
    - ``black_player_name``: 该盘执黑玩家的 ``name``
    - ``white_player_name``: 该盘执白玩家的 ``name``
    - ``num_moves``        : 实际落子手数
    - ``moves``            : 按顺序记录的 action_index 列表
    - ``reason``           : ``"black_win"`` / ``"white_win"`` / ``"draw"`` / ``"max_moves"``
    """

    winner: int
    black_player_name: str
    white_player_name: str
    num_moves: int
    moves: List[int] = field(default_factory=list)
    reason: str = "draw"


# ---------------------------------------------------------------------------
# Arena
# ---------------------------------------------------------------------------
class Arena:
    """两个玩家对战的最小 Arena。"""

    def __init__(
        self,
        player_black,
        player_white,
        board_size: int = BOARD_SIZE,
        max_moves: int = BOARD_SIZE * BOARD_SIZE,
    ) -> None:
        self.player_black = player_black
        self.player_white = player_white
        self.board_size = int(board_size)
        self.max_moves = int(max_moves)

    # ---- 单盘 -------------------------------------------------------------
    def play_one_game(self) -> GameResult:
        return self._play(self.player_black, self.player_white)

    # ---- 多盘 -------------------------------------------------------------
    def play_many_games(
        self,
        num_games: int,
        alternate_sides: bool = True,
    ) -> List[GameResult]:
        """跑 ``num_games`` 盘；``alternate_sides=True`` 时奇数局交换黑白。"""
        if num_games <= 0:
            return []

        results: List[GameResult] = []
        for i in range(num_games):
            if alternate_sides and i % 2 == 1:
                black = self.player_white
                white = self.player_black
            else:
                black = self.player_black
                white = self.player_white
            results.append(self._play(black, white))
        return results

    # ---- 内部：核心对局循环 ----------------------------------------------
    def _play(self, player_black, player_white) -> GameResult:
        board = Board()
        moves: List[int] = []
        move_number = 0

        black_name = getattr(player_black, "name", "black")
        white_name = getattr(player_white, "name", "white")

        while move_number < self.max_moves:
            if is_game_over(board, board.last_move):
                break

            current_player = (
                player_black if board.current_player == BLACK else player_white
            )

            action = current_player.select_action(board)
            if action is None:
                # 玩家声明没有合法动作可下，结束对局
                break

            action = int(action)
            if not (0 <= action < self.board_size * self.board_size):
                raise RuntimeError(
                    f"玩家 {getattr(current_player, 'name', '?')} "
                    f"返回非法 action_index: {action}"
                )

            x, y = index_to_action(action, self.board_size)
            if not board.is_legal_move(x, y):
                raise RuntimeError(
                    f"玩家 {getattr(current_player, 'name', '?')} "
                    f"选了已占用 / 越界的位置: ({x}, {y}) -> action {action}"
                )

            board.place_stone(x, y)
            moves.append(action)
            move_number += 1

        # 终局判断
        winner = check_winner(board, board.last_move)
        if winner == 0:
            winner = _find_any_winner(board)

        if winner == BLACK:
            reason = "black_win"
        elif winner == WHITE:
            reason = "white_win"
        elif board.move_count >= self.board_size * self.board_size:
            reason = "draw"
        elif move_number >= self.max_moves:
            reason = "max_moves"
        else:
            reason = "draw"

        return GameResult(
            winner=int(winner),
            black_player_name=black_name,
            white_player_name=white_name,
            num_moves=move_number,
            moves=moves,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# Match：两名玩家的多盘对战
# ---------------------------------------------------------------------------
def run_match(
    player_a,
    player_b,
    num_games: int,
    alternate_sides: bool = True,
    board_size: int = BOARD_SIZE,
    max_moves: int = BOARD_SIZE * BOARD_SIZE,
) -> Tuple[Dict[str, Any], List[GameResult]]:
    """跑 ``player_a`` vs ``player_b``，返回 ``(summary, results)``。

    - ``alternate_sides=True``：偶数局 ``player_a`` 执黑，奇数局执白。
    - ``summary`` 是 :func:`evaluate.metrics.summarize_results` 的输出。
    """
    # 延迟导入避免本模块与 metrics 互相依赖。
    from evaluate.metrics import summarize_results

    arena = Arena(
        player_black=player_a,
        player_white=player_b,
        board_size=board_size,
        max_moves=max_moves,
    )
    results = arena.play_many_games(num_games, alternate_sides=alternate_sides)
    summary = summarize_results(
        results,
        player_a_name=getattr(player_a, "name", "player_a"),
        player_b_name=getattr(player_b, "name", "player_b"),
    )
    return summary, results


__all__ = ["GameResult", "Arena", "run_match"]
