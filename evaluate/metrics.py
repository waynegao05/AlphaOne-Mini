"""评估指标统计与晋级判断。

围绕 :class:`evaluate.arena.GameResult` 提供：

- :func:`summarize_results`     : 汇总成统计字典。
- :func:`compute_win_rate`      : 指定玩家名的胜率(跨黑 / 白席位)。
- :func:`compute_draw_rate`     : 平局率。
- :func:`should_promote`        : 给定 ``candidate_win_rate`` 与阈值返回是否晋级。
- :func:`format_match_summary`  : 把 summary 字典格式化成给人看的字符串。

注意：``player_a`` 的胜率 **不是** 黑胜率：玩家可能在不同局执不同颜色，
本模块严格按 ``GameResult`` 中的 ``black_player_name`` / ``white_player_name``
与 ``winner`` 来归属胜场。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from game.board import BLACK, WHITE

from .arena import GameResult


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _safe_div(num: float, denom: float) -> float:
    return float(num) / float(denom) if denom else 0.0


# ---------------------------------------------------------------------------
# 主要 API
# ---------------------------------------------------------------------------
def summarize_results(
    results: Sequence[GameResult],
    player_a_name: Optional[str] = None,
    player_b_name: Optional[str] = None,
) -> Dict[str, Any]:
    """根据 ``results`` 汇总各项统计。"""
    total = len(results)

    black_wins = sum(1 for r in results if r.winner == BLACK)
    white_wins = sum(1 for r in results if r.winner == WHITE)
    draws = sum(1 for r in results if r.winner == 0)

    player_a_wins = 0
    player_b_wins = 0
    player_a_games = 0
    player_b_games = 0

    if player_a_name is not None:
        for r in results:
            if r.black_player_name == player_a_name or r.white_player_name == player_a_name:
                player_a_games += 1
            if r.winner == BLACK and r.black_player_name == player_a_name:
                player_a_wins += 1
            elif r.winner == WHITE and r.white_player_name == player_a_name:
                player_a_wins += 1

    if player_b_name is not None:
        for r in results:
            if r.black_player_name == player_b_name or r.white_player_name == player_b_name:
                player_b_games += 1
            if r.winner == BLACK and r.black_player_name == player_b_name:
                player_b_wins += 1
            elif r.winner == WHITE and r.white_player_name == player_b_name:
                player_b_wins += 1

    avg_moves = _safe_div(sum(r.num_moves for r in results), total)

    summary: Dict[str, Any] = {
        "total_games": total,
        "black_wins": black_wins,
        "white_wins": white_wins,
        "draws": draws,
        "draw_rate": _safe_div(draws, total),
        "avg_moves": avg_moves,
        "player_a_name": player_a_name,
        "player_b_name": player_b_name,
        "player_a_wins": player_a_wins,
        "player_b_wins": player_b_wins,
        "player_a_games": player_a_games,
        "player_b_games": player_b_games,
        "player_a_win_rate": _safe_div(player_a_wins, total),
        "player_b_win_rate": _safe_div(player_b_wins, total),
    }
    return summary


def compute_win_rate(
    results: Sequence[GameResult], target_player_name: str
) -> float:
    """``target_player_name`` 在所有 ``results`` 中的胜率(跨黑 / 白席位)。"""
    if not results:
        return 0.0
    wins = 0
    for r in results:
        if r.winner == BLACK and r.black_player_name == target_player_name:
            wins += 1
        elif r.winner == WHITE and r.white_player_name == target_player_name:
            wins += 1
    return wins / len(results)


def compute_draw_rate(results: Sequence[GameResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.winner == 0) / len(results)


def should_promote(
    candidate_win_rate: float, threshold: float = 0.55
) -> bool:
    """胜率 >= 阈值即可晋级。"""
    return float(candidate_win_rate) >= float(threshold)


def format_match_summary(summary: Dict[str, Any]) -> str:
    """把 :func:`summarize_results` 的输出格式化成多行字符串。"""
    lines: List[str] = []
    lines.append(f"对局总数 : {summary.get('total_games', 0)}")
    lines.append(
        f"黑胜 : {summary.get('black_wins', 0)} | "
        f"白胜 : {summary.get('white_wins', 0)} | "
        f"平局 : {summary.get('draws', 0)}"
    )
    pa = summary.get("player_a_name")
    pb = summary.get("player_b_name")
    if pa is not None:
        lines.append(
            f"{pa}: {summary.get('player_a_wins', 0)} 胜 / "
            f"{summary.get('total_games', 0)} 盘 "
            f"(胜率 {summary.get('player_a_win_rate', 0.0) * 100:.1f}%)"
        )
    if pb is not None:
        lines.append(
            f"{pb}: {summary.get('player_b_wins', 0)} 胜 / "
            f"{summary.get('total_games', 0)} 盘 "
            f"(胜率 {summary.get('player_b_win_rate', 0.0) * 100:.1f}%)"
        )
    lines.append(
        f"平局率 : {summary.get('draw_rate', 0.0) * 100:.1f}% | "
        f"平均手数 : {summary.get('avg_moves', 0.0):.1f}"
    )
    return "\n".join(lines)


__all__ = [
    "summarize_results",
    "compute_win_rate",
    "compute_draw_rate",
    "should_promote",
    "format_match_summary",
]
