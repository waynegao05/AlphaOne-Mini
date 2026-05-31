"""Lightweight tactical-player evaluation entry point."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

from evaluate.players import RandomPlayer
from game.board import BLACK, BOARD_SIZE, WHITE, Board
from game.encoder import index_to_action
from game.rules_basic import _find_any_winner, check_winner, is_game_over
from game.rules_forbidden import get_game_result_forbidden


@dataclass
class TacticalEvalResult:
    winner: int
    tactical_color: int
    num_moves: int
    reason: str


def _build_player(kind: str, rule_mode: str, num_simulations: int, device: str):
    if kind == "tactical":
        from engine.tactical_player import TacticalPlayer

        return TacticalPlayer(rule_mode=rule_mode, name="TacticalPlayer")
    if kind == "hybrid":
        from engine.hybrid_player import HybridPlayer
        from model.policy_value_net import PolicyValueNet

        model = PolicyValueNet()
        model.eval()
        return HybridPlayer(
            model=model,
            num_simulations=num_simulations,
            device=device,
            rule_mode=rule_mode,
            name="HybridPlayer",
        )
    raise ValueError(f"unknown --player: {kind!r}")


def _game_status(board: Board, rule_mode: str) -> tuple[bool, int, str]:
    if rule_mode == "forbidden":
        result = get_game_result_forbidden(board, board.last_move)
        if result.is_over:
            winner = 0 if result.winner is None else int(result.winner)
            return True, winner, result.reason
        return False, 0, "ongoing"

    if is_game_over(board, board.last_move):
        winner = check_winner(board, board.last_move)
        if winner == 0:
            winner = _find_any_winner(board)
        if winner == BLACK:
            return True, BLACK, "black_win"
        if winner == WHITE:
            return True, WHITE, "white_win"
        return True, 0, "draw"
    return False, 0, "ongoing"


def play_eval_game(
    tactical_player,
    random_player,
    tactical_color: int,
    rule_mode: str,
    max_moves: int = BOARD_SIZE * BOARD_SIZE,
) -> TacticalEvalResult:
    board = Board()
    players = {
        tactical_color: tactical_player,
        -tactical_color: random_player,
    }
    move_count = 0
    reason = "max_moves"
    winner = 0

    while move_count < max_moves:
        over, winner, reason = _game_status(board, rule_mode)
        if over:
            break
        player = players[board.current_player]
        action = player.select_action(board)
        if action is None:
            reason = "no_legal_moves"
            winner = 0
            break
        x, y = index_to_action(int(action), BOARD_SIZE)
        if not board.is_legal_move(x, y):
            raise RuntimeError(
                f"{getattr(player, 'name', '?')} returned illegal action {action}"
            )
        board.place_stone(x, y)
        move_count += 1

    if move_count >= max_moves:
        over, winner, reason = _game_status(board, rule_mode)
        if not over:
            winner = 0
            reason = "max_moves"

    return TacticalEvalResult(
        winner=int(winner),
        tactical_color=int(tactical_color),
        num_moves=move_count,
        reason=reason,
    )


def run_tactical_evaluation(
    games: int,
    rule_mode: str,
    player_kind: str,
    num_simulations: int,
    device: str,
    max_moves: int,
    seed: Optional[int] = 0,
) -> dict:
    tactical = _build_player(player_kind, rule_mode, num_simulations, device)
    random_player = RandomPlayer(seed=seed, name="RandomPlayer")

    results: list[TacticalEvalResult] = []
    for index in range(games):
        tactical_color = BLACK if index % 2 == 0 else WHITE
        results.append(
            play_eval_game(
                tactical,
                random_player,
                tactical_color=tactical_color,
                rule_mode=rule_mode,
                max_moves=max_moves,
            )
        )

    tactical_wins = sum(1 for result in results if result.winner == result.tactical_color)
    random_wins = sum(1 for result in results if result.winner == -result.tactical_color)
    draws = sum(1 for result in results if result.winner == 0)
    black_wins = sum(1 for result in results if result.winner == BLACK)
    white_wins = sum(1 for result in results if result.winner == WHITE)
    avg_moves = sum(result.num_moves for result in results) / max(1, len(results))
    return {
        "games": games,
        "rule_mode": rule_mode,
        "player": player_kind,
        "tactical_wins": tactical_wins,
        "random_wins": random_wins,
        "draws": draws,
        "black_wins": black_wins,
        "white_wins": white_wins,
        "tactical_win_rate": tactical_wins / max(1, games),
        "avg_moves": avg_moves,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Tactical/Hybrid player vs RandomPlayer")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--rule-mode", choices=["basic", "forbidden"], default="basic")
    parser.add_argument("--player", choices=["tactical", "hybrid"], default="tactical")
    parser.add_argument("--num-simulations", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-moves", type=int, default=BOARD_SIZE * BOARD_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    summary = run_tactical_evaluation(
        games=args.games,
        rule_mode=args.rule_mode,
        player_kind=args.player,
        num_simulations=args.num_simulations,
        device=args.device,
        max_moves=args.max_moves,
        seed=args.seed,
    )
    print(f"player: {summary['player']} vs RandomPlayer")
    print(f"rule_mode: {summary['rule_mode']}")
    print(f"games: {summary['games']}")
    print(
        f"tactical wins: {summary['tactical_wins']} "
        f"({summary['tactical_win_rate']:.1%})"
    )
    print(f"random wins: {summary['random_wins']} | draws: {summary['draws']}")
    print(f"black wins: {summary['black_wins']} | white wins: {summary['white_wins']}")
    print(f"avg moves: {summary['avg_moves']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
