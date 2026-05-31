"""AlphaOne-Mini / StrongPlayer: tactical + VCF/VCT + MCTS fallback.

Decision priority (top wins, never falls through to a weaker tier without
exhausting the stronger one):

1. **Direct win** — if any legal move creates an immediate five for us, take it.
2. **Block direct win** — if opponent has an immediate five, block it.
3. **Our VCF mate** — :func:`vcf_first_move` proves a forced continuous-four
   win; play its first move.
4. **Defend opponent VCF mate** — if opponent has a VCF mate, try cells that
   destroy it (preempt + counter-threat).
5. **Our open-four** — make our own open-four (forces opponent to defend).
6. **Block opponent open-four** — must block.
7. **MCTS / neural fallback** — defer to the underlying MCTS player.
8. **Tactical fallback** — if MCTS errors out, fall back to the rule-based
   TacticalPlayer.

This composition does NOT touch any existing module's behaviour; it only adds
a new player class that calls them.

The implementation also includes bounded VCT, one-ply threat safety, and
candidate-pruned opponent lookahead tiers added after the original docstring.
Threat safety intentionally runs before lookahead because avoiding a known
opponent escalation is more reliable than following a speculative line.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from game.board import BOARD_SIZE, Board
from game.encoder import index_to_action

from .candidate_moves import generate_candidate_moves
from .simulation import temporary_stone
from .tactical_player import TacticalPlayer
from .opponent_lookahead import select_lookahead_move
from .threats import (
    find_immediate_blocking_moves,
    find_immediate_winning_moves,
    find_open_four_moves,
    is_forbidden_action,
)
from .threat_safety import select_threat_safe_move
from .vcf_search import vcf_defends, vcf_first_move
from .vct_search import vct_defends, vct_first_move


class StrongPlayer:
    """Composite high-strength player used publicly as AlphaOne-Mini.

    Parameters
    ----------
    mcts_player : optional
        An already-constructed player that supports ``select_action(board) ->
        Optional[int]``. If absent, ``model`` must be provided so we can build a
        :class:`evaluate.players.ModelMCTSPlayer` internally.
    model : optional
        :class:`model.policy_value_net.PolicyValueNet`-style model.
    num_simulations : int
        MCTS sims when we build the MCTS player internally. Default is heavier
        than the GUI default (600) because StrongPlayer is meant for real play.
    c_puct, device : usual MCTS knobs.
    rule_mode : ``"basic"`` or ``"forbidden"``.
    vcf_depth : VCF half-move depth (9 ≈ 5 attacker moves).
    vcf_defense_depth : VCF half-move depth used when defending. Slightly less
        than ``vcf_depth`` to keep defensive search responsive.
    vcf_node_budget : hard cap on VCF search nodes.
    """

    def __init__(
        self,
        mcts_player=None,
        model=None,
        num_simulations: int = 600,
        c_puct: float = 5.0,
        device: str = "cpu",
        rule_mode: str = "basic",
        vcf_depth: int = 9,
        vcf_defense_depth: int = 7,
        vcf_node_budget: int = 20000,
        vct_depth: int = 7,
        vct_node_budget: int = 20000,
        lookahead_depth: int = 4,
        lookahead_branch_factor: int = 3,
        candidate_radius: int = 2,
        max_candidates: int = 80,
        name: str = "AlphaOne-Mini",
    ) -> None:
        self.name = name
        self.rule_mode = rule_mode
        self.vcf_depth = int(vcf_depth)
        self.vcf_defense_depth = int(vcf_defense_depth)
        self.vcf_node_budget = int(vcf_node_budget)
        self.vct_depth = int(vct_depth)
        self.vct_node_budget = int(vct_node_budget)
        self.lookahead_depth = int(lookahead_depth)
        self.lookahead_branch_factor = int(lookahead_branch_factor)
        self.candidate_radius = int(candidate_radius)
        self.max_candidates = int(max_candidates)
        self.last_decision_reason = "-"
        self.decision_reason = "-"

        self.tactical = TacticalPlayer(
            name=f"{name}_tactical",
            rule_mode=rule_mode,
            candidate_radius=candidate_radius,
            max_candidates=max_candidates,
        )

        if mcts_player is None and model is not None:
            from evaluate.players import ModelMCTSPlayer

            mcts_player = ModelMCTSPlayer(
                model=model,
                num_simulations=num_simulations,
                c_puct=c_puct,
                device=device,
                board_size=BOARD_SIZE,
                name=f"{name}_mcts",
            )
        self.mcts_player = mcts_player

    def _return(self, action: Optional[int], reason: str) -> Optional[int]:
        self.last_decision_reason = reason
        self.decision_reason = reason
        return None if action is None else int(action)

    # ------------------------------------------------------------------
    # legality helpers
    # ------------------------------------------------------------------
    def _legal(self, board: Board, action: int, color: int) -> bool:
        try:
            x, y = index_to_action(int(action), BOARD_SIZE)
        except ValueError:
            return False
        if not board.is_legal_move(x, y):
            return False
        if self.rule_mode == "forbidden" and is_forbidden_action(
            board, int(action), color, self.rule_mode
        ):
            return False
        return True

    def _filter(self, board: Board, actions: Iterable[int], color: int) -> List[int]:
        return [int(a) for a in actions if self._legal(board, int(a), color)]

    # ------------------------------------------------------------------
    # tier 4 helper: try to destroy opponent's VCF mate
    # ------------------------------------------------------------------
    def _disrupt_opponent_vcf(
        self, board: Board, color: int, opp_mate_first: int
    ) -> Optional[int]:
        """Try a small set of candidate disruptions in priority order.

        Priority:
        a) Play opponent's VCF first-move ourselves (often kills the mate).
        b) Play a move that is one of OUR four-threats — this forces opponent
           to defend instead of running their VCF.
        c) Play a strong heuristic move and check whether the mate disappears.
        """
        # a) preempt
        if self._legal(board, opp_mate_first, color):
            if vcf_defends(
                board,
                color,
                opp_mate_first,
                max_depth=self.vcf_defense_depth,
                rule_mode=self.rule_mode,
                node_budget=self.vcf_node_budget,
            ):
                return int(opp_mate_first)

        # b) counter-threat — try our own four-creators
        own_four = _our_four_creators(
            board, color, self.rule_mode,
            radius=self.candidate_radius, limit=self.max_candidates,
        )
        for action in own_four:
            if not self._legal(board, action, color):
                continue
            if vcf_defends(
                board,
                color,
                action,
                max_depth=self.vcf_defense_depth,
                rule_mode=self.rule_mode,
                node_budget=self.vcf_node_budget,
            ):
                return int(action)

        # c) top heuristic candidate
        try:
            heuristic = self.tactical._pick_best(
                board,
                self._filter(
                    board,
                    generate_candidate_moves(
                        board,
                        radius=self.candidate_radius,
                        max_candidates=self.max_candidates,
                    ),
                    color,
                ),
                color,
            )
        except Exception:
            heuristic = None
        if heuristic is not None and self._legal(board, heuristic, color):
            if vcf_defends(
                board,
                color,
                heuristic,
                max_depth=self.vcf_defense_depth,
                rule_mode=self.rule_mode,
                node_budget=self.vcf_node_budget,
            ):
                return int(heuristic)

        return None

    def _disrupt_opponent_vct(
        self, board: Board, color: int, opp_vct_first: int
    ) -> Optional[int]:
        """Try to destroy opponent's bounded VCT threat by preempting it."""
        if self._legal(board, opp_vct_first, color):
            if vct_defends(
                board,
                color,
                opp_vct_first,
                max_depth=self.vct_depth,
                rule_mode=self.rule_mode,
                node_budget=self.vct_node_budget,
                candidate_radius=self.candidate_radius,
                max_candidates=self.max_candidates,
            ):
                return int(opp_vct_first)
        return None

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------
    def select_action(self, board: Board) -> Optional[int]:
        if not board.get_legal_moves():
            return None
        color = int(board.current_player)

        # Tier 1: immediate win
        wins = self._filter(
            board, find_immediate_winning_moves(board, color, self.rule_mode), color
        )
        if wins:
            return self._return(self.tactical._pick_best(board, wins, color), "immediate_win")

        # Tier 2: block opponent's immediate win
        blocks = self._filter(
            board, find_immediate_blocking_moves(board, color, self.rule_mode), color
        )
        if blocks:
            return self._return(
                self.tactical._pick_best(board, blocks, color),
                "immediate_block",
            )

        # Tier 3: our VCF mate
        mate = vcf_first_move(
            board, color,
            max_depth=self.vcf_depth,
            rule_mode=self.rule_mode,
            node_budget=self.vcf_node_budget,
        )
        if mate is not None and self._legal(board, mate, color):
            return self._return(int(mate), "vcf_attack")

        # Tier 4: defend opponent's VCF mate
        opp_mate = vcf_first_move(
            board, -color,
            max_depth=self.vcf_defense_depth,
            rule_mode=self.rule_mode,
            node_budget=self.vcf_node_budget,
        )
        if opp_mate is not None:
            disruption = self._disrupt_opponent_vcf(board, color, opp_mate)
            if disruption is not None:
                return self._return(int(disruption), "vcf_defense")
            # If we cannot disrupt, we are losing — fall through to the strongest
            # remaining heuristic (still better than a known-loss).

        # Tier 5: defend opponent's bounded VCT.
        opp_vct = vct_first_move(
            board,
            -color,
            max_depth=self.vct_depth,
            rule_mode=self.rule_mode,
            node_budget=self.vct_node_budget,
            candidate_radius=self.candidate_radius,
            max_candidates=self.max_candidates,
        )
        if opp_vct is not None:
            disruption = self._disrupt_opponent_vct(board, color, opp_vct)
            if disruption is not None:
                return self._return(int(disruption), "vct_defense")

        # Tier 6: our VCT threat.
        vct = vct_first_move(
            board,
            color,
            max_depth=self.vct_depth,
            rule_mode=self.rule_mode,
            node_budget=self.vct_node_budget,
            candidate_radius=self.candidate_radius,
            max_candidates=self.max_candidates,
        )
        if vct is not None and self._legal(board, vct, color):
            return self._return(int(vct), "vct_attack")

        # Tier 7: our open-four
        own_o4 = self._filter(
            board, find_open_four_moves(board, color, self.rule_mode), color
        )
        if own_o4:
            return self._return(
                self.tactical._pick_best(board, own_o4, color),
                "open_four_attack",
            )

        # Tier 8: block opponent's open-four
        opp_o4 = self._filter(
            board, find_open_four_moves(board, -color, self.rule_mode), color
        )
        if opp_o4:
            return self._return(
                self.tactical._pick_best(board, opp_o4, color),
                "open_four_defense",
            )

        # Tier 9: one-ply threat safety before speculative lookahead.
        safety = select_threat_safe_move(
            board,
            color,
            rule_mode=self.rule_mode,
            candidate_radius=self.candidate_radius,
            max_candidates=self.max_candidates,
            evaluate_top_n=36,
            max_replies=40,
        )
        if safety is not None and self._legal(board, safety.action, color):
            return self._return(
                int(safety.action),
                f"threat_safety:risk={safety.reply_risk:.0f}",
            )

        # Tier 10: opponent-model lookahead (candidate-pruned 3-4 ply search).
        lookahead = select_lookahead_move(
            board,
            color,
            rule_mode=self.rule_mode,
            depth=self.lookahead_depth,
            branch_factor=self.lookahead_branch_factor,
            candidate_radius=self.candidate_radius,
            max_candidates=min(self.max_candidates, 24),
        )
        if lookahead is not None and self._legal(board, lookahead.action, color):
            return self._return(int(lookahead.action), lookahead.reason)

        # Tier 11: MCTS / neural
        if self.mcts_player is not None:
            try:
                action = self.mcts_player.select_action(board)
            except Exception:
                action = None
            if action is not None:
                action = int(action)
                if self._legal(board, action, color):
                    return self._return(action, "mcts_phase3")

        # Tier 12: tactical fallback
        return self._return(self.tactical.select_action(board), "tactical_fallback")

    def select_move(self, board: Board):
        action = self.select_action(board)
        if action is None:
            return None
        return index_to_action(int(action), BOARD_SIZE)


def _our_four_creators(
    board: Board,
    color: int,
    rule_mode: str,
    radius: int = 2,
    limit: int = 80,
) -> List[int]:
    """Cheap helper: legal moves that create at least one immediate-five threat."""
    from .vcf_search import find_vcf_attack_candidates  # avoid cyclic import noise

    return find_vcf_attack_candidates(
        board, color, rule_mode,
        candidate_radius=radius, max_candidates=limit,
    )


__all__ = ["StrongPlayer"]
