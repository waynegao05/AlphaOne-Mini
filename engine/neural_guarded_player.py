"""Inference-layer ensemble player with tactical guardrails.

This player does not train models. It combines deterministic tactical checks,
specialized neural checkpoints, a v2 neural checkpoint, and a HybridPlayer
fallback into one `select_action(board) -> action_index` interface.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from game.board import BOARD_SIZE, Board
from game.encoder import action_to_index, encode_board, index_to_action
from model.checkpoint import load_checkpoint
from model.model_factory import create_model
from model.policy_value_net import PolicyValueNet

from .heuristic import score_moves
from .hybrid_player import HybridPlayer
from .tactical_player import TacticalPlayer
from .threats import (
    find_blocked_four_moves,
    find_immediate_blocking_moves,
    find_immediate_winning_moves,
    find_open_four_moves,
    find_open_three_moves,
    is_forbidden_action,
)


@dataclass
class NeuralPolicyChoice:
    action: Optional[int]
    entropy: float
    margin: float
    value: float
    top_actions: list[int]
    source: str


class NeuralGuardedPlayer:
    """Tactical guardrail + neural checkpoint ensemble + Hybrid fallback."""

    def __init__(
        self,
        tactical_checkpoint: str | None = None,
        v2_checkpoint: str | None = None,
        *,
        tactical_model: torch.nn.Module | None = None,
        v2_model: torch.nn.Module | None = None,
        hybrid_player=None,
        device: str = "cpu",
        rule_mode: str = "basic",
        num_simulations: int = 50,
        entropy_threshold: float = 2.5,
        margin_threshold: float = 0.05,
        fallback_mode: str = "normal",
        use_hybrid_fallback: bool = True,
        use_tactical_guardrail: bool = True,
        use_tactical_fallback: bool = True,
        enable_tactical_specialist: bool = True,
        enable_v2_policy: bool = True,
        enable_decision_log: bool = False,
        candidate_radius: int = 2,
        max_candidates: int = 96,
        name: str = "NeuralGuardedPlayer",
    ) -> None:
        self.name = name
        self.device = str(device)
        self.rule_mode = rule_mode
        self.num_simulations = int(num_simulations)
        self.entropy_threshold = float(entropy_threshold)
        self.margin_threshold = float(margin_threshold)
        valid_fallback_modes = {"off", "conservative", "normal", "aggressive"}
        if fallback_mode not in valid_fallback_modes:
            raise ValueError(
                f"fallback_mode must be one of {sorted(valid_fallback_modes)}, got {fallback_mode!r}"
            )
        self.fallback_mode = fallback_mode
        self.use_hybrid_fallback = bool(use_hybrid_fallback)
        self.use_tactical_guardrail = bool(use_tactical_guardrail)
        self.use_tactical_fallback = bool(use_tactical_fallback)
        self.enable_tactical_specialist = bool(enable_tactical_specialist)
        self.enable_v2_policy = bool(enable_v2_policy)
        self.enable_decision_log = bool(enable_decision_log)
        self.candidate_radius = int(candidate_radius)
        self.max_candidates = int(max_candidates)
        self.decision_reason: str | None = None
        self.last_decision: dict = {}
        self.decision_log: list[dict] = []
        self.tactical = TacticalPlayer(
            rule_mode=rule_mode,
            candidate_radius=candidate_radius,
            max_candidates=max_candidates,
            name=f"{name}_tactical_guard",
        )

        self.tactical_model = tactical_model or self._load_model(tactical_checkpoint)
        self.v2_model = v2_model or self._load_model(v2_checkpoint)
        self.hybrid_player = hybrid_player
        if self.hybrid_player is None and self.use_hybrid_fallback:
            self.hybrid_player = HybridPlayer(
                model=PolicyValueNet(),
                num_simulations=max(1, min(self.num_simulations, 50)),
                device=self.device,
                rule_mode=rule_mode,
                candidate_radius=candidate_radius,
                max_candidates=max_candidates,
                name=f"{name}_hybrid_fallback",
            )

        for model in (self.tactical_model, self.v2_model):
            if model is not None:
                model.to(self.device)
                model.eval()

    def _load_model(self, checkpoint_path: str | None) -> torch.nn.Module | None:
        if not checkpoint_path:
            return None
        if not os.path.exists(checkpoint_path):
            return None
        model = create_model("advanced")
        load_checkpoint(model, checkpoint_path, device=self.device)
        model.eval()
        return model

    def _new_log_entry(self, board: Board, color: int) -> dict:
        return {
            "move_index": int(board.move_count),
            "current_player": int(color),
            "selected_action": None,
            "decision_reason": None,
            "guardrail_candidate_actions": {},
            "tactical_specialist_top5": [],
            "v2_top5": [],
            "hybrid_action": None,
            "final_action_source": None,
            "policy_entropy": None,
            "top1_top2_margin": None,
            "whether_fallback_used": False,
            "whether_action_legal": None,
            "whether_forbidden_filtered": False,
            "fallback_mode": self.fallback_mode,
        }

    def _record(
        self,
        action: Optional[int],
        reason: str,
        *,
        log_entry: dict | None = None,
        **extra,
    ) -> Optional[int]:
        self.decision_reason = reason
        self.last_decision = {"action": None if action is None else int(action), "reason": reason}
        self.last_decision.update(extra)
        if log_entry is not None:
            log_entry.update(extra)
            log_entry["selected_action"] = None if action is None else int(action)
            log_entry["decision_reason"] = reason
            log_entry["final_action_source"] = extra.get("final_action_source", reason)
            if "entropy" in extra:
                log_entry["policy_entropy"] = extra["entropy"]
            if "margin" in extra:
                log_entry["top1_top2_margin"] = extra["margin"]
            if "top_actions" in extra:
                source = extra.get("neural_source")
                if source == "tactical_specialist":
                    log_entry["tactical_specialist_top5"] = list(extra["top_actions"])
                elif source == "v2_policy":
                    log_entry["v2_top5"] = list(extra["top_actions"])
            if self.enable_decision_log:
                self.decision_log.append(dict(log_entry))
        return None if action is None else int(action)

    def _is_forbidden_for_color(self, board: Board, action: int, color: int) -> bool:
        return self.rule_mode == "forbidden" and is_forbidden_action(
            board, int(action), color, self.rule_mode
        )

    def _legal_for_color(self, board: Board, action: int, color: int) -> bool:
        try:
            x, y = index_to_action(int(action), BOARD_SIZE)
        except ValueError:
            return False
        if not board.is_legal_move(x, y):
            return False
        if self._is_forbidden_for_color(board, int(action), color):
            return False
        return True

    def _filter_legal(
        self,
        board: Board,
        actions: list[int],
        color: int,
        log_entry: dict | None = None,
    ) -> list[int]:
        legal: list[int] = []
        for action in actions:
            action = int(action)
            if self._is_forbidden_for_color(board, action, color):
                if log_entry is not None:
                    log_entry["whether_forbidden_filtered"] = True
                continue
            if self._legal_for_color(board, action, color):
                legal.append(action)
        return legal

    def _pick_best(self, board: Board, actions: list[int], color: int) -> Optional[int]:
        legal = self._filter_legal(board, actions, color)
        if not legal:
            return None
        scored = score_moves(board, legal, color, self.rule_mode)
        return int(scored[0][0]) if scored else int(legal[0])

    def _force_tactical_action(
        self,
        board: Board,
        color: int,
        log_entry: dict | None = None,
    ) -> tuple[Optional[int], Optional[str]]:
        if not self.use_tactical_guardrail:
            return None, None
        groups = [
            (find_immediate_winning_moves(board, color, self.rule_mode), "immediate_win"),
            (find_immediate_blocking_moves(board, color, self.rule_mode), "immediate_block"),
            (find_open_four_moves(board, color, self.rule_mode), "open_four_attack"),
            (find_open_four_moves(board, -color, self.rule_mode), "open_four_defense"),
        ]
        for actions, reason in groups:
            actions = [int(action) for action in actions]
            legal_actions = self._filter_legal(board, actions, color, log_entry)
            if log_entry is not None:
                log_entry["guardrail_candidate_actions"][reason] = legal_actions
            action = self._pick_best(board, legal_actions, color)
            if action is not None:
                return action, reason
        return None, None

    def _has_tactical_context(self, board: Board, color: int) -> bool:
        groups = (
            find_open_four_moves(board, color, self.rule_mode),
            find_open_four_moves(board, -color, self.rule_mode),
            find_blocked_four_moves(board, color, self.rule_mode),
            find_blocked_four_moves(board, -color, self.rule_mode),
            find_open_three_moves(board, color, self.rule_mode),
            find_open_three_moves(board, -color, self.rule_mode),
        )
        return any(self._filter_legal(board, list(actions), color) for actions in groups)

    def _model_choice(
        self,
        model: torch.nn.Module | None,
        board: Board,
        color: int,
        source: str,
        log_entry: dict | None = None,
    ) -> NeuralPolicyChoice:
        if model is None:
            return NeuralPolicyChoice(None, math.inf, 0.0, 0.0, [], source)
        state = encode_board(board, current_player=color)
        tensor = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            outputs = model(tensor)
            if isinstance(outputs, dict):
                logits = outputs["policy_logits"][0]
                value = outputs["value"]
            else:
                logits, value = outputs
                logits = logits[0]
        logits_cpu = logits.detach().cpu().numpy().astype(np.float64)
        legal_actions = []
        for x, y in board.get_legal_moves():
            action = action_to_index(x, y, BOARD_SIZE)
            if self._is_forbidden_for_color(board, action, color):
                if log_entry is not None:
                    log_entry["whether_forbidden_filtered"] = True
                continue
            if self._legal_for_color(board, action, color):
                legal_actions.append(action)
        if not legal_actions:
            return NeuralPolicyChoice(None, math.inf, 0.0, float(value.reshape(-1)[0]), [], source)
        legal_logits = np.asarray([logits_cpu[action] for action in legal_actions], dtype=np.float64)
        probs = F.softmax(torch.from_numpy(legal_logits), dim=0).numpy()
        order = np.argsort(-probs)
        top_actions = [int(legal_actions[int(i)]) for i in order[:5]]
        entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum())
        margin = float(probs[order[0]] - probs[order[1]]) if len(order) > 1 else 1.0
        if log_entry is not None:
            key = "tactical_specialist_top5" if source == "tactical_specialist" else "v2_top5"
            log_entry[key] = list(top_actions)
        return NeuralPolicyChoice(
            action=top_actions[0] if top_actions else None,
            entropy=entropy,
            margin=margin,
            value=float(value.detach().cpu().numpy().reshape(-1)[0]),
            top_actions=top_actions,
            source=source,
        )

    def _fallback_reason(self, reason: str, extra: dict) -> str:
        entropy = extra.get("entropy")
        margin = extra.get("margin")
        if self.fallback_mode in {"normal", "aggressive"} and entropy is not None and float(entropy) > self.entropy_threshold:
            return "high_entropy_fallback"
        if self.fallback_mode in {"normal", "aggressive"} and margin is not None and float(margin) < self.margin_threshold:
            return "low_margin_fallback"
        if "invalid" in reason or "unavailable" in reason:
            return "illegal_neural_fallback"
        return "hybrid_fallback"

    def _first_legal_fallback(self, board: Board, color: int) -> Optional[int]:
        center = action_to_index(BOARD_SIZE // 2, BOARD_SIZE // 2, BOARD_SIZE)
        if self._legal_for_color(board, center, color):
            return center
        for x, y in board.get_legal_moves():
            action = action_to_index(x, y, BOARD_SIZE)
            if self._legal_for_color(board, action, color):
                return action
        return None

    def _fallback(
        self,
        board: Board,
        color: int,
        reason: str,
        *,
        log_entry: dict | None = None,
        **extra,
    ) -> Optional[int]:
        decision_reason = self._fallback_reason(reason, extra)
        allow_hybrid = (
            self.hybrid_player is not None
            and self.use_hybrid_fallback
            and self.fallback_mode != "off"
        )
        if allow_hybrid:
            try:
                action = self.hybrid_player.select_action(board)
            except Exception:
                action = None
            if log_entry is not None:
                log_entry["hybrid_action"] = None if action is None else int(action)
            if action is not None and self._legal_for_color(board, int(action), color):
                if log_entry is not None:
                    log_entry["whether_fallback_used"] = True
                    log_entry["whether_action_legal"] = True
                return self._record(
                    int(action),
                    decision_reason,
                    log_entry=log_entry,
                    fallback_reason=reason,
                    final_action_source="hybrid_fallback",
                    **extra,
                )
        action = self.tactical.select_action(board) if self.use_tactical_fallback else None
        if action is not None and self._legal_for_color(board, int(action), color):
            if log_entry is not None:
                log_entry["whether_fallback_used"] = True
                log_entry["whether_action_legal"] = True
            return self._record(
                int(action),
                decision_reason,
                log_entry=log_entry,
                fallback_reason=reason,
                final_action_source="tactical_fallback",
                **extra,
            )
        action = self._first_legal_fallback(board, color)
        if action is not None:
            if log_entry is not None:
                log_entry["whether_fallback_used"] = True
                log_entry["whether_action_legal"] = True
            return self._record(
                int(action),
                decision_reason,
                log_entry=log_entry,
                fallback_reason=reason,
                final_action_source="first_legal_fallback",
                **extra,
            )
        if log_entry is not None:
            log_entry["whether_action_legal"] = False
        return self._record(
            None,
            "no_legal_action",
            log_entry=log_entry,
            fallback_reason=reason,
            **extra,
        )

    def _should_fallback(self, choice: NeuralPolicyChoice) -> bool:
        if choice.action is None:
            return True
        if self.fallback_mode in {"off", "conservative"}:
            return False
        if choice.entropy > self.entropy_threshold:
            return True
        if choice.margin < self.margin_threshold:
            return True
        return False

    def select_action(self, board: Board) -> Optional[int]:
        if not board.get_legal_moves():
            return self._record(None, "no_legal_action")
        color = int(board.current_player)
        log_entry = self._new_log_entry(board, color)

        forced, reason = self._force_tactical_action(board, color, log_entry)
        if forced is not None:
            return self._record(
                forced,
                reason or "tactical_guardrail",
                log_entry=log_entry,
                final_action_source="tactical_guardrail",
                whether_action_legal=True,
            )

        tactical_context = self._has_tactical_context(board, color)
        if tactical_context and self.enable_tactical_specialist and self.tactical_model is not None:
            choice = self._model_choice(
                self.tactical_model,
                board,
                color,
                "tactical_specialist",
                log_entry,
            )
            if choice.action is not None and self._legal_for_color(board, choice.action, color):
                if not self._should_fallback(choice):
                    return self._record(
                        choice.action,
                        "tactical_specialist",
                        log_entry=log_entry,
                        entropy=choice.entropy,
                        margin=choice.margin,
                        value=choice.value,
                        top_actions=choice.top_actions,
                        neural_source="tactical_specialist",
                        final_action_source="tactical_specialist",
                        whether_action_legal=True,
                    )
                if self.use_hybrid_fallback:
                    return self._fallback(
                        board,
                        color,
                        "tactical_specialist_uncertain",
                        log_entry=log_entry,
                        entropy=choice.entropy,
                        margin=choice.margin,
                        top_actions=choice.top_actions,
                        neural_source="tactical_specialist",
                    )

        if self.enable_v2_policy:
            choice = self._model_choice(self.v2_model, board, color, "v2_policy", log_entry)
            if choice.action is not None and self._legal_for_color(board, choice.action, color):
                if not self._should_fallback(choice):
                    return self._record(
                        choice.action,
                        "v2_policy",
                        log_entry=log_entry,
                        entropy=choice.entropy,
                        margin=choice.margin,
                        value=choice.value,
                        top_actions=choice.top_actions,
                        neural_source="v2_policy",
                        final_action_source="v2_policy",
                        whether_action_legal=True,
                    )
                if self.use_hybrid_fallback:
                    return self._fallback(
                        board,
                        color,
                        "v2_policy_uncertain",
                        log_entry=log_entry,
                        entropy=choice.entropy,
                        margin=choice.margin,
                        top_actions=choice.top_actions,
                        neural_source="v2_policy",
                    )
        return self._fallback(board, color, "neural_invalid_or_unavailable", log_entry=log_entry)

    def select_move(self, board: Board):
        action = self.select_action(board)
        if action is None:
            return None
        return index_to_action(action, BOARD_SIZE)


__all__ = ["NeuralGuardedPlayer", "NeuralPolicyChoice"]
