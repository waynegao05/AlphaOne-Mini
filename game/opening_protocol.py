"""Engineering opening protocol state machine for 15x15 Gomoku.

This module models prescribed opening, three-move swap, and fifth-move
N-candidate selection at the protocol layer.  It does not change ``Board`` or
the basic/forbidden rule modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Sequence, Union

from .board import BLACK, BOARD_SIZE, WHITE, Board
from .coordinates import coord_to_index, index_to_coord
from .encoder import action_to_index, index_to_action


PLAYER_A = "player_a"
PLAYER_B = "player_b"
PLAYERS = (PLAYER_A, PLAYER_B)
CENTER_ACTION = action_to_index(7, 7)
CENTER_X = 7
CENTER_Y = 7
BLACK_3_MIN = 5
BLACK_3_MAX = 9

CoordLike = Union[str, int, tuple[int, int]]


class OpeningPhase(str, Enum):
    DESIGNATED_BLACK_1 = "designated_black_1"
    DESIGNATED_WHITE_2 = "designated_white_2"
    DESIGNATED_BLACK_3 = "designated_black_3"
    SWAP_DECISION = "swap_decision"
    NORMAL_WHITE_4 = "normal_white_4"
    BLACK_5_CANDIDATES = "black_5_candidates"
    WHITE_SELECT_BLACK_5 = "white_select_black_5"
    NORMAL_PLAY = "normal_play"
    FINISHED = "finished"


class OpeningProtocolError(ValueError):
    """Protocol-level validation error with phase and player context."""

    def __init__(self, phase: OpeningPhase, player_id: object, reason: str) -> None:
        super().__init__(f"phase={phase.value}, player={player_id!r}: {reason}")
        self.phase = phase
        self.player_id = player_id
        self.reason = reason


@dataclass(frozen=True)
class ProtocolMove:
    phase: OpeningPhase
    player_id: str
    action: int
    x: int
    y: int
    stone_color: int
    note: str = ""


class OpeningProtocol:
    """State machine for prescribed opening, swap, and fifth-move N play."""

    def __init__(
        self,
        board: Board,
        n_for_fifth: int = 2,
        use_designated_opening: bool = True,
        enable_swap: bool = True,
        enable_fifth_n: bool = True,
    ) -> None:
        if n_for_fifth <= 0:
            raise ValueError("n_for_fifth must be positive")
        self.board = board
        self.n_for_fifth = int(n_for_fifth)
        self.use_designated_opening = bool(use_designated_opening)
        self.enable_swap = bool(enable_swap)
        self.enable_fifth_n = bool(enable_fifth_n)
        self.player_to_color: dict[str, int] = {PLAYER_A: BLACK, PLAYER_B: WHITE}
        self.color_to_player: dict[int, str] = {BLACK: PLAYER_A, WHITE: PLAYER_B}
        self.candidate_black5_actions: list[int] = []
        self.swap_decision: Optional[bool] = None
        self.move_history: list[ProtocolMove] = []
        self.protocol_finished = False
        self.phase = (
            OpeningPhase.DESIGNATED_BLACK_1
            if self.use_designated_opening
            else OpeningPhase.NORMAL_PLAY
        )
        if self.phase == OpeningPhase.NORMAL_PLAY:
            self.protocol_finished = True

    def get_phase(self) -> OpeningPhase:
        return self.phase

    def is_protocol_finished(self) -> bool:
        return self.protocol_finished or self.phase == OpeningPhase.NORMAL_PLAY

    def get_current_stone_color(self) -> Optional[int]:
        if self.phase in (
            OpeningPhase.DESIGNATED_BLACK_1,
            OpeningPhase.DESIGNATED_BLACK_3,
            OpeningPhase.BLACK_5_CANDIDATES,
            OpeningPhase.WHITE_SELECT_BLACK_5,
        ):
            return BLACK
        if self.phase in (
            OpeningPhase.DESIGNATED_WHITE_2,
            OpeningPhase.NORMAL_WHITE_4,
        ):
            return WHITE
        if self.phase == OpeningPhase.NORMAL_PLAY:
            return self.board.current_player
        return None

    def get_current_player_id(self) -> Optional[str]:
        if self.phase == OpeningPhase.SWAP_DECISION:
            return PLAYER_B
        if self.phase == OpeningPhase.WHITE_SELECT_BLACK_5:
            return self.color_to_player[WHITE]
        color = self.get_current_stone_color()
        if color in (BLACK, WHITE):
            return self.color_to_player[color]
        return None

    def coord_to_action(self, coord_or_action: CoordLike) -> int:
        try:
            if isinstance(coord_or_action, int):
                index_to_action(coord_or_action, BOARD_SIZE)
                return coord_or_action
            if isinstance(coord_or_action, str):
                x, y = coord_to_index(coord_or_action)
                return action_to_index(x, y)
            if (
                isinstance(coord_or_action, tuple)
                and len(coord_or_action) == 2
                and all(isinstance(v, int) for v in coord_or_action)
            ):
                x, y = coord_or_action
                return action_to_index(x, y)
        except ValueError as exc:
            raise self._error(self.get_current_player_id(), str(exc)) from exc
        raise self._error(
            self.get_current_player_id(), f"unsupported coordinate {coord_or_action!r}"
        )

    def action_to_coord(self, action: int) -> tuple[int, int]:
        try:
            return index_to_action(action, BOARD_SIZE)
        except ValueError as exc:
            raise self._error(self.get_current_player_id(), str(exc)) from exc

    def get_legal_actions_for_phase(self) -> list[int]:
        if self.phase == OpeningPhase.DESIGNATED_BLACK_1:
            return [CENTER_ACTION] if self.board.is_legal_move(CENTER_X, CENTER_Y) else []
        if self.phase == OpeningPhase.DESIGNATED_WHITE_2:
            return self._all_legal_actions()
        if self.phase == OpeningPhase.DESIGNATED_BLACK_3:
            return [
                action_to_index(x, y)
                for x in range(BLACK_3_MIN, BLACK_3_MAX + 1)
                for y in range(BLACK_3_MIN, BLACK_3_MAX + 1)
                if self.board.is_legal_move(x, y)
            ]
        if self.phase == OpeningPhase.NORMAL_WHITE_4:
            return self._all_legal_actions()
        if self.phase == OpeningPhase.BLACK_5_CANDIDATES:
            return self._all_legal_actions()
        if self.phase == OpeningPhase.WHITE_SELECT_BLACK_5:
            return list(self.candidate_black5_actions)
        if self.phase == OpeningPhase.NORMAL_PLAY:
            return self._all_legal_actions()
        return []

    def validate_phase_action(self, player_id: str, action: int) -> None:
        self._ensure_known_player(player_id)
        if player_id != self.get_current_player_id():
            raise self._error(player_id, "player is not allowed to act in this phase")
        if action not in self.get_legal_actions_for_phase():
            coord = self._format_action(action)
            raise self._error(player_id, f"illegal action for phase: {coord}")

    def play_designated_move(self, player_id: str, coord_or_action: CoordLike) -> int:
        if self.phase not in (
            OpeningPhase.DESIGNATED_BLACK_1,
            OpeningPhase.DESIGNATED_WHITE_2,
            OpeningPhase.DESIGNATED_BLACK_3,
        ):
            raise self._error(player_id, "not in designated move phase")
        action = self.coord_to_action(coord_or_action)
        self.validate_phase_action(player_id, action)

        if self.phase == OpeningPhase.DESIGNATED_BLACK_1:
            color = BLACK
            next_phase = OpeningPhase.DESIGNATED_WHITE_2
            note = "black_1"
        elif self.phase == OpeningPhase.DESIGNATED_WHITE_2:
            color = WHITE
            next_phase = OpeningPhase.DESIGNATED_BLACK_3
            note = "white_2"
        else:
            color = BLACK
            next_phase = (
                OpeningPhase.SWAP_DECISION
                if self.enable_swap
                else OpeningPhase.NORMAL_WHITE_4
            )
            note = "black_3"

        self._place_action(player_id, action, color, note)
        self.phase = next_phase
        return action

    def decide_swap(self, player_id: str, swap: bool) -> None:
        if self.phase != OpeningPhase.SWAP_DECISION:
            raise self._error(player_id, "not in swap decision phase")
        if player_id != PLAYER_B:
            raise self._error(player_id, "only original white player can decide swap")

        self.swap_decision = bool(swap)
        if self.swap_decision:
            self.player_to_color = {PLAYER_A: WHITE, PLAYER_B: BLACK}
            self.color_to_player = {BLACK: PLAYER_B, WHITE: PLAYER_A}
        self.phase = OpeningPhase.NORMAL_WHITE_4

    def play_white_4(self, player_id: str, coord_or_action: CoordLike) -> int:
        if self.phase != OpeningPhase.NORMAL_WHITE_4:
            raise self._error(player_id, "not in white 4 phase")
        action = self.coord_to_action(coord_or_action)
        self.validate_phase_action(player_id, action)
        self._place_action(player_id, action, WHITE, "white_4")
        if self.enable_fifth_n:
            self.phase = OpeningPhase.BLACK_5_CANDIDATES
        else:
            self.phase = OpeningPhase.NORMAL_PLAY
            self.protocol_finished = True
        return action

    def submit_black5_candidates(
        self, player_id: str, candidates: Iterable[CoordLike]
    ) -> list[int]:
        if self.phase != OpeningPhase.BLACK_5_CANDIDATES:
            raise self._error(player_id, "not in black 5 candidate phase")
        if player_id != self.color_to_player[BLACK]:
            raise self._error(player_id, "only current black player can submit candidates")

        actions = [self.coord_to_action(candidate) for candidate in candidates]
        if len(actions) != self.n_for_fifth:
            raise self._error(
                player_id,
                f"expected {self.n_for_fifth} black 5 candidates, got {len(actions)}",
            )
        if len(set(actions)) != len(actions):
            raise self._error(player_id, "black 5 candidates must be unique")
        for action in actions:
            x, y = index_to_action(action, BOARD_SIZE)
            if not self.board.is_legal_move(x, y):
                raise self._error(
                    player_id, f"black 5 candidate is not empty: {self._format_action(action)}"
                )

        self.candidate_black5_actions = list(actions)
        self.phase = OpeningPhase.WHITE_SELECT_BLACK_5
        return list(actions)

    def select_black5_candidate(self, player_id: str, candidate: CoordLike) -> int:
        if self.phase != OpeningPhase.WHITE_SELECT_BLACK_5:
            raise self._error(player_id, "not in white select black 5 phase")
        if player_id != self.color_to_player[WHITE]:
            raise self._error(player_id, "only current white player can select black 5")

        action = self.coord_to_action(candidate)
        if action not in self.candidate_black5_actions:
            raise self._error(
                player_id, f"selected action is not a black 5 candidate: {self._format_action(action)}"
            )
        self._place_action(player_id, action, BLACK, "black_5_selected")
        self.phase = OpeningPhase.NORMAL_PLAY
        self.protocol_finished = True
        return action

    def play_normal_move(self, player_id: str, coord_or_action: CoordLike) -> int:
        if self.phase != OpeningPhase.NORMAL_PLAY:
            raise self._error(player_id, "normal play has not started")
        action = self.coord_to_action(coord_or_action)
        self.validate_phase_action(player_id, action)
        color = self.board.current_player
        self._place_action(player_id, action, color, "normal")
        return action

    def _all_legal_actions(self) -> list[int]:
        return [action_to_index(x, y) for x, y in self.board.get_legal_moves()]

    def _ensure_known_player(self, player_id: str) -> None:
        if player_id not in PLAYERS:
            raise self._error(player_id, "unknown player_id")

    def _place_action(self, player_id: str, action: int, color: int, note: str) -> None:
        x, y = index_to_action(action, BOARD_SIZE)
        if not self.board.is_legal_move(x, y):
            raise self._error(player_id, f"position is occupied or out of bounds: {self._format_action(action)}")
        if self.board.current_player != color:
            raise self._error(
                player_id,
                f"board.current_player={self.board.current_player} does not match required stone_color={color}",
            )
        self.board.place_stone(x, y)
        self.move_history.append(
            ProtocolMove(
                phase=self.phase,
                player_id=player_id,
                action=action,
                x=x,
                y=y,
                stone_color=color,
                note=note,
            )
        )

    def _format_action(self, action: int) -> str:
        x, y = index_to_action(action, BOARD_SIZE)
        return f"{index_to_coord(x, y)}({action})"

    def _error(self, player_id: object, reason: str) -> OpeningProtocolError:
        return OpeningProtocolError(self.phase, player_id, reason)


__all__ = [
    "OpeningPhase",
    "OpeningProtocol",
    "OpeningProtocolError",
    "ProtocolMove",
    "PLAYER_A",
    "PLAYER_B",
    "CENTER_ACTION",
]
