"""Tests for prescribed opening, swap, and fifth-move N protocol."""

from __future__ import annotations

import pytest

from game.board import BLACK, WHITE, Board
from game.encoder import action_to_index
from game.opening_protocol import (
    CENTER_ACTION,
    PLAYER_A,
    PLAYER_B,
    OpeningPhase,
    OpeningProtocol,
    OpeningProtocolError,
)


def _new_protocol(n_for_fifth: int = 2) -> OpeningProtocol:
    return OpeningProtocol(Board(), n_for_fifth=n_for_fifth)


def _after_black1() -> OpeningProtocol:
    protocol = _new_protocol()
    protocol.play_designated_move(PLAYER_A, "H8")
    return protocol


def _after_white2() -> OpeningProtocol:
    protocol = _after_black1()
    protocol.play_designated_move(PLAYER_B, "A1")
    return protocol


def _after_black3() -> OpeningProtocol:
    protocol = _after_white2()
    protocol.play_designated_move(PLAYER_A, "I8")
    return protocol


def _after_swap_decision(swap: bool = False) -> OpeningProtocol:
    protocol = _after_black3()
    protocol.decide_swap(PLAYER_B, swap=swap)
    return protocol


def _after_white4(swap: bool = False) -> OpeningProtocol:
    protocol = _after_swap_decision(swap=swap)
    white_player = PLAYER_A if swap else PLAYER_B
    protocol.play_white_4(white_player, "O15")
    return protocol


def _after_black5(swap: bool = False) -> OpeningProtocol:
    protocol = _after_white4(swap=swap)
    black_player = PLAYER_B if swap else PLAYER_A
    white_player = PLAYER_A if swap else PLAYER_B
    protocol.submit_black5_candidates(black_player, ["B2", "C2"])
    protocol.select_black5_candidate(white_player, "B2")
    return protocol


def test_initial_state():
    protocol = _new_protocol()

    assert protocol.get_phase() == OpeningPhase.DESIGNATED_BLACK_1
    assert protocol.player_to_color[PLAYER_A] == BLACK
    assert protocol.player_to_color[PLAYER_B] == WHITE
    assert protocol.get_current_player_id() == PLAYER_A
    assert protocol.get_current_stone_color() == BLACK
    assert protocol.is_protocol_finished() is False


def test_black1_must_be_h8():
    protocol = _new_protocol()
    protocol.play_designated_move(PLAYER_A, "H8")

    assert protocol.get_phase() == OpeningPhase.DESIGNATED_WHITE_2
    assert protocol.board.grid[7][7] == BLACK
    assert protocol.board.last_move == (7, 7, BLACK)
    assert protocol.board.current_player == WHITE

    with pytest.raises(OpeningProtocolError, match="player is not allowed"):
        _new_protocol().play_designated_move(PLAYER_B, "H8")
    with pytest.raises(OpeningProtocolError, match="illegal action"):
        _new_protocol().play_designated_move(PLAYER_A, "A1")


def test_white2_legal_move():
    protocol = _after_black1()
    protocol.play_designated_move(PLAYER_B, "A1")

    assert protocol.get_phase() == OpeningPhase.DESIGNATED_BLACK_3
    assert protocol.board.grid[0][0] == WHITE
    assert protocol.board.current_player == BLACK

    with pytest.raises(OpeningProtocolError, match="illegal action"):
        _after_black1().play_designated_move(PLAYER_B, "H8")
    with pytest.raises(OpeningProtocolError, match="player is not allowed"):
        _after_black1().play_designated_move(PLAYER_A, "A1")


def test_black3_is_limited_to_center_5x5_area():
    protocol = _after_white2()
    protocol.play_designated_move(PLAYER_A, "I8")

    assert protocol.get_phase() == OpeningPhase.SWAP_DECISION
    assert protocol.board.grid[8][7] == BLACK
    assert protocol.board.current_player == WHITE

    with pytest.raises(OpeningProtocolError, match="illegal action"):
        _after_white2().play_designated_move(PLAYER_A, "K8")
    with pytest.raises(OpeningProtocolError, match="illegal action"):
        _after_white2().play_designated_move(PLAYER_A, "H8")


def test_swap_decision_false_keeps_player_colors():
    protocol = _after_black3()
    protocol.decide_swap(PLAYER_B, swap=False)

    assert protocol.swap_decision is False
    assert protocol.player_to_color[PLAYER_A] == BLACK
    assert protocol.player_to_color[PLAYER_B] == WHITE
    assert protocol.get_phase() == OpeningPhase.NORMAL_WHITE_4
    assert protocol.get_current_player_id() == PLAYER_B


def test_swap_decision_true_swaps_players_not_stones():
    protocol = _after_black3()
    before_stones = {
        (7, 7): protocol.board.grid[7][7],
        (0, 0): protocol.board.grid[0][0],
        (8, 7): protocol.board.grid[8][7],
    }
    protocol.decide_swap(PLAYER_B, swap=True)

    assert protocol.swap_decision is True
    assert protocol.player_to_color[PLAYER_A] == WHITE
    assert protocol.player_to_color[PLAYER_B] == BLACK
    assert protocol.get_phase() == OpeningPhase.NORMAL_WHITE_4
    assert protocol.get_current_player_id() == PLAYER_A
    assert protocol.board.grid[7][7] == before_stones[(7, 7)] == BLACK
    assert protocol.board.grid[0][0] == before_stones[(0, 0)] == WHITE
    assert protocol.board.grid[8][7] == before_stones[(8, 7)] == BLACK


def test_white4_uses_current_white_player():
    protocol = _after_swap_decision(swap=True)
    protocol.play_white_4(PLAYER_A, "O15")

    assert protocol.get_phase() == OpeningPhase.BLACK_5_CANDIDATES
    assert protocol.board.grid[14][14] == WHITE
    assert protocol.board.current_player == BLACK

    with pytest.raises(OpeningProtocolError, match="player is not allowed"):
        _after_swap_decision(swap=True).play_white_4(PLAYER_B, "O15")


def test_black5_candidate_submission_validation():
    protocol = _after_white4(swap=False)
    protocol.submit_black5_candidates(PLAYER_A, ["B2", "C2"])

    assert protocol.get_phase() == OpeningPhase.WHITE_SELECT_BLACK_5
    assert protocol.candidate_black5_actions == [
        action_to_index(1, 1),
        action_to_index(2, 1),
    ]

    with pytest.raises(OpeningProtocolError, match="expected 2"):
        _after_white4().submit_black5_candidates(PLAYER_A, ["B2"])
    with pytest.raises(OpeningProtocolError, match="expected 2"):
        _after_white4().submit_black5_candidates(PLAYER_A, ["B2", "C2", "D2"])
    with pytest.raises(OpeningProtocolError, match="unique"):
        _after_white4().submit_black5_candidates(PLAYER_A, ["B2", "B2"])
    with pytest.raises(OpeningProtocolError, match="not empty"):
        _after_white4().submit_black5_candidates(PLAYER_A, ["H8", "B2"])
    with pytest.raises(OpeningProtocolError, match="only current black"):
        _after_white4().submit_black5_candidates(PLAYER_B, ["B2", "C2"])


def test_white_selects_black5_candidate():
    protocol = _after_white4(swap=False)
    protocol.submit_black5_candidates(PLAYER_A, ["B2", "C2"])
    protocol.select_black5_candidate(PLAYER_B, "C2")

    assert protocol.board.grid[2][1] == BLACK
    assert protocol.board.current_player == WHITE
    assert protocol.get_phase() == OpeningPhase.NORMAL_PLAY
    assert protocol.is_protocol_finished() is True
    assert protocol.get_current_player_id() == PLAYER_B

    with pytest.raises(OpeningProtocolError, match="not a black 5 candidate"):
        protocol = _after_white4()
        protocol.submit_black5_candidates(PLAYER_A, ["B2", "C2"])
        protocol.select_black5_candidate(PLAYER_B, "D2")

    with pytest.raises(OpeningProtocolError, match="only current white"):
        protocol = _after_white4()
        protocol.submit_black5_candidates(PLAYER_A, ["B2", "C2"])
        protocol.select_black5_candidate(PLAYER_A, "B2")


def test_normal_play_after_protocol():
    protocol = _after_black5(swap=False)

    protocol.play_normal_move(PLAYER_B, "D2")
    assert protocol.board.grid[3][1] == WHITE
    assert protocol.board.current_player == BLACK

    with pytest.raises(OpeningProtocolError, match="player is not allowed"):
        protocol.play_normal_move(PLAYER_B, "E2")
    with pytest.raises(OpeningProtocolError, match="illegal action"):
        protocol.play_normal_move(PLAYER_A, "D2")


def test_action_index_and_coordinate_mapping_are_consistent():
    protocol = _new_protocol()

    assert protocol.coord_to_action("H8") == CENTER_ACTION == 112
    assert protocol.coord_to_action("A1") == 0
    assert protocol.coord_to_action("O15") == 224
    assert protocol.coord_to_action((7, 7)) == 112
    assert protocol.action_to_coord(112) == (7, 7)
    assert protocol.play_designated_move(PLAYER_A, 112) == 112


def test_wrong_phase_operations_raise_clear_errors():
    protocol = _new_protocol()

    with pytest.raises(OpeningProtocolError, match="not in black 5 candidate phase"):
        protocol.submit_black5_candidates(PLAYER_A, ["B2", "C2"])
    with pytest.raises(OpeningProtocolError, match="not in swap decision phase"):
        protocol.decide_swap(PLAYER_B, swap=False)
    with pytest.raises(OpeningProtocolError, match="not in white 4 phase"):
        protocol.play_white_4(PLAYER_B, "A1")
    with pytest.raises(OpeningProtocolError, match="normal play has not started"):
        protocol.play_normal_move(PLAYER_A, "A1")

    protocol = _after_swap_decision()
    with pytest.raises(OpeningProtocolError, match="not in black 5 candidate phase"):
        protocol.submit_black5_candidates(PLAYER_A, ["B2", "C2"])
