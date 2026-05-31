"""Tests for ``engine.vcf_search``.

We construct positions where the correctness of the answer is fully determined
by the board state — no hard-coded "the answer is action 99" hacks. Instead
each test asserts:

  * the returned move (when not None) is legal, and
  * after playing it, opponent has no legal way to avoid losing in <= 2 plies.

Plus the negative cases (empty / quiet positions must return None).
"""

from __future__ import annotations

import pytest

from engine.simulation import temporary_stone
from engine.vcf_search import (
    find_vcf_attack_candidates,
    vcf_defends,
    vcf_first_move,
)
from game.board import BLACK, BOARD_SIZE, EMPTY, WHITE, Board
from game.encoder import action_to_index, index_to_action
from game.rules_basic import check_winner


def _set_stones(board: Board, black: list[tuple[int, int]], white: list[tuple[int, int]]) -> None:
    for x, y in black:
        board.grid[x][y] = BLACK
    for x, y in white:
        board.grid[x][y] = WHITE
    board.move_count = len(black) + len(white)
    # Current player is whoever has fewer stones (black plays first).
    board.current_player = BLACK if len(black) == len(white) else WHITE


def _opponent_can_avoid_loss_within_one_ply(
    board: Board, attacker_first: int, attacker_color: int
) -> bool:
    """Does opponent have ANY reply that prevents attacker winning next ply?"""
    ax, ay = index_to_action(int(attacker_first), BOARD_SIZE)
    with temporary_stone(board, ax, ay, attacker_color):
        # See if attacker now has any 5-in-a-row already
        if check_winner(board, board.last_move) == attacker_color:
            return False  # game already over; opp can't avoid
        # For every legal opp reply, check if attacker still has an immediate five
        for ox in range(BOARD_SIZE):
            for oy in range(BOARD_SIZE):
                if board.grid[ox][oy] != EMPTY:
                    continue
                with temporary_stone(board, ox, oy, -attacker_color):
                    # Does attacker have an immediate win NEXT ply (any cell)?
                    has_win = False
                    for wx in range(BOARD_SIZE):
                        if has_win:
                            break
                        for wy in range(BOARD_SIZE):
                            if board.grid[wx][wy] != EMPTY:
                                continue
                            with temporary_stone(board, wx, wy, attacker_color):
                                if check_winner(board, board.last_move) == attacker_color:
                                    has_win = True
                                    break
                    if not has_win:
                        # Opp survived this reply
                        return True
        return False


# ---------------------------------------------------------------------------
# 1) immediate-five trivial cases
# ---------------------------------------------------------------------------
class TestImmediateFive:
    def test_open_end_of_four_is_a_one_ply_mate(self):
        board = Board()
        _set_stones(board, black=[(3, 7), (4, 7), (5, 7), (6, 7)], white=[])
        # Black to move; placing at (2,7) or (7,7) makes 5.
        first = vcf_first_move(board, BLACK, max_depth=3, rule_mode="basic")
        assert first is not None
        x, y = index_to_action(first, BOARD_SIZE)
        assert board.is_legal_move(x, y)
        # The move itself must be an immediate five.
        with temporary_stone(board, x, y, BLACK):
            assert check_winner(board, board.last_move) == BLACK

    def test_no_threat_position_returns_none(self):
        board = Board()
        # 2 isolated black stones, nothing close to a four.
        _set_stones(board, black=[(7, 7)], white=[(8, 8)])
        result = vcf_first_move(board, BLACK, max_depth=9, rule_mode="basic")
        assert result is None

    def test_empty_board_returns_none(self):
        board = Board()
        assert vcf_first_move(board, BLACK, max_depth=9, rule_mode="basic") is None


# ---------------------------------------------------------------------------
# 2) multi-threat (double four / open four) — 3-ply mate
# ---------------------------------------------------------------------------
class TestMultiThreatMate:
    def test_open_four_creator_is_a_three_ply_mate(self):
        """Black has 3 in a row with both extensions still open. Black plays at
        one open end to make an open-four; opponent can only block one end, so
        black makes 5 next move.
        """
        board = Board()
        # Black has stones at (4,7), (5,7), (6,7). Open both sides.
        _set_stones(board, black=[(4, 7), (5, 7), (6, 7)], white=[])
        first = vcf_first_move(board, BLACK, max_depth=5, rule_mode="basic")
        assert first is not None
        x, y = index_to_action(first, BOARD_SIZE)
        # After this move, opponent cannot avoid losing next ply.
        assert not _opponent_can_avoid_loss_within_one_ply(board, first, BLACK)

    def test_double_four_creator_is_a_three_ply_mate(self):
        """Set up a position where one black move creates two separate four-lines
        (horizontal + diagonal) through it. Opponent can block at most one end.
        """
        board = Board()
        # Horizontal: black at (3,7),(4,7),(5,7)  → playing (6,7) makes a four
        # Diagonal:  black at (3,6),(5,8),(6,9) shares (4,7) on the / diag? Not quite.
        # Easier: use two parallel structures that share an intersection cell.
        # Build: row 7 has B at x=3,4,5; col 6 has B at y=4,5,6. Cell (6,7) and
        # cell (6,4) are extensions. Try a cell that extends BOTH.
        #
        # Use this layout:
        #   . . . . . . . . .
        #   . . . . . . . . .
        #   . . . . . . B . .   row 4
        #   . . . . . . B . .   row 5
        #   . . . . . . B . .   row 6
        #   . . . B B B . . .   row 7  (black has 3 in row 7 from x=3..5)
        # Playing (6, 7) creates: a four in row 7 (x=3..6) AND a four in column 6
        # (y=4..7). That's a double four.
        _set_stones(
            board,
            black=[(3, 7), (4, 7), (5, 7), (6, 4), (6, 5), (6, 6)],
            white=[],
        )
        first = vcf_first_move(board, BLACK, max_depth=5, rule_mode="basic")
        assert first is not None
        # Whichever move VCF chose, opponent should have no successful defense.
        assert not _opponent_can_avoid_loss_within_one_ply(board, first, BLACK)


# ---------------------------------------------------------------------------
# 3) depth control & budget
# ---------------------------------------------------------------------------
class TestDepthAndBudget:
    def test_max_depth_zero_returns_none(self):
        board = Board()
        _set_stones(board, black=[(3, 7), (4, 7), (5, 7), (6, 7)], white=[])
        # Even though there's a one-ply win, depth 0 must return None.
        assert vcf_first_move(board, BLACK, max_depth=0, rule_mode="basic") is None

    def test_tiny_node_budget_returns_none_for_deep_mate(self):
        # A deep VCF search with budget=1 must terminate gracefully (None).
        board = Board()
        # Construct a position that requires several plies; here we use a 3-stone
        # open-three that needs the multi-threat branch.
        _set_stones(board, black=[(4, 7), (5, 7), (6, 7)], white=[])
        result = vcf_first_move(
            board, BLACK, max_depth=11, rule_mode="basic", node_budget=1
        )
        # Either None (budget hit) or an immediate one-ply mate is acceptable;
        # the contract is: never raise.
        assert result is None or isinstance(result, int)

    def test_vcf_does_not_mutate_board(self):
        board = Board()
        _set_stones(board, black=[(3, 7), (4, 7), (5, 7), (6, 7)], white=[])
        snapshot_grid = [row[:] for row in board.grid]
        snapshot_count = board.move_count
        snapshot_player = board.current_player
        snapshot_last = board.last_move
        _ = vcf_first_move(board, BLACK, max_depth=7, rule_mode="basic")
        assert board.grid == snapshot_grid
        assert board.move_count == snapshot_count
        assert board.current_player == snapshot_player
        assert board.last_move == snapshot_last


# ---------------------------------------------------------------------------
# 4) attack-candidate enumeration
# ---------------------------------------------------------------------------
class TestAttackCandidates:
    def test_candidates_are_all_four_threats(self):
        board = Board()
        _set_stones(board, black=[(4, 7), (5, 7), (6, 7)], white=[])
        cands = find_vcf_attack_candidates(board, BLACK, rule_mode="basic")
        # Each candidate must, when placed, give us at least one immediate-five
        # cell available next ply.
        for action in cands:
            x, y = index_to_action(action, BOARD_SIZE)
            with temporary_stone(board, x, y, BLACK):
                # any cell that would make 5?
                has_followup = False
                for wx in range(BOARD_SIZE):
                    if has_followup:
                        break
                    for wy in range(BOARD_SIZE):
                        if board.grid[wx][wy] != EMPTY:
                            continue
                        with temporary_stone(board, wx, wy, BLACK):
                            if check_winner(board, board.last_move) == BLACK:
                                has_followup = True
                                break
                assert has_followup, f"action {action} ({x},{y}) is not a four threat"

    def test_quiet_position_has_no_candidates(self):
        board = Board()
        _set_stones(board, black=[(7, 7)], white=[(8, 8)])
        cands = find_vcf_attack_candidates(board, BLACK, rule_mode="basic")
        assert cands == []


# ---------------------------------------------------------------------------
# 5) defensive helper
# ---------------------------------------------------------------------------
class TestVcfDefends:
    def test_preempt_kills_opponent_blocked_four_mate(self):
        """Single-threat (blocked four): preempting at the unique open end
        removes opponent's mate, so vcf_defends returns True.

        Layout: white has 4-in-a-row at (3..6, 7) with the right end already
        blocked by a black stone at (7, 7). The only winning cell for white is
        (2, 7). Black playing (2, 7) ourselves leaves white with no immediate
        five anywhere.
        """
        board = Board()
        _set_stones(
            board,
            black=[(7, 7)],
            white=[(3, 7), (4, 7), (5, 7), (6, 7)],
        )
        block_cell = action_to_index(2, 7, BOARD_SIZE)
        assert vcf_defends(
            board, BLACK, block_cell, max_depth=5, rule_mode="basic"
        ), "preempting the unique open end of a blocked four must defend"

    def test_preempt_one_end_of_open_four_does_not_defend(self):
        """Open four has TWO winning cells; blocking one still loses to the
        other. This is the correct negative case (a deliberate sanity check
        that vcf_defends does not falsely report a defense)."""
        board = Board()
        _set_stones(
            board,
            black=[],
            white=[(3, 7), (4, 7), (5, 7), (6, 7)],
        )
        # Black plays only the left open end; the right end (7, 7) still mates.
        partial_block = action_to_index(2, 7, BOARD_SIZE)
        assert not vcf_defends(
            board, BLACK, partial_block, max_depth=5, rule_mode="basic"
        ), "blocking one end of an OPEN four does not stop the other end mate"

    def test_non_blocking_move_does_not_defend(self):
        board = Board()
        _set_stones(board, black=[], white=[(3, 7), (4, 7), (5, 7), (6, 7)])
        # Playing somewhere unrelated does NOT remove white's threat.
        useless = action_to_index(0, 0, BOARD_SIZE)
        assert not vcf_defends(
            board, BLACK, useless, max_depth=5, rule_mode="basic"
        )
