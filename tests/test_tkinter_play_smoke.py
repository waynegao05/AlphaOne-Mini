from __future__ import annotations

import importlib
import tkinter as tk

import pytest


def test_main_tkinter_play_import_does_not_start_gui():
    module = importlib.import_module("main_tkinter_play")
    assert hasattr(module, "main")
    assert hasattr(module, "parse_args")


def test_board_coordinate_conversion_round_trips():
    from ui.tkinter_board import board_to_canvas, canvas_to_board

    x, y = board_to_canvas(7, 7, margin=30, cell_size=32)
    assert (x, y) == (254, 254)
    assert canvas_to_board(x, y, margin=30, cell_size=32, board_size=15) == (7, 7)
    a1_x, a1_y = board_to_canvas(0, 0, margin=30, cell_size=32, board_size=15)
    o15_x, o15_y = board_to_canvas(14, 14, margin=30, cell_size=32, board_size=15)
    assert (a1_x, a1_y) == (30, 478)
    assert (o15_x, o15_y) == (478, 30)
    assert canvas_to_board(a1_x, a1_y, margin=30, cell_size=32, board_size=15) == (0, 0)
    assert canvas_to_board(o15_x, o15_y, margin=30, cell_size=32, board_size=15) == (14, 14)
    assert canvas_to_board(0, 0, margin=30, cell_size=32, board_size=15) is None


def test_gomoku_tk_app_can_instantiate_hidden_root():
    from ui.tkinter_board import GomokuTkApp

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk is not available in this environment: {exc}")
    root.withdraw()
    try:
        app = GomokuTkApp(root, device="cpu", auto_build_players=False)
        assert app.board_size == 15
        assert app.rule_mode_var.get() == "basic"
        assert app.black_player_var.get() == "Human"
        assert app.white_player_var.get() == "AlphaOne-Mini"
        assert app.canvas is not None
    finally:
        root.destroy()


class _CenterAi:
    name = "CenterAi"
    decision_reason = "test_center"

    def select_action(self, board):
        from game.encoder import action_to_index

        return action_to_index(7, 7)


class _RightAi:
    name = "RightAi"
    decision_reason = "test_right"

    def select_action(self, board):
        from game.encoder import action_to_index

        return action_to_index(8, 7)


def test_tkinter_dual_human_mode_does_not_auto_move():
    from game.board import BLACK, WHITE
    from ui.player_factory import PlayerSlot
    from ui.tkinter_board import GomokuTkApp

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk is not available in this environment: {exc}")
    root.withdraw()
    try:
        app = GomokuTkApp(root, device="cpu", auto_build_players=False)
        app.players = {
            BLACK: PlayerSlot("human", BLACK, None, "Human", True),
            WHITE: PlayerSlot("human", WHITE, None, "Human", True),
        }
        app.schedule_next_turn()
        assert app.board.move_count == 0
        assert app.current_slot().is_human
    finally:
        root.destroy()


def test_tkinter_ai_step_can_execute_one_move_without_mainloop():
    from game.board import BLACK, WHITE
    from ui.player_factory import PlayerSlot
    from ui.tkinter_board import GomokuTkApp

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk is not available in this environment: {exc}")
    root.withdraw()
    try:
        app = GomokuTkApp(root, device="cpu", auto_build_players=False)
        app.players = {
            BLACK: PlayerSlot("test_ai", BLACK, _CenterAi(), "TestAI", False),
            WHITE: PlayerSlot("human", WHITE, None, "Human", True),
        }
        assert app.execute_current_ai_move_sync() is True
        assert app.board.move_count == 1
        assert app.board.grid[7][7] == BLACK
        assert app.current_slot().is_human
    finally:
        root.destroy()


def test_tkinter_human_click_then_ai_can_respond_without_mainloop():
    from game.board import BLACK, WHITE
    from ui.player_factory import PlayerSlot
    from ui.tkinter_board import GomokuTkApp, board_to_canvas

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk is not available in this environment: {exc}")
    root.withdraw()

    class Event:
        pass

    try:
        app = GomokuTkApp(root, device="cpu", auto_build_players=False)
        app.players = {
            BLACK: PlayerSlot("human", BLACK, None, "Human", True),
            WHITE: PlayerSlot("test_ai", WHITE, _RightAi(), "TestAI", False),
        }
        px, py = board_to_canvas(7, 7, margin=app.margin, cell_size=app.cell_size)
        event = Event()
        event.x = px
        event.y = py

        app.on_canvas_click(event)
        assert app.board.grid[7][7] == BLACK
        assert app.board.current_player == WHITE

        assert app.execute_current_ai_move_sync() is True
        assert app.board.grid[8][7] == WHITE
        assert app.board.move_count == 2
    finally:
        root.destroy()


def test_tkinter_undo_on_human_turn_removes_human_ai_pair():
    from game.board import BLACK, WHITE, EMPTY
    from ui.player_factory import PlayerSlot
    from ui.tkinter_board import GomokuTkApp, board_to_canvas

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk is not available in this environment: {exc}")
    root.withdraw()

    class Event:
        pass

    try:
        app = GomokuTkApp(root, device="cpu", auto_build_players=False)
        app.players = {
            BLACK: PlayerSlot("human", BLACK, None, "Human", True),
            WHITE: PlayerSlot("test_ai", WHITE, _RightAi(), "TestAI", False),
        }
        px, py = board_to_canvas(7, 7, margin=app.margin, cell_size=app.cell_size)
        event = Event()
        event.x = px
        event.y = py

        app.on_canvas_click(event)
        assert app.execute_current_ai_move_sync() is True
        assert app.board.move_count == 2
        assert app.current_slot().is_human

        app.undo_human_move()
        assert app.board.move_count == 0
        assert app.board.grid[7][7] == EMPTY
        assert app.board.grid[8][7] == EMPTY
        assert app.current_slot().is_human
    finally:
        root.destroy()


def test_tkinter_builds_standard_c5_record_and_tracks_time(tmp_path):
    from game.board import BLACK, WHITE
    from records.file_io import read_record_file, write_record_file
    from ui.player_factory import PlayerSlot
    from ui.tkinter_board import GomokuTkApp

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk is not available in this environment: {exc}")
    root.withdraw()
    try:
        app = GomokuTkApp(root, device="cpu", auto_build_players=False)
        app.players = {
            BLACK: PlayerSlot("human", BLACK, None, "Human", True),
            WHITE: PlayerSlot("human", WHITE, None, "Human", True),
        }

        app._place_move(7, 7, app.players[BLACK], decision_reason="human", elapsed=1.25)
        app._place_move(8, 7, app.players[WHITE], decision_reason="human", elapsed=2.5)

        record_text = app.build_record_text()
        assert record_text.startswith("{[C5]")
        assert ";B(H,8);W(I,8)" in record_text
        assert app.elapsed_by_color[BLACK] == pytest.approx(1.25)
        assert app.elapsed_by_color[WHITE] == pytest.approx(2.5)

        timing = app._timing_payload()
        assert timing["black_elapsed_seconds"] == pytest.approx(1.25)
        assert timing["white_elapsed_seconds"] == pytest.approx(2.5)

        path = tmp_path / "record.txt"
        write_record_file(path, record_text)
        assert read_record_file(path) == record_text
    finally:
        root.destroy()


def test_tkinter_three_move_swap_exchanges_player_slots_only():
    from game.board import BLACK, WHITE
    from ui.player_factory import PlayerSlot
    from ui.tkinter_board import GomokuTkApp

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk is not available in this environment: {exc}")
    root.withdraw()
    try:
        app = GomokuTkApp(root, device="cpu", auto_build_players=False)
        app.players = {
            BLACK: PlayerSlot("human", BLACK, None, "BlackHuman", True),
            WHITE: PlayerSlot("human", WHITE, None, "WhiteHuman", True),
        }
        app.black_player_var.set("Human")
        app.white_player_var.set("AlphaOne-Mini")

        app._place_move(7, 7, app.players[BLACK], decision_reason="test", elapsed=0.1)
        app._place_move(8, 7, app.players[WHITE], decision_reason="test", elapsed=0.1)
        app._place_move(8, 8, app.players[BLACK], decision_reason="test", elapsed=0.1)

        assert app.can_swap_players()
        app.swap_players()

        assert app.swap_performed is True
        assert app.players[BLACK].label == "WhiteHuman"
        assert app.players[WHITE].label == "BlackHuman"
        assert app.board.grid[7][7] == BLACK
        assert app.board.grid[8][7] == WHITE
        assert app.board.grid[8][8] == BLACK
        assert app.board.current_player == WHITE
        assert len(app.move_history) == 3
    finally:
        root.destroy()


def test_tkinter_autosave_json_can_restore_position(tmp_path):
    from game.board import BLACK, WHITE
    from ui.player_factory import PlayerSlot
    from ui.tkinter_board import GomokuTkApp

    try:
        root = tk.Tk()
        root2 = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk is not available in this environment: {exc}")
    root.withdraw()
    root2.withdraw()
    try:
        autosave = tmp_path / "autosave.json"
        app = GomokuTkApp(root, device="cpu", auto_build_players=False)
        app.autosave_path = autosave
        app.players = {
            BLACK: PlayerSlot("human", BLACK, None, "Human", True),
            WHITE: PlayerSlot("human", WHITE, None, "Human", True),
        }
        app._place_move(7, 7, app.players[BLACK], decision_reason="human", elapsed=1.0)
        app._place_move(8, 7, app.players[WHITE], decision_reason="human", elapsed=2.0)
        assert autosave.exists()

        restored = GomokuTkApp(root2, device="cpu", auto_build_players=False)
        restored.autosave_path = autosave
        restored.load_record_path(autosave)
        assert restored.board.move_count == 2
        assert restored.board.grid[7][7] == BLACK
        assert restored.board.grid[8][7] == WHITE
        assert restored.elapsed_by_color[BLACK] == pytest.approx(1.0)
        assert restored.elapsed_by_color[WHITE] == pytest.approx(2.0)
    finally:
        root.destroy()
        root2.destroy()


def test_tkinter_loads_standard_c5_text_record(tmp_path):
    from game.board import BLACK, WHITE
    from records.file_io import write_record_file
    from ui.tkinter_board import GomokuTkApp

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk is not available in this environment: {exc}")
    root.withdraw()
    try:
        path = tmp_path / "standard.txt"
        write_record_file(
            path,
            "{[C5][先手参赛队 B][后手参赛队 W][未结束][2026.05.28 本地测试][AlphaOne-Mini];B(H,8);W(I,8)}",
        )
        app = GomokuTkApp(root, device="cpu", auto_build_players=False)
        app.load_record_path(path)
        assert app.board.move_count == 2
        assert app.board.grid[7][7] == BLACK
        assert app.board.grid[8][7] == WHITE
        assert app.move_history[0]["coord"] == "H8"
        assert app.move_history[1]["coord"] == "I8"
    finally:
        root.destroy()


def test_tkinter_competition_protocol_opening_swap_and_five_n():
    from game.board import BLACK, WHITE
    from ui.player_factory import PlayerSlot
    from ui.tkinter_board import (
        GomokuTkApp,
        PHASE_BLACK5_CANDIDATES,
        PHASE_DESIGNATED_BLACK_1,
        PHASE_DESIGNATED_BLACK_3,
        PHASE_DESIGNATED_WHITE_2,
        PHASE_NORMAL_PLAY,
        PHASE_NORMAL_WHITE_4,
        PHASE_SWAP_DECISION,
        PHASE_WHITE_SELECT_BLACK5,
    )

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk is not available in this environment: {exc}")
    root.withdraw()
    try:
        app = GomokuTkApp(root, device="cpu", auto_build_players=False)
        app.players = {
            BLACK: PlayerSlot("human", BLACK, None, "BlackHuman", True),
            WHITE: PlayerSlot("human", WHITE, None, "WhiteHuman", True),
        }
        app.competition_protocol_var.set(True)
        app.fifth_n_var.set(2)
        app.restart_game(rebuild_players=False)

        assert app.protocol_phase == PHASE_DESIGNATED_BLACK_1
        assert app._is_legal_action_for_current_player(7, 7)
        assert not app._is_legal_action_for_current_player(6, 7)

        app._place_move(7, 7, app.current_slot(), decision_reason="test", elapsed=0.1)
        assert app.protocol_phase == PHASE_DESIGNATED_WHITE_2
        assert app.current_slot().label == "BlackHuman"
        assert app._is_legal_action_for_current_player(8, 7)
        assert not app._is_legal_action_for_current_player(10, 10)

        app._place_move(8, 7, app.current_slot(), decision_reason="test", elapsed=0.1)
        assert app.protocol_phase == PHASE_DESIGNATED_BLACK_3
        assert app.board.grid[8][7] == WHITE
        assert app.current_slot().label == "BlackHuman"
        assert app._is_legal_action_for_current_player(9, 9)
        assert not app._is_legal_action_for_current_player(11, 11)

        app._place_move(9, 9, app.current_slot(), decision_reason="test", elapsed=0.1)
        assert app.protocol_phase == PHASE_SWAP_DECISION
        assert app.can_swap_players()

        app.decline_swap()
        assert app.protocol_phase == PHASE_NORMAL_WHITE_4
        assert app.current_slot().label == "WhiteHuman"

        app._place_move(6, 7, app.current_slot(), decision_reason="white4", elapsed=0.1)
        assert app.protocol_phase == PHASE_BLACK5_CANDIDATES
        assert app.current_slot().label == "BlackHuman"

        app._add_black5_candidate(6, 8, app.current_slot(), elapsed=0.2)
        assert app.protocol_phase == PHASE_BLACK5_CANDIDATES
        app._add_black5_candidate(7, 8, app.current_slot(), elapsed=0.2)
        assert app.protocol_phase == PHASE_WHITE_SELECT_BLACK5
        assert len(app.black5_candidate_actions) == 2

        app._select_black5_candidate(6, 8, app.current_slot(), elapsed=0.1)
        assert app.protocol_phase == PHASE_NORMAL_PLAY
        assert app.board.move_count == 5
        assert app.board.grid[6][8] == BLACK
        assert app.board.current_player == WHITE
    finally:
        root.destroy()


def test_tkinter_clock_timeout_forfeits_current_side(monkeypatch):
    from ui.tkinter_board import GomokuTkApp

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk is not available in this environment: {exc}")
    root.withdraw()
    monkeypatch.setattr("ui.tkinter_board.messagebox.showinfo", lambda *args, **kwargs: None)
    try:
        app = GomokuTkApp(root, device="cpu", auto_build_players=False)
        app.time_limit_seconds = 0.01
        app.turn_started_at -= 1.0
        assert app._check_clock_timeout() is True
        assert app.game_over is True
        assert app.result_text == "White wins (black_timeout)"
    finally:
        root.destroy()
