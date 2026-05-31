"""Tkinter board UI for local Gomoku testing."""

from __future__ import annotations

import json
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from game.board import BLACK, BOARD_SIZE, EMPTY, WHITE, Board
from game.coordinates import index_to_coord
from game.encoder import action_to_index, index_to_action
from game.notation import MoveRecord
from game.rules_basic import _find_any_winner, check_winner, is_game_over
from game.rules_forbidden import get_game_result_forbidden
from records.exporter import export_standard_record
from records.file_io import read_record_file, write_record_file
from records.metadata import RecordMetadata
from records.parser import parse_record
from train.progress import format_seconds

from .player_factory import (
    DEFAULT_EXTERNAL_AI_PATH,
    PLAYER_TYPE_LABELS,
    PlayerSlot,
    create_player,
    normalize_player_type,
    resolve_device,
)


DEFAULT_AUTOSAVE_PATH = Path("outputs") / "autosave" / "alphaone_mini_tkinter_latest.json"
TIME_LIMIT_SECONDS = 15 * 60

PHASE_NORMAL_PLAY = "normal_play"
PHASE_DESIGNATED_BLACK_1 = "designated_black_1"
PHASE_DESIGNATED_WHITE_2 = "designated_white_2"
PHASE_DESIGNATED_BLACK_3 = "designated_black_3"
PHASE_SWAP_DECISION = "swap_decision"
PHASE_NORMAL_WHITE_4 = "normal_white_4"
PHASE_BLACK5_CANDIDATES = "black5_candidates"
PHASE_WHITE_SELECT_BLACK5 = "white_select_black5"

# ---------------------------------------------------------------------------
# 26 standard designated openings (中国五子棋竞赛规则 指定开局)
#
# Coordinate system:  x = column (A=0 … O=14),  y = row-1 (so H8 = (7,7)).
# Black 1 always at centre (7,7) = H8.
# 直指 (direct):   White 2 at H9 (7,8) — orthogonally adjacent above centre.
# 斜指 (diagonal): White 2 at I9 (8,8) — diagonally adjacent above-right.
# Black 3 must lie within the 5×5 region centred on H8 (x=5…9, y=5…9).
# ---------------------------------------------------------------------------
STANDARD_OPENING_26: dict[str, tuple[int, int, int, int]] = {
    # ---- 直指 — W2 = H9 (7,8) ----
    "1  寒星 (Cold Star)":      (7, 8,  7, 9),   # B3 H10
    "2  溪月 (Stream Moon)":    (7, 8,  8, 9),   # B3 I10
    "3  疏星 (Sparse Star)":    (7, 8,  9, 9),   # B3 J10
    "4  花月 (Flower Moon)":    (7, 8,  8, 8),   # B3 I9
    "5  残月 (Waning Moon)":    (7, 8,  9, 8),   # B3 J9
    "6  雨月 (Rain Moon)":      (7, 8,  8, 7),   # B3 I8
    "7  金星 (Venus)":          (7, 8,  9, 7),   # B3 J8
    "8  松月 (Pine Moon)":      (7, 8,  7, 6),   # B3 H7
    "9  丘月 (Hill Moon)":      (7, 8,  8, 6),   # B3 I7
    "10 新月 (New Moon)":       (7, 8,  9, 6),   # B3 J7
    "11 瑞星 (Auspicious Star)":(7, 8,  7, 5),   # B3 H6
    "12 山月 (Mountain Moon)":  (7, 8,  8, 5),   # B3 I6
    "13 游星 (Roaming Star)":   (7, 8,  9, 5),   # B3 J6
    # ---- 斜指 — W2 = I9 (8,8) ----
    "14 长星 (Long Star)":      (8, 8,  9, 9),   # B3 J10
    "15 峡月 (Gorge Moon)":     (8, 8,  9, 8),   # B3 J9
    "16 恒星 (Fixed Star)":     (8, 8,  9, 7),   # B3 J8
    "17 水月 (Water Moon)":     (8, 8,  9, 6),   # B3 J7
    "18 流星 (Shooting Star)":  (8, 8,  9, 5),   # B3 J6
    "19 云月 (Cloud Moon)":     (8, 8,  8, 7),   # B3 I8
    "20 浦月 (Shore Moon)":     (8, 8,  8, 6),   # B3 I7
    "21 岚月 (Mist Moon)":      (8, 8,  8, 5),   # B3 I6
    "22 银月 (Silver Moon)":    (8, 8,  7, 6),   # B3 H7
    "23 明星 (Bright Star)":    (8, 8,  7, 5),   # B3 H6
    "24 斜月 (Slant Moon)":     (8, 8,  6, 6),   # B3 G7
    "25 名月 (Famous Moon)":    (8, 8,  6, 5),   # B3 G6
    "26 彗星 (Comet)":          (8, 8,  5, 4),   # B3 F5
}


def board_to_canvas(
    x: int,
    y: int,
    *,
    margin: int,
    cell_size: int,
    board_size: int = BOARD_SIZE,
) -> tuple[int, int]:
    # C5 records use A1 at the lower-left corner, so canvas Y is inverted.
    return margin + int(x) * cell_size, margin + (int(board_size) - 1 - int(y)) * cell_size


def canvas_to_board(
    px: float,
    py: float,
    *,
    margin: int,
    cell_size: int,
    board_size: int = BOARD_SIZE,
) -> Optional[tuple[int, int]]:
    x = round((float(px) - margin) / cell_size)
    canvas_row = round((float(py) - margin) / cell_size)
    y = int(board_size) - 1 - canvas_row
    if not (0 <= x < board_size and 0 <= y < board_size):
        return None
    cx, cy = board_to_canvas(x, y, margin=margin, cell_size=cell_size, board_size=board_size)
    if abs(float(px) - cx) > cell_size * 0.45 or abs(float(py) - cy) > cell_size * 0.45:
        return None
    return int(x), int(y)


class GomokuTkApp:
    """Local GUI where black and white slots can each be Human or AI."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        rule_mode: str = "basic",
        ai_player: str = "alphaone_mini",
        black_player: str = "human",
        white_player: Optional[str] = None,
        external_ai_path: str = DEFAULT_EXTERNAL_AI_PATH,
        device: str = "cuda",
        auto_build_players: bool = True,
        auto_build_ai: Optional[bool] = None,
        board_size: int = BOARD_SIZE,
        num_simulations: int = 50,
        move_delay_ms: int = 250,
        competition_protocol: bool = False,
        fifth_n: int = 2,
    ) -> None:
        if auto_build_ai is not None:
            auto_build_players = bool(auto_build_ai)
        if white_player is None:
            white_player = ai_player

        self.root = root
        self.root.title("AlphaOne-Mini 五子棋")
        self.board_size = int(board_size)
        self.margin = 34
        self.cell_size = 34
        self.stone_radius = 14
        self.num_simulations = int(num_simulations)
        self.requested_device = device
        self.device = resolve_device(device)
        self.move_delay_ms = int(move_delay_ms)

        self.board = Board()
        self.players: dict[int, PlayerSlot] = {
            BLACK: PlayerSlot("human", BLACK, None, "人类", True),
            WHITE: PlayerSlot("human", WHITE, None, "人类", True),
        }
        self.ai_thinking = False
        self.game_over = False
        self.paused = False
        self.last_ai_decision = "-"
        self.move_history: list[dict] = []
        now = time.perf_counter()
        self.game_started_at = now
        self.turn_started_at = now
        self.elapsed_by_color = {BLACK: 0.0, WHITE: 0.0}
        self.result_text = "ongoing"
        self.swap_performed = False
        self.autosave_path = DEFAULT_AUTOSAVE_PATH
        self.time_limit_seconds = TIME_LIMIT_SECONDS
        self.protocol_phase = PHASE_NORMAL_PLAY
        self.black5_candidate_actions: list[int] = []
        self._timer_job: Optional[str] = None
        self._clock_active: bool = False  # frozen until the first move

        self.rule_mode_var = tk.StringVar(value=rule_mode)
        self.device_var = tk.StringVar(value=device)
        self.black_player_var = tk.StringVar(value=PLAYER_TYPE_LABELS[normalize_player_type(black_player)])
        self.white_player_var = tk.StringVar(value=PLAYER_TYPE_LABELS[normalize_player_type(white_player)])
        self.external_ai_path_var = tk.StringVar(value=external_ai_path)
        self.move_delay_var = tk.IntVar(value=self.move_delay_ms)
        self.competition_protocol_var = tk.BooleanVar(value=bool(competition_protocol))
        self.fifth_n_var = tk.IntVar(value=max(2, min(5, int(fifth_n))))
        self.status_var = tk.StringVar()

        self._build_layout()
        if auto_build_players:
            self.rebuild_players()
        self.restart_game(rebuild_players=False, write_autosave=False)
        self._schedule_timer_tick()

    def _build_layout(self) -> None:
        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Label(top, text="规则").grid(row=0, column=0, sticky="w")
        ttk.OptionMenu(
            top,
            self.rule_mode_var,
            self.rule_mode_var.get(),
            "basic",
            "forbidden",
            command=lambda _value: self.start_game(),
        ).grid(row=0, column=1, sticky="w", padx=(4, 12))

        ttk.Label(top, text="设备").grid(row=0, column=2, sticky="w")
        ttk.OptionMenu(
            top,
            self.device_var,
            self.device_var.get(),
            "cuda",
            "cpu",
        ).grid(row=0, column=3, sticky="w", padx=(4, 12))

        player_values = list(PLAYER_TYPE_LABELS.values())
        ttk.Label(top, text="黑方").grid(row=1, column=0, sticky="w")
        ttk.OptionMenu(
            top,
            self.black_player_var,
            self.black_player_var.get(),
            *player_values,
        ).grid(row=1, column=1, sticky="w", padx=(4, 12))

        ttk.Label(top, text="白方").grid(row=1, column=2, sticky="w")
        ttk.OptionMenu(
            top,
            self.white_player_var,
            self.white_player_var.get(),
            *player_values,
        ).grid(row=1, column=3, sticky="w", padx=(4, 12))

        ttk.Label(top, text="外部AI路径").grid(row=2, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.external_ai_path_var, width=58).grid(
            row=2, column=1, columnspan=3, sticky="we", padx=(4, 6)
        )
        ttk.Button(top, text="浏览", command=self.browse_external_ai).grid(
            row=2, column=4, sticky="w"
        )

        ttk.Label(top, text="延迟ms").grid(row=0, column=4, sticky="e", padx=(8, 2))
        ttk.Spinbox(top, from_=0, to=5000, increment=50, textvariable=self.move_delay_var, width=7).grid(
            row=0, column=5, sticky="w"
        )

        buttons = ttk.Frame(top)
        buttons.grid(row=1, column=4, columnspan=2, sticky="w", padx=(8, 0))
        ttk.Button(buttons, text="开始", command=self.start_game).pack(side=tk.LEFT, padx=2)
        ttk.Button(buttons, text="暂停", command=self.pause_game).pack(side=tk.LEFT, padx=2)
        ttk.Button(buttons, text="单步", command=self.step_game).pack(side=tk.LEFT, padx=2)
        self.undo_button = ttk.Button(buttons, text="悔棋", command=self.undo_human_move)
        self.undo_button.pack(side=tk.LEFT, padx=2)
        self.swap_button = ttk.Button(buttons, text="换手", command=self.swap_players)
        self.swap_button.pack(side=tk.LEFT, padx=2)
        ttk.Button(buttons, text="重开", command=self.restart_game).pack(side=tk.LEFT, padx=2)
        ttk.Button(buttons, text="加载棋谱", command=self.load_record_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(buttons, text="加载存档", command=self.load_autosave).pack(side=tk.LEFT, padx=2)
        ttk.Button(buttons, text="保存JSON", command=self.save_record_json).pack(side=tk.LEFT, padx=2)
        ttk.Button(buttons, text="保存棋谱", command=self.save_record_text).pack(side=tk.LEFT, padx=2)

        protocol = ttk.Frame(top)
        protocol.grid(row=3, column=0, columnspan=6, sticky="w", pady=(6, 0))
        ttk.Checkbutton(
            protocol,
            text="竞赛规则：指定开局 + 换手 + 五手N打",
            variable=self.competition_protocol_var,
            command=self.start_game,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(protocol, text="N").pack(side=tk.LEFT)
        ttk.Spinbox(protocol, from_=2, to=5, increment=1, textvariable=self.fifth_n_var, width=4).pack(
            side=tk.LEFT, padx=(2, 8)
        )
        self.no_swap_button = ttk.Button(protocol, text="不换手", command=self.decline_swap)
        self.no_swap_button.pack(side=tk.LEFT, padx=2)

        ttk.Label(protocol, text="  开局").pack(side=tk.LEFT, padx=(12, 2))
        self.opening_var = tk.StringVar(value="")
        self.opening_combo = ttk.Combobox(
            protocol,
            textvariable=self.opening_var,
            values=[""] + list(STANDARD_OPENING_26.keys()),
            state="readonly",
            width=30,
        )
        self.opening_combo.pack(side=tk.LEFT, padx=2)
        self.opening_combo.bind("<<ComboboxSelected>>", self._on_opening_selected)

        body = ttk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True)

        size = self.margin * 2 + self.cell_size * (self.board_size - 1)
        self.canvas = tk.Canvas(body, width=size, height=size, bg="#d8a35d")
        self.canvas.pack(side=tk.LEFT, padx=8, pady=8)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        right = ttk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8), pady=8)

        # ---------- countdown timers ----------
        timer_frame = ttk.LabelFrame(right, text=" 倒计时 ", padding=6)
        timer_frame.pack(fill=tk.X, pady=(0, 8))

        # Black timer (top)
        black_row = ttk.Frame(timer_frame)
        black_row.pack(fill=tk.X, pady=(0, 4))
        self._black_timer_canvas = tk.Canvas(
            black_row, width=18, height=18, bg="#d8a35d", highlightthickness=0
        )
        self._black_timer_canvas.pack(side=tk.LEFT, padx=(0, 4))
        self._black_timer_canvas.create_oval(2, 2, 16, 16, fill="black", outline="black")
        ttk.Label(black_row, text="黑方", font=("", 9, "bold")).pack(side=tk.LEFT)
        self._black_timer_label = tk.Label(
            black_row, text="15:00", font=("Consolas", 18, "bold"), fg="#222"
        )
        self._black_timer_label.pack(side=tk.RIGHT)

        # White timer (bottom)
        white_row = ttk.Frame(timer_frame)
        white_row.pack(fill=tk.X)
        self._white_timer_canvas = tk.Canvas(
            white_row, width=18, height=18, bg="#d8a35d", highlightthickness=0
        )
        self._white_timer_canvas.pack(side=tk.LEFT, padx=(0, 4))
        self._white_timer_canvas.create_oval(2, 2, 16, 16, fill="white", outline="black")
        ttk.Label(white_row, text="白方", font=("", 9, "bold")).pack(side=tk.LEFT)
        self._white_timer_label = tk.Label(
            white_row, text="15:00", font=("Consolas", 18, "bold"), fg="#555"
        )
        self._white_timer_label.pack(side=tk.RIGHT)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 8))

        # ---------- move log ----------
        ttk.Label(right, text="走棋日志").pack(anchor="w")
        self.log_text = tk.Text(right, width=48, height=22, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        status = ttk.Label(self.root, textvariable=self.status_var, anchor="w")
        status.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 8))

    def browse_external_ai(self) -> None:
        path = filedialog.askopenfilename(
            title="选择AI.py",
            filetypes=[("Python", "*.py"), ("All files", "*.*")],
        )
        if path:
            self.external_ai_path_var.set(path)

    def rebuild_players(self) -> bool:
        self.requested_device = self.device_var.get()
        self.device = resolve_device(self.requested_device)
        try:
            self.players = {
                BLACK: create_player(
                    self.black_player_var.get(),
                    BLACK,
                    rule_mode=self.rule_mode_var.get(),
                    device=self.device,
                    external_ai_path=self.external_ai_path_var.get(),
                    num_simulations=self.num_simulations,
                ),
                WHITE: create_player(
                    self.white_player_var.get(),
                    WHITE,
                    rule_mode=self.rule_mode_var.get(),
                    device=self.device,
                    external_ai_path=self.external_ai_path_var.get(),
                    num_simulations=self.num_simulations,
                ),
            }
            return True
        except Exception as exc:
            self.last_ai_decision = f"棋手加载失败：{exc}"
            self.update_status()
            messagebox.showerror("棋手加载失败", str(exc))
            return False

    def start_game(self) -> None:
        if not self.rebuild_players():
            return
        self.restart_game(rebuild_players=False)
        self.paused = False
        self.schedule_next_turn()

    def pause_game(self) -> None:
        self.paused = True
        self.update_status()

    def step_game(self) -> None:
        self.paused = True
        if not self.game_over and not self.ai_thinking:
            slot = self.current_slot()
            if self.protocol_phase == PHASE_SWAP_DECISION:
                if not self.players[WHITE].is_human:
                    self.request_ai_swap_decision()  # human decides via buttons
            elif self.protocol_phase == PHASE_BLACK5_CANDIDATES:
                if not self.players[BLACK].is_human:
                    self.request_ai_black5_candidates()  # human proposes by clicking
            elif self.protocol_phase == PHASE_WHITE_SELECT_BLACK5:
                if not slot.is_human:
                    self.request_ai_select_black5()
            elif not slot.is_human:
                self.request_ai_move(async_move=True, single_step=True)
        self.update_status()

    def can_swap_players(self) -> bool:
        """Return whether the standard three-move player swap is available."""
        if self.competition_protocol_var.get():
            return self.protocol_phase == PHASE_SWAP_DECISION and not self.swap_performed
        return (
            self.board.move_count == 3
            and not self.swap_performed
            and not self.game_over
            and not self.ai_thinking
        )

    def swap_players(self) -> None:
        """Swap black/white player slots after the first three stones.

        This implements the competition-style three-move swap at the GUI layer:
        stones remain unchanged, board.current_player remains a stone color, and
        only the controller assigned to black/white is exchanged.
        """
        if not self.can_swap_players():
            self.last_ai_decision = "换手不可用：需要恰好3手"
            self.update_status()
            return

        black_slot = self.players[BLACK]
        white_slot = self.players[WHITE]
        self.players = {
            BLACK: self._slot_with_color(white_slot, BLACK),
            WHITE: self._slot_with_color(black_slot, WHITE),
        }
        old_black_label = self.black_player_var.get()
        old_white_label = self.white_player_var.get()
        self.black_player_var.set(old_white_label)
        self.white_player_var.set(old_black_label)
        self.swap_performed = True
        if self.competition_protocol_var.get():
            self.protocol_phase = PHASE_NORMAL_WHITE_4
            self.turn_started_at = time.perf_counter()
        self.last_ai_decision = "three_move_swap"
        self._append_system_log("三手交换：黑白方控制器已交换")
        self.update_status()
        self._write_autosave()
        self.schedule_next_turn()

    def decline_swap(self) -> None:
        if not (self.competition_protocol_var.get() and self.protocol_phase == PHASE_SWAP_DECISION):
            self.last_ai_decision = "不换手不可用"
            self.update_status()
            return
        self.protocol_phase = PHASE_NORMAL_WHITE_4
        self.turn_started_at = time.perf_counter()
        self.last_ai_decision = "no_swap"
        self._append_system_log("三手交换被拒绝；白4落子")
        self.update_status()
        self._write_autosave()
        self.schedule_next_turn()

    def _slot_with_color(self, slot: PlayerSlot, color: int) -> PlayerSlot:
        return PlayerSlot(
            slot.player_type,
            int(color),
            slot.player,
            slot.label,
            slot.is_human,
            is_external=slot.is_external,
        )

    # ------------------------------------------------------------------
    # 26-opening selector
    # ------------------------------------------------------------------
    def _on_opening_selected(self, event: object = None) -> None:
        """Called when the user picks an opening from the combobox."""
        name = self.opening_var.get()
        if not name:
            return
        self._apply_opening(name)

    def _apply_opening(self, name: str) -> None:
        """Auto-place the first three stones for a standard opening."""
        coords = STANDARD_OPENING_26.get(name)
        if coords is None:
            return
        w2_x, w2_y, b3_x, b3_y = coords

        # Only meaningful when the protocol checkbox is on and the board is
        # in the designated-opening phase.
        if not self.competition_protocol_var.get():
            return
        if self.protocol_phase not in (
            PHASE_DESIGNATED_BLACK_1,
            PHASE_DESIGNATED_WHITE_2,
            PHASE_DESIGNATED_BLACK_3,
        ):
            return

        if self.board.move_count != 0:
            return  # only at the very start of a game

        self.paused = False

        # Place Black 1 (always centre).
        center = self.board_size // 2
        self.board.place_stone(center, center)
        self._record_opening_move(center, center, BLACK, "opening:black1")
        self.draw_board()

        # Place White 2.
        self.board.place_stone(w2_x, w2_y)
        self._record_opening_move(w2_x, w2_y, WHITE, f"opening:white2:{name}")
        self.draw_board()

        # Place Black 3.
        self.board.place_stone(b3_x, b3_y)
        self._record_opening_move(b3_x, b3_y, BLACK, f"opening:black3:{name}")
        self.draw_board()

        # Advance to the swap-decision phase.
        self.protocol_phase = PHASE_SWAP_DECISION
        self.turn_started_at = time.perf_counter()
        self.last_ai_decision = f"opening:{name}"
        self._append_system_log(f"已应用指定开局：{name}")
        self.update_status()
        self._write_autosave()
        self.schedule_next_turn()

    def _record_opening_move(self, x: int, y: int, color: int, reason: str) -> None:
        """Record an opening stone in the move-history."""
        from game.encoder import action_to_index
        from game.coordinates import index_to_coord

        self.move_history.append(
            {
                "move": len(self.move_history) + 1,
                "source": "opening",
                "player_label": "Opening",
                "color": "black" if color == BLACK else "white",
                "clock_color": "black" if color == BLACK else "white",
                "coord": index_to_coord(x, y),
                "x": x,
                "y": y,
                "action": action_to_index(x, y, self.board_size),
                "decision_reason": reason,
                "elapsed_seconds": 0.0,
                "is_external_ai": False,
            }
        )

    def restart_game(self, rebuild_players: bool = False, write_autosave: bool = True) -> None:
        if rebuild_players and not self.rebuild_players():
            return
        self.board = Board()
        self.game_over = False
        self.ai_thinking = False
        self.paused = True  # wait for user to click "开始"
        self.move_history = []
        self.last_ai_decision = "-"
        self.result_text = "ongoing"
        now = time.perf_counter()
        self.game_started_at = now
        self.turn_started_at = now
        self.elapsed_by_color = {BLACK: 0.0, WHITE: 0.0}
        self.swap_performed = False
        self.black5_candidate_actions = []
        self._clock_active = False  # frozen until the first move
        self.protocol_phase = (
            PHASE_DESIGNATED_BLACK_1 if self.competition_protocol_var.get() else PHASE_NORMAL_PLAY
        )
        self.opening_var.set("")
        self._clear_log()
        self.draw_board()
        self._update_timer_display()
        self.update_status()
        # Do NOT call schedule_next_turn() here — the user must click "开始" first.
        if write_autosave:
            self._write_autosave()

    def current_slot(self) -> PlayerSlot:
        if self.competition_protocol_var.get():
            if self.protocol_phase in (PHASE_DESIGNATED_WHITE_2, PHASE_DESIGNATED_BLACK_3):
                return self.players[BLACK]
            if self.protocol_phase == PHASE_SWAP_DECISION:
                return self.players[WHITE]
            if self.protocol_phase == PHASE_BLACK5_CANDIDATES:
                return self.players[BLACK]
            if self.protocol_phase == PHASE_WHITE_SELECT_BLACK5:
                return self.players[WHITE]
        return self.players[self.board.current_player]

    def draw_board(self) -> None:
        self.canvas.delete("all")
        for index in range(self.board_size):
            start_x, start_y = board_to_canvas(0, index, margin=self.margin, cell_size=self.cell_size)
            end_x, end_y = board_to_canvas(
                self.board_size - 1, index, margin=self.margin, cell_size=self.cell_size
            )
            self.canvas.create_line(start_x, start_y, end_x, end_y, fill="#4a2d12")
            start_x, start_y = board_to_canvas(index, 0, margin=self.margin, cell_size=self.cell_size)
            end_x, end_y = board_to_canvas(
                index, self.board_size - 1, margin=self.margin, cell_size=self.cell_size
            )
            self.canvas.create_line(start_x, start_y, end_x, end_y, fill="#4a2d12")

        for index in range(self.board_size):
            x, _ = board_to_canvas(index, self.board_size - 1, margin=self.margin, cell_size=self.cell_size)
            self.canvas.create_text(x, self.margin + self.cell_size * self.board_size - 2, text=chr(ord("A") + index))
            _, y = board_to_canvas(0, index, margin=self.margin, cell_size=self.cell_size)
            self.canvas.create_text(self.margin - 18, y, text=str(index + 1))

        for x in range(self.board_size):
            for y in range(self.board_size):
                color = self.board.grid[x][y]
                if color != EMPTY:
                    self._draw_stone(x, y, color)

        if self.board.last_move is not None:
            x, y, _color = self.board.last_move
            cx, cy = board_to_canvas(
                x,
                y,
                margin=self.margin,
                cell_size=self.cell_size,
                board_size=self.board_size,
            )
            self.canvas.create_rectangle(cx - 6, cy - 6, cx + 6, cy + 6, outline="red", width=2)
        for action in self.black5_candidate_actions:
            x, y = index_to_action(action, self.board_size)
            cx, cy = board_to_canvas(
                x,
                y,
                margin=self.margin,
                cell_size=self.cell_size,
                board_size=self.board_size,
            )
            self.canvas.create_oval(
                cx - self.stone_radius,
                cy - self.stone_radius,
                cx + self.stone_radius,
                cy + self.stone_radius,
                fill="",
                outline="#1f77b4",
                width=3,
            )
            self.canvas.create_text(cx, cy, text=str(self.black5_candidate_actions.index(action) + 1), fill="#1f77b4")

    def _draw_stone(self, x: int, y: int, color: int) -> None:
        cx, cy = board_to_canvas(x, y, margin=self.margin, cell_size=self.cell_size)
        fill = "black" if color == BLACK else "white"
        self.canvas.create_oval(
            cx - self.stone_radius,
            cy - self.stone_radius,
            cx + self.stone_radius,
            cy + self.stone_radius,
            fill=fill,
            outline="black",
            width=2,
        )

    def on_canvas_click(self, event) -> None:
        if self.game_over or self.ai_thinking:
            return
        slot = self.current_slot()
        if not slot.is_human:
            return
        position = canvas_to_board(
            event.x,
            event.y,
            margin=self.margin,
            cell_size=self.cell_size,
            board_size=self.board_size,
        )
        if position is None:
            return
        x, y = position
        if self._is_forbidden_point_for_current_player(x, y):
            self._forfeit_forbidden(slot, action_to_index(x, y, self.board_size))
            return
        if not self._is_legal_action_for_current_player(x, y):
            # Give feedback when a click is rejected in a protocol phase.
            action = action_to_index(x, y, self.board_size)
            if self.protocol_phase == PHASE_BLACK5_CANDIDATES:
                if action in self.black5_candidate_actions:
                    self.last_ai_decision = "已添加此候选"
                elif self._is_forbidden_action(action, BLACK):
                    self.last_ai_decision = "禁手点"
                else:
                    self.last_ai_decision = "非法落子"
            else:
                self.last_ai_decision = "非法落子"
            self.update_status()
            return
        elapsed = max(0.0, time.perf_counter() - self.turn_started_at)
        if self._would_exceed_time(slot, elapsed):
            self._forfeit_on_time(slot)
            return
        if self.protocol_phase == PHASE_BLACK5_CANDIDATES:
            self._add_black5_candidate(x, y, slot, elapsed)
            return
        if self.protocol_phase == PHASE_WHITE_SELECT_BLACK5:
            self._select_black5_candidate(x, y, slot, elapsed)
            return
        self._place_move(x, y, slot, decision_reason="human", elapsed=elapsed)
        self.schedule_next_turn()

    def _is_legal_action_for_current_player(self, x: int, y: int) -> bool:
        if not self.board.is_legal_move(x, y):
            return False
        if self.competition_protocol_var.get():
            action = action_to_index(x, y, self.board_size)
            if self.protocol_phase == PHASE_DESIGNATED_BLACK_1:
                center = self.board_size // 2
                return (x, y) == (center, center)
            if self.protocol_phase == PHASE_DESIGNATED_WHITE_2:
                center = self.board_size // 2
                return max(abs(x - center), abs(y - center)) == 1
            if self.protocol_phase == PHASE_DESIGNATED_BLACK_3:
                center = self.board_size // 2
                return center - 2 <= x <= center + 2 and center - 2 <= y <= center + 2
            if self.protocol_phase == PHASE_SWAP_DECISION:
                return False
            if self.protocol_phase == PHASE_BLACK5_CANDIDATES:
                return action not in self.black5_candidate_actions and not self._is_forbidden_action(action, BLACK)
            if self.protocol_phase == PHASE_NORMAL_WHITE_4:
                return self.board.current_player == WHITE
            if self.protocol_phase == PHASE_WHITE_SELECT_BLACK5:
                return action in self.black5_candidate_actions
        if self.board.move_count == 0:
            center = self.board_size // 2
            if (x, y) != (center, center):
                return False
        action = action_to_index(x, y, self.board_size)
        return not self._is_forbidden_action(action, self.board.current_player)

    def _is_forbidden_point_for_current_player(self, x: int, y: int) -> bool:
        if not self.board.is_legal_move(x, y):
            return False
        action = action_to_index(x, y, self.board_size)
        color = BLACK if self.protocol_phase in (PHASE_BLACK5_CANDIDATES, PHASE_WHITE_SELECT_BLACK5) else self.board.current_player
        return self._is_forbidden_action(action, color)

    def undo_human_move(self) -> None:
        """Undo only while a human slot is to move.

        Human-vs-AI undo removes the last AI reply and the last human move so
        the same human can choose again. Human-vs-Human undo removes one move.
        AI-vs-AI has no human turn, so undo is disabled by rule.
        """

        if self.ai_thinking:
            self.last_ai_decision = "悔棋不可用：AI思考中"
            self.update_status()
            return
        if not self.current_slot().is_human:
            self.last_ai_decision = "悔棋仅限人类回合"
            self.update_status()
            return
        if not self.move_history:
            self.last_ai_decision = "无可悔棋"
            self.update_status()
            return

        black_human = self.players[BLACK].is_human
        white_human = self.players[WHITE].is_human
        undo_count = 1
        if not (black_human and white_human):
            undo_count = self._human_vs_ai_undo_count()
            if undo_count <= 0:
                self.last_ai_decision = "无人类落子可悔"
                self.update_status()
                return

        removed = self.move_history[-undo_count:]
        self.move_history = self.move_history[:-undo_count]
        self._rebuild_board_from_history()
        self.game_over = False
        self.ai_thinking = False
        self.paused = False
        self.turn_started_at = time.perf_counter()
        coords = ",".join(move["coord"] for move in removed)
        self.last_ai_decision = f"undo:{coords}"
        self._rewrite_log()
        self.draw_board()
        self.update_status()
        self._write_autosave()
        self.schedule_next_turn()

    def _human_vs_ai_undo_count(self) -> int:
        count = 0
        for move in reversed(self.move_history):
            count += 1
            if move.get("source") == "human":
                return count
        return 0

    def _rebuild_board_from_history(self) -> None:
        rebuilt = Board()
        self.elapsed_by_color = {BLACK: 0.0, WHITE: 0.0}
        for move in self.move_history:
            expected = BLACK if move["color"] == "black" else WHITE
            if rebuilt.current_player != expected:
                raise RuntimeError("move history color order is inconsistent")
            rebuilt.place_stone(int(move["x"]), int(move["y"]))
            clock_color = BLACK if move.get("clock_color", move["color"]) == "black" else WHITE
            self.elapsed_by_color[clock_color] += float(move.get("elapsed_seconds", 0.0))
        self.board = rebuilt

    def _is_forbidden_action(self, action: int, color: int) -> bool:
        if self.rule_mode_var.get() != "forbidden" or color != BLACK:
            return False
        from engine.threats import is_forbidden_action

        return bool(is_forbidden_action(self.board, action, color, "forbidden"))

    def _add_black5_candidate(self, x: int, y: int, slot: PlayerSlot, elapsed: float) -> None:
        action = action_to_index(x, y, self.board_size)
        if action in self.black5_candidate_actions:
            self.last_ai_decision = "此候选已存在"
            self.update_status()
            return
        self.black5_candidate_actions.append(action)
        required = self._fifth_n()
        self.last_ai_decision = f"black5_candidate:{index_to_coord(x, y)} ({len(self.black5_candidate_actions)}/{required})"
        self._append_system_log(self.last_ai_decision)
        if len(self.black5_candidate_actions) >= required:
            self.elapsed_by_color[slot.color] = self.elapsed_by_color.get(slot.color, 0.0) + float(elapsed)
            self.protocol_phase = PHASE_WHITE_SELECT_BLACK5
            self.turn_started_at = time.perf_counter()
            self._append_system_log("black 5 candidates submitted; white selects one to remain")
        self.draw_board()
        self.update_status()
        self._write_autosave()
        self.schedule_next_turn()

    def _select_black5_candidate(self, x: int, y: int, slot: PlayerSlot, elapsed: float) -> None:
        action = action_to_index(x, y, self.board_size)
        if action not in self.black5_candidate_actions:
            return
        self.last_ai_decision = f"select_black5:{index_to_coord(x, y)}"
        self._place_move(x, y, slot, decision_reason="select_black5", elapsed=elapsed)
        self.black5_candidate_actions = []
        self.protocol_phase = PHASE_NORMAL_PLAY
        self._append_system_log("black 5 selected; normal play begins")
        self.draw_board()
        self.update_status()
        self._write_autosave()
        self.schedule_next_turn()

    def _fifth_n(self) -> int:
        try:
            return max(2, min(5, int(self.fifth_n_var.get())))
        except Exception:
            return 2

    def _place_move(
        self,
        x: int,
        y: int,
        slot: PlayerSlot,
        *,
        decision_reason: str = "-",
        elapsed: float = 0.0,
    ) -> None:
        # The chess clock is frozen until the very first move is played.
        # After that it behaves as a standard countdown where each side's
        # time ticks only during its own turn.
        if not self._clock_active:
            self._clock_active = True
            self.turn_started_at = time.perf_counter()

        color = self.board.current_player
        self.board.place_stone(x, y)
        action = action_to_index(x, y, self.board_size)
        clock_color = slot.color if slot.color in (BLACK, WHITE) else color
        self.elapsed_by_color[clock_color] = self.elapsed_by_color.get(clock_color, 0.0) + float(elapsed)
        move_info = {
            "move": self.board.move_count,
            "source": slot.player_type,
            "player_label": slot.label,
            "color": "black" if color == BLACK else "white",
            "clock_color": "black" if clock_color == BLACK else "white",
            "coord": index_to_coord(x, y),
            "x": x,
            "y": y,
            "action": action,
            "decision_reason": decision_reason,
            "elapsed_seconds": round(float(elapsed), 4),
            "is_external_ai": bool(slot.is_external),
        }
        self.move_history.append(move_info)
        self._append_log(move_info)
        self.draw_board()
        self._check_game_over()
        self._advance_protocol_after_move()
        self.turn_started_at = time.perf_counter()
        self.update_status()
        self._write_autosave()

    def _advance_protocol_after_move(self) -> None:
        if not self.competition_protocol_var.get() or self.game_over:
            return
        if self.protocol_phase == PHASE_DESIGNATED_BLACK_1:
            self.protocol_phase = PHASE_DESIGNATED_WHITE_2
        elif self.protocol_phase == PHASE_DESIGNATED_WHITE_2:
            self.protocol_phase = PHASE_DESIGNATED_BLACK_3
        elif self.protocol_phase == PHASE_DESIGNATED_BLACK_3:
            self.protocol_phase = PHASE_SWAP_DECISION
        elif self.protocol_phase == PHASE_NORMAL_WHITE_4:
            self.protocol_phase = PHASE_BLACK5_CANDIDATES
            self.black5_candidate_actions = []
        elif self.protocol_phase == PHASE_WHITE_SELECT_BLACK5:
            self.protocol_phase = PHASE_NORMAL_PLAY

    def _check_game_over(self) -> None:
        if self.rule_mode_var.get() == "forbidden":
            result = get_game_result_forbidden(self.board, self.board.last_move)
            if result.is_over:
                self.game_over = True
                self._show_result(result.winner, result.reason)
            return
        winner = check_winner(self.board, self.board.last_move)
        if winner == 0:
            winner = _find_any_winner(self.board)
        if winner != 0:
            self.game_over = True
            self._show_result(winner, "five")
        elif is_game_over(self.board, self.board.last_move):
            self.game_over = True
            self._show_result(0, "draw")

    def _show_result(self, winner: Optional[int], reason: str) -> None:
        if winner == BLACK:
            text = f"黑胜 ({reason})"
        elif winner == WHITE:
            text = f"白胜 ({reason})"
        else:
            text = f"平局 ({reason})"
        self.result_text = text
        self.status_var.set(text)
        messagebox.showinfo("对局结束", text)

    def schedule_next_turn(self) -> None:
        if self.game_over or self.ai_thinking or self.paused:
            return
        if self._check_clock_timeout():
            return

        delay = max(0, int(self.move_delay_var.get()))

        # Swap decision belongs to the White controller (the side that may take
        # Black).  An AI decides automatically; a human clicks 换手/不换手.
        if self.protocol_phase == PHASE_SWAP_DECISION:
            if not self.players[WHITE].is_human:
                self.root.after(delay, self.request_ai_swap_decision)
            else:
                self.update_status()
            return

        # Black-5 candidates are proposed by the Black controller.  An AI
        # proposes automatically; a human clicks N points on the board.
        if self.protocol_phase == PHASE_BLACK5_CANDIDATES:
            if not self.players[BLACK].is_human:
                self.root.after(delay, self.request_ai_black5_candidates)
            else:
                self.update_status()
            return

        # White selects among black-5: AI if White is AI.
        if self.protocol_phase == PHASE_WHITE_SELECT_BLACK5:
            if not self.players[WHITE].is_human:
                self.root.after(delay, self.request_ai_select_black5)
            else:
                self.update_status()
            return

        # Normal play.
        slot = self.current_slot()
        self.update_status()
        if slot.is_human:
            return
        self.root.after(delay, self.request_ai_move)

    def request_ai_move(self, async_move: bool = True, single_step: bool = False) -> None:
        if self.game_over or self.ai_thinking:
            return
        slot = self.current_slot()
        if slot.is_human:
            return
        if slot.player is None:
            self.last_ai_decision = "AI未加载"
            self.update_status()
            return
        if not async_move:
            self.execute_current_ai_move_sync(single_step=single_step)
            return
        self.ai_thinking = True
        self.update_status()
        board_copy = self.board.copy()
        color = self.board.current_player
        thread = threading.Thread(
            target=self._ai_worker,
            args=(slot, board_copy, color, single_step),
            daemon=True,
        )
        thread.start()

    def request_ai_swap_decision(self) -> None:
        """Let the AI White controller decide the three-move swap automatically."""
        if self.game_over or self.ai_thinking or self.protocol_phase != PHASE_SWAP_DECISION:
            return
        if self.players[WHITE].is_human:
            return
        swap = self._choose_swap_decision()
        if swap:
            self.last_ai_decision = "AI换手：交换执黑"
            self.swap_players()
        else:
            self.last_ai_decision = "AI判断：不换手"
            self.decline_swap()

    def _choose_swap_decision(self) -> bool:
        """White judges whether to swap and take over the Black side.

        Swap when the current position favours Black at least as much as White,
        i.e. the best Black continuation scores no lower than the best White
        continuation.  Falls back to "no swap" if evaluation is unavailable.
        """
        try:
            from engine.candidate_moves import generate_candidate_moves
            from engine.heuristic import evaluate_move_heuristic

            rule = self.rule_mode_var.get()
            candidates = generate_candidate_moves(self.board, radius=2, max_candidates=40)
            if not candidates:
                candidates = [
                    action_to_index(x, y, self.board_size)
                    for x, y in self.board.get_legal_moves()
                ]
            if not candidates:
                return False
            best_black = max(
                evaluate_move_heuristic(self.board, action, BLACK, rule)
                for action in candidates
            )
            best_white = max(
                evaluate_move_heuristic(self.board, action, WHITE, rule)
                for action in candidates
            )
            return best_black >= best_white
        except Exception:
            return False

    def request_ai_black5_candidates(self) -> None:
        if self.game_over or self.ai_thinking or self.protocol_phase != PHASE_BLACK5_CANDIDATES:
            return
        # The Black controller proposes the N fifth-move candidates.  Only an AI
        # controller may auto-generate them; a human proposes by clicking.
        slot = self.players[BLACK]
        if slot.is_human:
            return
        self.ai_thinking = True
        self.update_status()
        board_copy = self.board.copy()
        thread = threading.Thread(
            target=self._ai_black5_candidates_worker,
            args=(slot, board_copy),
            daemon=True,
        )
        thread.start()

    def _ai_black5_candidates_worker(self, slot: PlayerSlot, board_copy: Board) -> None:
        start = time.perf_counter()
        try:
            actions = self._choose_black5_candidates(board_copy)
            elapsed = time.perf_counter() - start
            self.root.after(0, lambda: self._finish_ai_black5_candidates(actions, slot, elapsed))
        except Exception as exc:
            self.root.after(0, lambda: self._finish_ai_error(exc))

    def _finish_ai_black5_candidates(self, actions: list[int], slot: PlayerSlot, elapsed: float) -> None:
        self.ai_thinking = False
        if self.game_over:
            return
        if self._would_exceed_time(slot, elapsed):
            self._forfeit_on_time(slot)
            return
        required = self._fifth_n()
        legal_actions = []
        for action in actions:
            x, y = index_to_action(action, self.board_size)
            if self.board.is_legal_move(x, y) and not self._is_forbidden_action(action, BLACK):
                if action not in legal_actions:
                    legal_actions.append(action)
            if len(legal_actions) >= required:
                break

        # If the primary generator didn't produce enough candidates (e.g.
        # forbidden rule filtered everything), fall back to a simple
        # distance-from-centre scan so the game never gets stuck.
        if len(legal_actions) < required:
            center = self.board_size // 2
            all_legal = [
                action_to_index(x, y, self.board_size)
                for x, y in self.board.get_legal_moves()
                if not self._is_forbidden_action(action_to_index(x, y, self.board_size), BLACK)
            ]
            all_legal.sort(
                key=lambda a: (
                    abs(index_to_action(a, self.board_size)[0] - center)
                    + abs(index_to_action(a, self.board_size)[1] - center),
                    a,
                )
            )
            for a in all_legal:
                if a not in legal_actions:
                    legal_actions.append(a)
                if len(legal_actions) >= required:
                    break

        self.black5_candidate_actions = legal_actions[:required]
        self.elapsed_by_color[slot.color] = self.elapsed_by_color.get(slot.color, 0.0) + float(elapsed)
        self.protocol_phase = PHASE_WHITE_SELECT_BLACK5
        self.turn_started_at = time.perf_counter()
        coords = ",".join(index_to_coord(*index_to_action(action, self.board_size)) for action in self.black5_candidate_actions)
        self.last_ai_decision = f"black5_candidates:{coords}"
        self._append_system_log(f"AI submitted black 5 candidates: {coords}")
        self.draw_board()
        self.update_status()
        self._write_autosave()
        self.schedule_next_turn()

    def _choose_black5_candidates(self, board: Board) -> list[int]:
        required = self._fifth_n()
        try:
            from engine.candidate_moves import generate_candidate_moves
            from engine.heuristic import evaluate_move_heuristic

            candidates = generate_candidate_moves(board, radius=2, max_candidates=80)
            if len(candidates) < required:
                candidates = [
                    action_to_index(x, y, self.board_size)
                    for x, y in board.get_legal_moves()
                ]
            candidates = [
                action for action in candidates
                if board.is_legal_move(*index_to_action(action, self.board_size))
                and not self._is_forbidden_action(action, BLACK)
            ]
            candidates.sort(
                key=lambda action: (
                    -evaluate_move_heuristic(board, action, BLACK, self.rule_mode_var.get()),
                    action,
                )
            )
            return [int(action) for action in candidates[:required]]
        except Exception:
            center = self.board_size // 2
            legal = [
                action_to_index(x, y, self.board_size)
                for x, y in board.get_legal_moves()
                if not self._is_forbidden_action(action_to_index(x, y, self.board_size), BLACK)
            ]
            legal.sort(
                key=lambda action: (
                    abs(index_to_action(action, self.board_size)[0] - center)
                    + abs(index_to_action(action, self.board_size)[1] - center),
                    action,
                )
            )
            return legal[:required]

    def request_ai_select_black5(self) -> None:
        if self.game_over or self.ai_thinking or self.protocol_phase != PHASE_WHITE_SELECT_BLACK5:
            return
        slot = self.current_slot()
        if slot.is_human:
            return
        self.ai_thinking = True
        self.update_status()
        board_copy = self.board.copy()
        thread = threading.Thread(
            target=self._ai_select_black5_worker,
            args=(slot, board_copy),
            daemon=True,
        )
        thread.start()

    def _ai_select_black5_worker(self, slot: PlayerSlot, board_copy: Board) -> None:
        start = time.perf_counter()
        try:
            action = self._choose_black5_selection(board_copy)
            elapsed = time.perf_counter() - start
            self.root.after(0, lambda: self._finish_ai_select_black5(action, slot, elapsed))
        except Exception as exc:
            self.root.after(0, lambda: self._finish_ai_error(exc))

    def _finish_ai_select_black5(self, action: int, slot: PlayerSlot, elapsed: float) -> None:
        self.ai_thinking = False
        if self.game_over:
            return
        if self._would_exceed_time(slot, elapsed):
            self._forfeit_on_time(slot)
            return
        x, y = index_to_action(action, self.board_size)
        if action not in self.black5_candidate_actions:
            self.last_ai_decision = "非法黑5选择"
            self.update_status()
            return
        self._select_black5_candidate(x, y, slot, elapsed)

    def _choose_black5_selection(self, board: Board) -> int:
        try:
            from engine.heuristic import evaluate_move_heuristic

            return min(
                self.black5_candidate_actions,
                key=lambda action: (
                    evaluate_move_heuristic(board, action, BLACK, self.rule_mode_var.get()),
                    action,
                ),
            )
        except Exception:
            return int(self.black5_candidate_actions[0])

    def execute_current_ai_move_sync(self, single_step: bool = True) -> bool:
        if self.game_over or self.ai_thinking:
            return False
        slot = self.current_slot()
        if slot.is_human or slot.player is None:
            return False
        board_copy = self.board.copy()
        start = time.perf_counter()
        try:
            action = slot.player.select_action(board_copy)
            reason = self._extract_decision_reason(slot.player)
            elapsed = time.perf_counter() - start
        except Exception as exc:
            self._finish_ai_error(exc)
            return False
        self._finish_ai_move(action, reason, slot, elapsed, single_step=single_step)
        return True

    def _ai_worker(self, slot: PlayerSlot, board_copy: Board, color: int, single_step: bool) -> None:
        start = time.perf_counter()
        try:
            action = slot.player.select_action(board_copy) if slot.player is not None else None
            reason = self._extract_decision_reason(slot.player)
            elapsed = time.perf_counter() - start
            self.root.after(0, lambda: self._finish_ai_move(action, reason, slot, elapsed, single_step=single_step))
        except Exception as exc:
            self.root.after(0, lambda: self._finish_ai_error(exc))

    def _extract_decision_reason(self, player: object | None) -> str:
        if player is None:
            return "-"
        return (
            getattr(player, "decision_reason", None)
            or getattr(player, "last_decision_reason", None)
            or "-"
        )

    def _finish_ai_error(self, exc: Exception) -> None:
        self.ai_thinking = False
        self.last_ai_decision = f"AI错误：{exc}"
        self.update_status()
        messagebox.showerror("AI错误", str(exc))

    def _finish_ai_move(
        self,
        action: Optional[int],
        reason: str,
        slot: PlayerSlot,
        elapsed: float,
        *,
        single_step: bool = False,
    ) -> None:
        self.ai_thinking = False
        if self.game_over:
            return
        if action is None:
            self.last_ai_decision = "无合法落子"
            self.game_over = True
            self.update_status()
            return
        if self._would_exceed_time(slot, elapsed):
            self._forfeit_on_time(slot)
            return
        try:
            x, y = index_to_action(int(action), self.board_size)
        except ValueError:
            self.last_ai_decision = f"illegal_action:{action}"
            self.update_status()
            messagebox.showerror("AI非法操作", str(action))
            return
        if self._is_forbidden_point_for_current_player(x, y):
            self._forfeit_forbidden(slot, int(action))
            return
        if not self._is_legal_action_for_current_player(x, y):
            self.last_ai_decision = f"illegal:{index_to_coord(x, y)}"
            self.update_status()
            messagebox.showerror("AI非法落子", index_to_coord(x, y))
            return
        self.last_ai_decision = reason
        self._place_move(x, y, slot, decision_reason=reason, elapsed=elapsed)
        if not self.game_over and not single_step:
            self.schedule_next_turn()

    def update_status(self) -> None:
        player = "black" if self.board.current_player == BLACK else "white"
        slot = self.current_slot()
        if hasattr(self, "undo_button"):
            can_undo = (
                bool(self.move_history)
                and slot.is_human
                and not self.ai_thinking
                and not self.game_over
            )
            self.undo_button.configure(state=tk.NORMAL if can_undo else tk.DISABLED)
        if hasattr(self, "swap_button"):
            # In competition mode the swap is decided by the White controller;
            # the manual button is only offered when that side is a human.
            swap_enabled = self.can_swap_players() and (
                not self.competition_protocol_var.get() or self.players[WHITE].is_human
            )
            self.swap_button.configure(state=tk.NORMAL if swap_enabled else tk.DISABLED)
        if hasattr(self, "no_swap_button"):
            can_decline = (
                self.competition_protocol_var.get()
                and self.protocol_phase == PHASE_SWAP_DECISION
                and self.players[WHITE].is_human
            )
            self.no_swap_button.configure(state=tk.NORMAL if can_decline else tk.DISABLED)
        thinking = " | AI思考中…" if self.ai_thinking else ""
        paused = " | 已暂停" if self.paused else ""
        last = "-"
        if self.board.last_move is not None:
            x, y, _ = self.board.last_move
            last = index_to_coord(x, y)
        total_elapsed = max(0.0, time.perf_counter() - self.game_started_at)
        turn_elapsed = 0.0 if self.game_over else max(0.0, time.perf_counter() - self.turn_started_at)
        black_left = self._remaining_seconds(BLACK)
        white_left = self._remaining_seconds(WHITE)
        protocol = ""
        if self.competition_protocol_var.get():
            protocol = f" | 阶段={self.protocol_phase} | N={self._fifth_n()} | 候选={len(self.black5_candidate_actions)}"
        result = f" | 结果={self.result_text}" if self.game_over else ""
        self.status_var.set(
            f"规则={self.rule_mode_var.get()} | 设备={self.device} | "
            f"黑={self.players[BLACK].label} | 白={self.players[WHITE].label} | "
            f"当前={player}:{slot.label} | 手数={self.board.move_count} | "
            f"上步={last} | 决策={self.last_ai_decision} | "
            f"总用时={format_seconds(total_elapsed)} | "
            f"黑已用={format_seconds(self.elapsed_by_color.get(BLACK, 0.0))} | "
            f"白已用={format_seconds(self.elapsed_by_color.get(WHITE, 0.0))} | "
            f"黑剩余={format_seconds(black_left)} | 白剩余={format_seconds(white_left)} | "
            f"回合={format_seconds(turn_elapsed)}{protocol}{result}{thinking}{paused}"
        )

    def _schedule_timer_tick(self) -> None:
        if self._timer_job is not None:
            try:
                self.root.after_cancel(self._timer_job)
            except Exception:
                pass
        self._timer_job = self.root.after(1000, self._timer_tick)

    def _timer_tick(self) -> None:
        if self._clock_active:
            self._check_clock_timeout()
        self._update_timer_display()
        self.update_status()
        self._timer_job = self.root.after(500, self._timer_tick)

    # ------------------------------------------------------------------
    # countdown display
    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_mmss(seconds: float) -> str:
        """Format seconds as ``MM:SS`` (countdown style)."""
        if seconds <= 0:
            return "00:00"
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    def _update_timer_display(self) -> None:
        """Refresh the countdown labels and highlight the active side."""
        if not self._clock_active:
            # Clock is frozen — show static initial time for both sides.
            self._black_timer_label.configure(
                text=self._fmt_mmss(self.time_limit_seconds), fg="#222"
            )
            self._white_timer_label.configure(
                text=self._fmt_mmss(self.time_limit_seconds), fg="#555"
            )
            return

        black_left = self._remaining_seconds(BLACK)
        white_left = self._remaining_seconds(WHITE)

        self._black_timer_label.configure(text=self._fmt_mmss(black_left))
        self._white_timer_label.configure(text=self._fmt_mmss(white_left))

        # Highlight the side that is currently to move.
        if self.game_over:
            self._black_timer_label.configure(fg="#222")
            self._white_timer_label.configure(fg="#555")
            return

        clock_color = self._current_clock_color()
        if clock_color == BLACK:
            self._black_timer_label.configure(fg="#c00")   # red = ticking
            self._white_timer_label.configure(fg="#555")
        else:
            self._black_timer_label.configure(fg="#222")
            self._white_timer_label.configure(fg="#c00")

        # Flash warning when below 60 seconds.
        if black_left <= 60 and clock_color == BLACK:
            self._black_timer_label.configure(
                fg="#f00" if int(time.perf_counter() * 2) % 2 == 0 else "#c00"
            )
        if white_left <= 60 and clock_color == WHITE:
            self._white_timer_label.configure(
                fg="#f00" if int(time.perf_counter() * 2) % 2 == 0 else "#c00"
            )

    def _current_clock_color(self) -> int:
        return self.current_slot().color

    def _used_seconds_including_turn(self, color: int) -> float:
        if not self._clock_active:
            return 0.0
        used = self.elapsed_by_color.get(color, 0.0)
        if not self.game_over and self._current_clock_color() == color:
            used += max(0.0, time.perf_counter() - self.turn_started_at)
        return used

    def _remaining_seconds(self, color: int) -> float:
        return max(0.0, self.time_limit_seconds - self._used_seconds_including_turn(color))

    def _would_exceed_time(self, slot: PlayerSlot, elapsed: float) -> bool:
        return self.elapsed_by_color.get(slot.color, 0.0) + float(elapsed) > self.time_limit_seconds

    def _check_clock_timeout(self) -> bool:
        if self.game_over:
            return False
        color = self._current_clock_color()
        used = self._used_seconds_including_turn(color)
        if not self._clock_active:
            used = self.elapsed_by_color.get(color, 0.0) + max(
                0.0,
                time.perf_counter() - self.turn_started_at,
            )
        if used >= self.time_limit_seconds:
            self._forfeit_on_time(self.current_slot())
            return True
        return False

    def _forfeit_on_time(self, slot: PlayerSlot) -> None:
        winner = -slot.color
        self.game_over = True
        self.result_text = (
            "White wins (black_timeout)" if winner == WHITE else "Black wins (white_timeout)"
        )
        self.last_ai_decision = "超时"
        self._append_system_log(f"{slot.label} loses on time")
        self.update_status()
        self._write_autosave()
        messagebox.showinfo("Game over", self.result_text)

    def _forfeit_forbidden(self, slot: PlayerSlot, action: int) -> None:
        x, y = index_to_action(action, self.board_size)
        self.game_over = True
        self.result_text = "白胜（黑禁手）"
        self.last_ai_decision = f"forbidden:{index_to_coord(x, y)}"
        self._append_system_log(f"black forbidden move at {index_to_coord(x, y)}; white wins")
        self.update_status()
        self._write_autosave()
        messagebox.showinfo("Game over", self.result_text)

    def _append_log(self, move_info: dict) -> None:
        line = (
            f"{move_info['move']:03d} {move_info['color']} {move_info['player_label']} "
            f"{move_info['coord']} action={move_info['action']} "
            f"reason={move_info['decision_reason']} time={move_info['elapsed_seconds']:.3f}s"
        )
        if move_info["is_external_ai"]:
            line += " external_ai=true"
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, line + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _append_system_log(self, message: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"# {message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _clear_log(self) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _rewrite_log(self) -> None:
        self._clear_log()
        for move_info in self.move_history:
            self._append_log(move_info)

    def save_record_json(self) -> None:
        path = filedialog.asksaveasfilename(
            title="保存棋谱JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        self._save_record_json_path(path)
        self.last_ai_decision = f"saved:{path}"
        self.update_status()

    def _build_record_json_payload(self) -> dict:
        return {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "rule_mode": self.rule_mode_var.get(),
            "device": self.device,
            "black_player": self.players[BLACK].player_type,
            "black_player_label": self.players[BLACK].label,
            "white_player": self.players[WHITE].player_type,
            "white_player_label": self.players[WHITE].label,
            "external_ai_path": self.external_ai_path_var.get(),
            "result": self.result_text,
            "swap_performed": self.swap_performed,
            "competition_protocol": bool(self.competition_protocol_var.get()),
            "protocol_phase": self.protocol_phase,
            "fifth_n": self._fifth_n(),
            "black5_candidate_actions": list(self.black5_candidate_actions),
            "time_limit_seconds": self.time_limit_seconds,
            "timing": self._timing_payload(),
            "moves": self.move_history,
        }

    def _save_record_json_path(self, path: str | Path) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._build_record_json_payload()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def _write_autosave(self) -> None:
        try:
            self._save_record_json_path(self.autosave_path)
        except Exception as exc:
            self.last_ai_decision = f"autosave failed: {exc}"

    def load_record_dialog(self) -> None:
        path = filedialog.askopenfilename(
            title="加载棋谱",
            filetypes=[
                ("Record", "*.json *.txt"),
                ("JSON", "*.json"),
                ("C5 record", "*.txt"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self.load_record_path(path)
        except Exception as exc:
            self.last_ai_decision = f"load failed: {exc}"
            self.update_status()
            messagebox.showerror("加载失败", str(exc))

    def load_autosave(self) -> None:
        try:
            self.load_record_path(self.autosave_path)
        except Exception as exc:
            self.last_ai_decision = f"load autosave failed: {exc}"
            self.update_status()
            messagebox.showerror("加载自动存档失败", str(exc))

    def load_record_path(self, path: str | Path) -> None:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(str(file_path))
        if file_path.suffix.lower() == ".json":
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            self._load_json_payload(payload)
        else:
            text = read_record_file(file_path)
            moves = parse_record(text)
            self._load_move_records(moves)
        self.last_ai_decision = f"loaded:{file_path}"
        self.update_status()
        self.schedule_next_turn()

    def _load_json_payload(self, payload: dict) -> None:
        self.paused = True
        self.ai_thinking = False
        self.game_over = False
        self.rule_mode_var.set(payload.get("rule_mode", self.rule_mode_var.get()))
        self.device_var.set(payload.get("device", self.device_var.get()))
        self.external_ai_path_var.set(payload.get("external_ai_path", self.external_ai_path_var.get()))
        self.competition_protocol_var.set(bool(payload.get("competition_protocol", False)))
        self.fifth_n_var.set(int(payload.get("fifth_n", self.fifth_n_var.get())))
        self.time_limit_seconds = float(payload.get("time_limit_seconds", TIME_LIMIT_SECONDS) or TIME_LIMIT_SECONDS)
        self._set_player_var_from_type(self.black_player_var, payload.get("black_player", "human"))
        self._set_player_var_from_type(self.white_player_var, payload.get("white_player", "alphaone_mini"))
        if not self.rebuild_players():
            raise RuntimeError("failed to rebuild players while loading record")
        self.swap_performed = bool(payload.get("swap_performed", False))
        self.protocol_phase = str(payload.get("protocol_phase", PHASE_NORMAL_PLAY))
        self.black5_candidate_actions = [int(action) for action in payload.get("black5_candidate_actions", [])]
        self.move_history = []
        for idx, move in enumerate(payload.get("moves", []), start=1):
            self.move_history.append(self._normalize_loaded_move(move, idx))
        self._rebuild_board_from_history()
        self.result_text = str(payload.get("result") or "ongoing")
        self.game_over = self.result_text != "ongoing"
        timing = payload.get("timing", {})
        total_elapsed = float(timing.get("total_elapsed_seconds", 0.0) or 0.0)
        current_turn_elapsed = float(timing.get("current_turn_elapsed_seconds", 0.0) or 0.0)
        now = time.perf_counter()
        self.game_started_at = now - max(0.0, total_elapsed)
        self.turn_started_at = now - max(0.0, current_turn_elapsed)
        self._rewrite_log()
        self.draw_board()
        self._update_timer_display()
        self._write_autosave()

    def _load_move_records(self, moves: list[MoveRecord]) -> None:
        self.paused = True
        self.ai_thinking = False
        self.game_over = False
        self.move_history = []
        for idx, move in enumerate(moves, start=1):
            self.move_history.append(
                {
                    "move": idx,
                    "source": "loaded_record",
                    "player_label": "Loaded Record",
                    "color": "black" if move.color == BLACK else "white",
                    "coord": index_to_coord(move.x, move.y),
                    "x": int(move.x),
                    "y": int(move.y),
                    "action": action_to_index(move.x, move.y, self.board_size),
                    "decision_reason": "loaded_record",
                    "elapsed_seconds": 0.0,
                    "is_external_ai": False,
                }
            )
        self._rebuild_board_from_history()
        self.result_text = "ongoing"
        self.swap_performed = False
        self.black5_candidate_actions = []
        self.protocol_phase = PHASE_NORMAL_PLAY
        now = time.perf_counter()
        self.game_started_at = now
        self.turn_started_at = now
        self._rewrite_log()
        self.draw_board()
        self._update_timer_display()
        self._write_autosave()

    def _set_player_var_from_type(self, variable: tk.StringVar, player_type: str) -> None:
        try:
            variable.set(PLAYER_TYPE_LABELS[normalize_player_type(player_type)])
        except Exception:
            variable.set(PLAYER_TYPE_LABELS["human"])

    def _normalize_loaded_move(self, move: dict, fallback_move_number: int) -> dict:
        x = int(move.get("x"))
        y = int(move.get("y"))
        color = str(move.get("color", "black"))
        return {
            "move": int(move.get("move", fallback_move_number)),
            "source": str(move.get("source", "loaded_json")),
            "player_label": str(move.get("player_label", "Loaded JSON")),
            "color": color,
            "clock_color": str(move.get("clock_color", color)),
            "coord": str(move.get("coord") or index_to_coord(x, y)),
            "x": x,
            "y": y,
            "action": int(move.get("action", action_to_index(x, y, self.board_size))),
            "decision_reason": str(move.get("decision_reason", "loaded_json")),
            "elapsed_seconds": float(move.get("elapsed_seconds", 0.0) or 0.0),
            "is_external_ai": bool(move.get("is_external_ai", False)),
        }

    def _timing_payload(self) -> dict:
        return {
            "total_elapsed_seconds": round(max(0.0, time.perf_counter() - self.game_started_at), 4),
            "black_elapsed_seconds": round(self.elapsed_by_color.get(BLACK, 0.0), 4),
            "white_elapsed_seconds": round(self.elapsed_by_color.get(WHITE, 0.0), 4),
            "current_turn_elapsed_seconds": round(
                0.0 if self.game_over else max(0.0, time.perf_counter() - self.turn_started_at),
                4,
            ),
        }

    def _move_records(self) -> list[MoveRecord]:
        records: list[MoveRecord] = []
        for move in self.move_history:
            color = BLACK if move["color"] == "black" else WHITE
            x = int(move["x"])
            y = int(move["y"])
            coord = index_to_coord(x, y)
            records.append(MoveRecord(color=color, x=x, y=y, coord=coord, raw=coord))
        return records

    def _record_metadata(self) -> RecordMetadata:
        if "黑胜" in self.result_text:
            result = "先手胜"
        elif "白胜" in self.result_text:
            result = "后手胜"
        elif "平局" in self.result_text:
            result = "平局"
        else:
            result = "未结束"
        return RecordMetadata(
            game_type="C5",
            black_team=f"先手参赛队 {self.players[BLACK].label}",
            white_team=f"后手参赛队 {self.players[WHITE].label}",
            result=result,
            datetime_location=datetime.now().strftime("%Y.%m.%d %H:%M 本地测试"),
            event_name="AlphaOne-Mini",
        )

    def build_record_text(self) -> str:
        """Build the CCGC-style C5 text record shown in the competition spec."""
        return export_standard_record(self._move_records(), self._record_metadata())

    def save_record_text(self) -> None:
        path = filedialog.asksaveasfilename(
            title="保存C5棋谱",
            defaultextension=".txt",
            filetypes=[("C5 record", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        write_record_file(path, self.build_record_text())
        self.last_ai_decision = f"saved_record:{path}"
        self.update_status()


__all__ = ["GomokuTkApp", "board_to_canvas", "canvas_to_board"]
