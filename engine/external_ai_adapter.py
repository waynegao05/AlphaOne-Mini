"""Adapter for loading a local external Gomoku AI.py file."""

from __future__ import annotations

import importlib.util
import inspect
import numbers
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Optional

from game.board import BLACK, BOARD_SIZE, Board
from game.coordinates import coord_to_index, is_valid_coord
from game.encoder import action_to_index, index_to_action


@contextmanager
def temporarily_add_to_syspath(path: str | Path):
    """Temporarily prepend a directory to sys.path for local AI dependencies."""

    resolved = str(Path(path).resolve())
    inserted = False
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(resolved)
            except ValueError:
                pass


class ExternalAIAdapter:
    """Load a user-provided AI.py and expose a stable select_action(board) API."""

    METHOD_NAMES = ("select_action", "get_move", "ai_move")
    CLASS_NAMES = ("AI", "Player", "GomokuAI")
    LEGACY_METHOD_NAMES = ("AI1",)

    def __init__(
        self,
        path: str | Path,
        *,
        rule_mode: str = "basic",
        board_size: int = BOARD_SIZE,
        name: str = "ExternalAI.py",
    ) -> None:
        self.path = Path(path)
        self.rule_mode = rule_mode
        self.board_size = int(board_size)
        self.name = name
        self.decision_reason = "-"
        self.last_error: Optional[str] = None
        self._legacy_method_name: Optional[str] = None
        self._module = self._load_module(self.path)
        self._callable = self._resolve_callable(self._module)

    def _load_module(self, path: Path) -> ModuleType:
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(str(path))
        module_name = f"external_ai_user_{abs(hash(str(path)))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load external AI module: {path}")
        module = importlib.util.module_from_spec(spec)
        graphics_stub = _make_graphics_stub() if _should_stub_graphics(path) else None
        previous_graphics = sys.modules.get("graphics")
        if graphics_stub is not None:
            sys.modules["graphics"] = graphics_stub
        try:
            with temporarily_add_to_syspath(path.parent):
                spec.loader.exec_module(module)
        finally:
            if graphics_stub is not None:
                if previous_graphics is None:
                    sys.modules.pop("graphics", None)
                else:
                    sys.modules["graphics"] = previous_graphics
        return module

    def _resolve_callable(self, module: ModuleType) -> Callable[..., Any]:
        for name in self.METHOD_NAMES:
            fn = getattr(module, name, None)
            if callable(fn):
                return fn

        for class_name in self.CLASS_NAMES:
            cls = getattr(module, class_name, None)
            if cls is None:
                continue
            instance = cls()
            for name in self.METHOD_NAMES:
                fn = getattr(instance, name, None)
                if callable(fn):
                    return fn

        for name in self.LEGACY_METHOD_NAMES:
            fn = getattr(module, name, None)
            if callable(fn):
                self._legacy_method_name = name
                return fn

        raise AttributeError(
            "External AI must provide select_action(board), get_move(board[, color]), "
            "ai_move(board[, color]), class AI/Player/GomokuAI with one of those methods, "
            "or a legacy AI1() decision entry."
        )

    def _call_external(self, board: Board, color: int) -> Any:
        if self._legacy_method_name == "AI1":
            return self._call_legacy_ai1(board, color)
        board_copy = board.copy()
        try:
            signature = inspect.signature(self._callable)
            positional = [
                param
                for param in signature.parameters.values()
                if param.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                and param.default is inspect.Parameter.empty
            ]
            if len(positional) >= 2:
                return self._callable(board_copy, color)
            return self._callable(board_copy)
        except (TypeError, ValueError):
            try:
                return self._callable(board_copy, color)
            except TypeError:
                return self._callable(board_copy)

    def _call_legacy_ai1(self, board: Board, color: int) -> Any:
        """Run old graphics-program AI1() without using its drawing go()."""

        module = self._module
        external_ai = 1
        external_opponent = 2
        module.ai = external_ai
        module.go_first = external_ai if color == BLACK else external_opponent
        module.start = external_ai
        module.num = [[0 for _ in range(16)] for _ in range(16)]
        for x in range(min(self.board_size, BOARD_SIZE)):
            for y in range(min(self.board_size, BOARD_SIZE)):
                stone = board.grid[x][y]
                if stone == color:
                    module.num[x][y] = external_ai
                elif stone == -color:
                    module.num[x][y] = external_opponent

        original_go = getattr(module, "go", None)

        def no_draw_go(x, y):
            return int(x), int(y)

        module.go = no_draw_go
        try:
            return self._callable()
        finally:
            if original_go is not None:
                module.go = original_go

    def _convert_output(self, value: Any) -> int:
        if isinstance(value, numbers.Integral):
            return int(value)

        if isinstance(value, str):
            text = value.strip()
            if text.isdigit():
                return int(text)
            coord = self._parse_coord_string(text)
            if coord is not None:
                x, y = coord
                return action_to_index(x, y, self.board_size)

        if isinstance(value, (tuple, list)) and len(value) >= 2:
            first, second = value[0], value[1]
            if isinstance(first, str):
                text = f"{first}{second}"
                if is_valid_coord(text):
                    x, y = coord_to_index(text)
                    return action_to_index(x, y, self.board_size)
            x, y = int(first), int(second)
            return action_to_index(x, y, self.board_size)

        raise ValueError(f"unsupported external AI move output: {value!r}")

    def _parse_coord_string(self, text: str) -> Optional[tuple[int, int]]:
        clean = text.strip().upper()
        if is_valid_coord(clean):
            return coord_to_index(clean)
        match = re.search(r"([A-O])\s*,?\s*(1[0-5]|[1-9])", clean)
        if match:
            return coord_to_index(f"{match.group(1)}{match.group(2)}")
        return None

    def _is_forbidden(self, board: Board, action: int, color: int) -> bool:
        if self.rule_mode != "forbidden" or color != BLACK:
            return False
        from engine.threats import is_forbidden_action

        return bool(is_forbidden_action(board, action, color, "forbidden"))

    def _is_legal(self, board: Board, action: int, color: int) -> bool:
        try:
            x, y = index_to_action(int(action), self.board_size)
        except ValueError:
            return False
        return board.is_legal_move(x, y) and not self._is_forbidden(board, int(action), color)

    def _fallback_action(self, board: Board, color: int) -> Optional[int]:
        legal: list[int] = []
        for x, y in board.get_legal_moves():
            action = action_to_index(x, y, self.board_size)
            if not self._is_forbidden(board, action, color):
                legal.append(action)
        if not legal:
            return None
        center = self.board_size // 2
        legal.sort(
            key=lambda action: (
                abs(index_to_action(action, self.board_size)[0] - center)
                + abs(index_to_action(action, self.board_size)[1] - center),
                action,
            )
        )
        return int(legal[0])

    def select_action(self, board: Board) -> Optional[int]:
        color = int(board.current_player)
        self.last_error = None
        try:
            raw = self._call_external(board, color)
            action = self._convert_output(raw)
        except Exception as exc:
            self.last_error = str(exc)
            self.decision_reason = f"external_error_fallback:{type(exc).__name__}"
            return self._fallback_action(board, color)

        if self._is_legal(board, action, color):
            self.decision_reason = "external_ai"
            return int(action)

        self.decision_reason = "external_illegal_fallback"
        return self._fallback_action(board, color)

    def select_move(self, board: Board):
        action = self.select_action(board)
        if action is None:
            return None
        return index_to_action(action, self.board_size)


def _should_stub_graphics(ai_path: Path) -> bool:
    try:
        ai_source = ai_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    if "graphics" not in ai_source:
        return False
    graphics_path = ai_path.parent / "graphics.py"
    if not graphics_path.exists():
        return False
    try:
        graphics_source = graphics_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return (
        "class GraphWin" in graphics_source
        and "tkinter" in graphics_source
        and ("Simple object oriented graphics library" in graphics_source or "John Zelle" in graphics_source)
    )


def _make_graphics_stub() -> ModuleType:
    module = ModuleType("graphics")

    class _Dummy:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.closed = False

        def draw(self, *args, **kwargs):
            return self

        def undraw(self, *args, **kwargs):
            return None

        def setFill(self, *args, **kwargs):
            return None

        def setText(self, *args, **kwargs):
            return None

        def setBackground(self, *args, **kwargs):
            return None

        def close(self):
            self.closed = True

        def getMouse(self):
            return Point(0, 0)

        def checkMouse(self):
            return None

        def isClosed(self):
            return self.closed

    class Point(_Dummy):
        def __init__(self, x=0, y=0):
            super().__init__(x, y)
            self.x = x
            self.y = y

        def getX(self):
            return self.x

        def getY(self):
            return self.y

    module.Point = Point
    module.GraphWin = _Dummy
    module.Text = _Dummy
    module.Line = _Dummy
    module.Rectangle = _Dummy
    module.Circle = _Dummy
    module.Oval = _Dummy
    module.Polygon = _Dummy
    module.Entry = _Dummy
    module.Image = _Dummy
    module.GraphicsError = RuntimeError
    module.update = lambda *args, **kwargs: None
    return module


__all__ = ["ExternalAIAdapter", "temporarily_add_to_syspath"]
