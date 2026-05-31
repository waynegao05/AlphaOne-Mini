"""mcts/node.py 与 mcts/mcts.py 的单元 + 集成测试。

依赖 PyTorch；环境没装则整个文件被自动跳过。
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402
import torch.nn as nn  # noqa: E402

from game.board import BLACK, BOARD_SIZE, EMPTY, WHITE, Board  # noqa: E402
from game.encoder import action_to_index, index_to_action  # noqa: E402
from mcts.mcts import MCTS  # noqa: E402
from mcts.node import MCTSNode  # noqa: E402


# ---------------------------------------------------------------------------
# 假模型：让测试稳定可重复
# ---------------------------------------------------------------------------
class FakePolicyValueNet(nn.Module):
    """输出固定 logits 与固定 value 的轻量模型，用于测试 MCTS 的结构逻辑。"""

    def __init__(self, value: float = 0.0, logits: torch.Tensor | None = None) -> None:
        super().__init__()
        # 至少注册一个参数，以便 ``model.eval()`` / ``.to(device)`` 自然可用。
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
        self._fixed_value = float(value)
        if logits is not None:
            assert logits.shape == (BOARD_SIZE * BOARD_SIZE,)
            self.register_buffer("_fixed_logits", logits.float().clone())
        else:
            self.register_buffer(
                "_fixed_logits", torch.zeros(BOARD_SIZE * BOARD_SIZE)
            )

    def forward(self, x):  # type: ignore[override]
        batch = x.shape[0]
        logits = self._fixed_logits.unsqueeze(0).expand(batch, -1).clone()
        value = torch.full(
            (batch, 1), self._fixed_value, dtype=torch.float32, device=x.device
        )
        return logits, value


# ===========================================================================
# 一、MCTSNode
# ===========================================================================
class TestMCTSNode:
    def test_initial_state(self):
        node = MCTSNode(current_player=BLACK)
        assert node.visit_count == 0
        assert node.total_value == 0.0
        assert node.mean_value == 0.0
        assert node.is_leaf()
        assert node.children == {}

    def test_update_increments_count_and_value(self):
        node = MCTSNode(current_player=BLACK)
        node.update(0.5)
        assert node.visit_count == 1
        assert node.total_value == pytest.approx(0.5)
        assert node.mean_value == pytest.approx(0.5)

        node.update(-0.1)
        assert node.visit_count == 2
        assert node.total_value == pytest.approx(0.4)
        assert node.mean_value == pytest.approx(0.2)

    def test_expand_creates_children_with_correct_links(self):
        parent = MCTSNode(current_player=BLACK)
        priors = {0: 0.6, 1: 0.4}
        parent.expand(priors, child_player=WHITE)

        assert not parent.is_leaf()
        assert set(parent.children.keys()) == {0, 1}
        for action, child in parent.children.items():
            assert child.parent is parent
            assert child.action == action
            assert child.current_player == WHITE
            assert child.visit_count == 0
        assert parent.children[0].prior_prob == pytest.approx(0.6)
        assert parent.children[1].prior_prob == pytest.approx(0.4)

    def test_expand_does_not_overwrite_existing_children(self):
        parent = MCTSNode(current_player=BLACK)
        parent.expand({0: 0.5}, child_player=WHITE)
        parent.children[0].update(0.7)  # 模拟一次访问
        # 重复 expand 不应清空已有统计
        parent.expand({0: 0.99, 1: 0.01}, child_player=WHITE)
        assert parent.children[0].visit_count == 1
        assert parent.children[0].prior_prob == pytest.approx(0.5)
        assert 1 in parent.children

    def test_select_child_returns_legal_pair(self):
        parent = MCTSNode(current_player=BLACK)
        parent.update(1.0)  # 模拟根节点的 backup
        parent.expand({0: 0.7, 1: 0.3}, child_player=WHITE)
        action, child = parent.select_child(c_puct=5.0)
        assert action in (0, 1)
        assert child is parent.children[action]
        # 在 Q 都为 0、N 都为 0 的初始状态下，prior 高的应该被选中
        assert action == 0

    def test_select_child_balances_exploitation_and_exploration(self):
        parent = MCTSNode(current_player=BLACK)
        parent.update(10.0)  # 模拟较高的 parent_N
        parent.expand({0: 0.5, 1: 0.5}, child_player=WHITE)
        # 让 child0 看起来很差：N 很高，Q 很低
        parent.children[0].visit_count = 8
        parent.children[0].total_value = -8.0  # Q = -1
        parent.children[1].visit_count = 1
        parent.children[1].total_value = 0.5  # Q = 0.5
        action, _ = parent.select_child(c_puct=5.0)
        assert action == 0

    def test_select_child_uses_parent_perspective_q(self):
        parent = MCTSNode(current_player=BLACK)
        parent.update(1.0)
        parent.expand({0: 0.5, 1: 0.5}, child_player=WHITE)
        parent.children[0].update(-0.9)  # bad for child WHITE, good for parent BLACK
        parent.children[1].update(0.9)   # good for child WHITE, bad for parent BLACK

        action, _ = parent.select_child(c_puct=0.0)

        assert action == 0

    def test_backup_alternates_sign_along_chain(self):
        root = MCTSNode(current_player=BLACK)
        root.expand({0: 1.0}, child_player=WHITE)
        child = root.children[0]
        child.expand({1: 1.0}, child_player=BLACK)
        grandchild = child.children[1]

        grandchild.backup(0.7)

        assert grandchild.visit_count == 1
        assert grandchild.total_value == pytest.approx(0.7)
        assert child.visit_count == 1
        assert child.total_value == pytest.approx(-0.7)
        assert root.visit_count == 1
        assert root.total_value == pytest.approx(0.7)

    def test_has_child_and_get_child(self):
        node = MCTSNode(current_player=BLACK)
        node.expand({3: 0.5, 9: 0.5}, child_player=WHITE)
        assert node.has_child(3)
        assert not node.has_child(7)
        assert node.get_child(3) is node.children[3]
        assert node.get_child(7) is None


# ===========================================================================
# 二、MCTS 输出
# ===========================================================================
class TestMCTSRun:
    def test_can_be_instantiated(self):
        mcts = MCTS(FakePolicyValueNet(), num_simulations=5)
        assert mcts.num_simulations == 5
        assert mcts.board_size == BOARD_SIZE
        assert mcts.action_size == BOARD_SIZE * BOARD_SIZE

    def test_run_output_shape_and_sum(self):
        mcts = MCTS(FakePolicyValueNet(), num_simulations=20)
        board = Board()
        probs = mcts.run(board)
        assert probs.shape == (BOARD_SIZE * BOARD_SIZE,)
        assert probs.dtype == np.float32
        assert (probs >= 0).all()
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)

    def test_run_zeros_illegal_actions(self):
        mcts = MCTS(FakePolicyValueNet(), num_simulations=20)
        board = Board()
        board.place_stone(7, 7)  # H8 黑
        board.place_stone(8, 8)  # I9 白
        probs = mcts.run(board)
        assert probs[action_to_index(7, 7)] == 0.0
        assert probs[action_to_index(8, 8)] == 0.0

    def test_run_does_not_mutate_input_board(self):
        mcts = MCTS(FakePolicyValueNet(), num_simulations=15)
        board = Board()
        board.place_stone(7, 7)
        board.place_stone(8, 8)

        snapshot_grid = [row[:] for row in board.grid]
        snapshot_player = board.current_player
        snapshot_count = board.move_count
        snapshot_last = board.last_move

        mcts.run(board)

        assert board.grid == snapshot_grid
        assert board.current_player == snapshot_player
        assert board.move_count == snapshot_count
        assert board.last_move == snapshot_last

    def test_run_with_real_policy_value_net_works(self):
        from model.policy_value_net import PolicyValueNet  # 延迟导入

        torch.manual_seed(0)
        model = PolicyValueNet()
        model.eval()
        mcts = MCTS(model, num_simulations=8)
        board = Board()
        probs = mcts.run(board)
        assert probs.shape == (BOARD_SIZE * BOARD_SIZE,)
        assert probs.sum() == pytest.approx(1.0, abs=1e-4)

    def test_only_one_legal_move_gets_full_probability(self):
        # 用前两批中的"满盘无 5 连"模式填棋盘，并空出 (0, 0)
        board = Board()
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if (x, y) == (0, 0):
                    continue
                block_index = (x + 2 * y) // 2
                board.grid[x][y] = BLACK if block_index % 2 == 0 else WHITE
        board.move_count = BOARD_SIZE * BOARD_SIZE - 1

        # 根据现存黑白数量决定 current_player(使其与 move_count 自洽)
        blacks = sum(
            1 for x in range(BOARD_SIZE) for y in range(BOARD_SIZE)
            if board.grid[x][y] == BLACK
        )
        whites = board.move_count - blacks
        # 白手数 = 黑手数 - 1 -> 下一手轮到白；白手数 = 黑手数 -> 下一手轮到黑。
        if blacks == whites + 1:
            board.current_player = WHITE
        else:
            board.current_player = BLACK
        # 任意一个非制胜的 last_move
        board.last_move = (BOARD_SIZE - 1, BOARD_SIZE - 1, board.grid[BOARD_SIZE - 1][BOARD_SIZE - 1])

        mcts = MCTS(FakePolicyValueNet(), num_simulations=5)
        probs = mcts.run(board)

        assert probs[action_to_index(0, 0)] == pytest.approx(1.0, abs=1e-5)
        # 其它位置必须严格为 0
        other_total = float(probs.sum() - probs[action_to_index(0, 0)])
        assert other_total == pytest.approx(0.0, abs=1e-6)


class TestSelectAction:
    def test_select_action_returns_legal_index(self):
        mcts = MCTS(FakePolicyValueNet(), num_simulations=12)
        board = Board()
        board.place_stone(7, 7)
        action = mcts.select_action(board, deterministic=True)
        x, y = index_to_action(action)
        assert board.is_legal_move(x, y)

    def test_select_action_deterministic_picks_argmax_of_run(self):
        mcts = MCTS(FakePolicyValueNet(), num_simulations=20)
        board = Board()
        # 给两次相同输入做确定性比较
        torch.manual_seed(123)
        np.random.seed(123)
        probs = mcts.run(board)
        expected = int(np.argmax(probs))

        torch.manual_seed(123)
        np.random.seed(123)
        action = mcts.select_action(board, deterministic=True)
        assert action == expected


# ===========================================================================
# 三、终局
# ===========================================================================
class TestTerminal:
    def _make_black_wins_board(self) -> Board:
        board = Board()
        for x in range(5):
            board.grid[x][0] = BLACK
        board.last_move = (4, 0, BLACK)
        board.move_count = 5
        # 黑刚刚形成 5 连，下一手轮到白
        board.current_player = WHITE
        return board

    def _make_full_no_winner_board(self) -> Board:
        board = Board()
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                block_index = (x + 2 * y) // 2
                board.grid[x][y] = BLACK if block_index % 2 == 0 else WHITE
        board.move_count = BOARD_SIZE * BOARD_SIZE
        last_color = board.grid[BOARD_SIZE - 1][BOARD_SIZE - 1]
        board.last_move = (BOARD_SIZE - 1, BOARD_SIZE - 1, last_color)
        board.current_player = -last_color
        return board

    def test_terminal_value_black_wins_from_black_view(self):
        mcts = MCTS(FakePolicyValueNet(), num_simulations=1)
        board = self._make_black_wins_board()
        assert mcts.get_terminal_value(board, current_player=BLACK) == 1.0

    def test_terminal_value_black_wins_from_white_view(self):
        mcts = MCTS(FakePolicyValueNet(), num_simulations=1)
        board = self._make_black_wins_board()
        assert mcts.get_terminal_value(board, current_player=WHITE) == -1.0

    def test_terminal_value_white_wins_perspectives(self):
        mcts = MCTS(FakePolicyValueNet(), num_simulations=1)
        board = Board()
        for y in range(5):
            board.grid[7][y] = WHITE
        board.last_move = (7, 4, WHITE)
        board.move_count = 5
        board.current_player = BLACK
        assert mcts.get_terminal_value(board, current_player=WHITE) == 1.0
        assert mcts.get_terminal_value(board, current_player=BLACK) == -1.0

    def test_terminal_value_draw_is_zero(self):
        mcts = MCTS(FakePolicyValueNet(), num_simulations=1)
        board = self._make_full_no_winner_board()
        assert mcts.get_terminal_value(board, current_player=BLACK) == 0.0
        assert mcts.get_terminal_value(board, current_player=WHITE) == 0.0

    def test_terminal_root_does_not_call_model(self):
        """已经胜利的局面跑 MCTS 时不应调用 evaluate_leaf / 扩展根节点。"""
        mcts = MCTS(FakePolicyValueNet(), num_simulations=8)
        board = self._make_black_wins_board()

        # 监视 evaluate_leaf 调用次数
        call_count = {"n": 0}
        original_eval = mcts.evaluate_leaf

        def counting_eval(b):
            call_count["n"] += 1
            return original_eval(b)

        mcts.evaluate_leaf = counting_eval  # type: ignore[assignment]

        mcts.run(board)
        assert call_count["n"] == 0, "终局根节点不应触发模型推理"
        # 根也不应被扩展
        assert mcts._last_root is not None
        assert not mcts._last_root.is_expanded
        assert mcts._last_root.children == {}


# ===========================================================================
# 四、工程：导入与维度
# ===========================================================================
def test_no_circular_imports_smoke():
    # 重新导入一遍主要模块，确保不会触发循环导入
    import importlib

    for name in (
        "game.board",
        "game.coordinates",
        "game.rules_basic",
        "game.encoder",
        "game.notation",
        "model.policy_value_net",
        "mcts.node",
        "mcts.mcts",
    ):
        importlib.import_module(name)


def test_action_index_round_trip_consistent_with_encoder():
    # 与 encoder 的约定保持一致(防止后续误改)
    assert action_to_index(0, 0) == 0
    assert action_to_index(7, 7) == 112
    assert action_to_index(14, 14) == 224
    assert index_to_action(112) == (7, 7)
