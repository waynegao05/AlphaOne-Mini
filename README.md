# AlphaOne-Mini 五子棋智能体

AlphaOne-Mini 是一个面向 15×15 五子棋的本地智能体工程。项目包含棋盘与坐标系统、基础规则、黑方禁手原型、比赛开局协议、策略价值网络、MCTS、自博弈训练、战术搜索、本地 Tkinter 对弈界面和外部 AI 适配器。

## 核心能力

- 15×15 棋盘，坐标范围 `A1` 到 `O15`。
- 动作索引约定：`action_index = y * 15 + x`。
- `basic` 模式：五连或以上直接判胜。
- `forbidden` 模式：黑方长连、四四、三三禁手工程实现；黑方精确五连优先。
- 指定开局、三手交换、五手 N 打协议状态机。
- CNN、SmallResNet、AdvancedPolicyValueNet 三档策略价值网络。
- MCTS、自博弈、监督预训练、战术蒸馏、数据增强和评估脚本。
- AlphaOne-Mini 集成棋手：规则强制战术、VCF、VCT、对手前瞻、MCTS 和战术兜底。
- Tkinter 本地对弈界面：支持 Human、AlphaOne-Mini 和外部 `AI.py`。
- 外部 AI 适配器：支持同目录依赖模块，并兼容常见决策接口。
- 棋谱解析与导出：支持简单序列格式和 C5 文本格式。

## 能力边界

- 公开仓库不包含训练得到的 `.pt` 权重。缺少权重时，AlphaOne-Mini 会退化到不依赖权重的战术搜索路径。
- `forbidden` 模式仍是工程实现，复杂正式比赛边界需要继续验证。
- 开局协议已实现为独立状态机，但训练流水线默认不会强制使用完整比赛协议。
- 本项目不能宣称达到正式比赛级棋力。棋力结论必须以固定规则、固定算力和足够局数的对局评估为准。

## 环境要求

- Python 3.10 或更高版本
- `numpy`
- `PyYAML`
- `torch`
- `pytest`

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

Tkinter 通常随 Python 一起安装。正式深度学习训练建议使用支持 CUDA 的 PyTorch；CPU 可用于功能验证。

## 快速启动

启动 Tkinter 本地对弈界面：

```powershell
python main_tkinter_play.py
```

界面中可为黑白双方分别选择：

- `Human`
- `AlphaOne-Mini`
- `External AI.py`

如需加载外部 AI，可在界面中选择 `AI.py` 文件，或使用环境变量配置默认路径：

```powershell
$env:ALPHAONE_EXTERNAL_AI_PATH="D:\path\to\AI.py"
python main_tkinter_play.py
```

启动命令行人机对弈：

```powershell
python main_play.py --ai-player strong --rule-mode basic --human-color black --device cpu
```

## 测试

运行完整测试：

```powershell
python -m pytest -q
```

运行核心规则、战术搜索和界面测试：

```powershell
python -m pytest tests/test_basic_rules.py tests/test_rules_forbidden_overline.py tests/test_rules_forbidden_double_four.py tests/test_rules_forbidden_double_three.py tests/test_vcf_search.py tests/test_vct_search.py tests/test_strong_player.py tests/test_tkinter_play_smoke.py -q
```

## 训练与评估

生成少量自博弈数据：

```powershell
python main_selfplay.py --num_games 1 --num_simulations 5
```

运行轻量训练：

```powershell
python main_train.py --data outputs/selfplay_data/selfplay_latest.npz --epochs 1 --batch-size 8 --device cpu
```

运行轻量评估：

```powershell
python main_evaluate.py --opponent random --games 2 --num-simulations 5 --device cpu
```

运行完整轻量流水线：

```powershell
python main_pipeline.py --selfplay-games 1 --num-simulations 5 --train-epochs 1 --evaluate-games 2 --device cpu
```

## 目录结构

```text
engine/      战术搜索、启发式、VCF、VCT、对手前瞻和外部 AI 适配
evaluate/    Arena、指标和模型对比
game/        棋盘、坐标、编码、规则和开局协议
mcts/        MCTS 节点和搜索
model/       CNN、ResNet、AdvancedPolicyValueNet 和 checkpoint 工具
pipeline/    轻量训练流水线
records/     棋谱解析、导出和文件读写
selfplay/    自博弈、回放缓冲区和数据增强
train/       训练、蒸馏、辅助标签和实验管理
ui/          CLI 与 Tkinter 界面
utils/       设备管理
configs/     可复现训练配置
tests/       自动化测试
```

## 模型权重

本仓库不分发训练权重。若要启用神经网络增强路径，请自行训练或将本地权重放入 `outputs/checkpoints/`。

## 开源参考说明

开发过程中参考过公开五子棋工程的训练流程、高性能搜索和数据增强思路，但本仓库未复制外部项目的大段代码。公开仓库不包含本地参考目录。

## 项目名称

统一名称：`AlphaOne-Mini`。
