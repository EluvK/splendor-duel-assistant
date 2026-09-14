# AGENTS.md

本项目为桌游璀璨宝石：对决 (Splendor Duel) 构建的 AI 策略分析与实时推荐辅助系统。

## 仓库结构

- `engine` 目录为 Rust 游戏引擎，负责模拟游戏规则、状态转移、合法动作生成与蒙特卡洛搜索等。
- `python` 目录为 Python AI 模块，负责神经网络定义 (Policy-Value 网络)、训练、自博弈循环与评估。
- 两者通过 PyO3 桥接，实现 Rust ↔ Python 的高效交互与训练数据传递。

`reference` 目录为参考项目（仅当明确指令时才可访问，且只读，勿修改）。

`docs` 目录下的 `*.md` 为本项目有效文档，需要优先参考这些文档并维护其内容的准确性。

- `docs/splendor-duel-rules.md`：游戏规则文档
- `docs/engine-architecture.md`：Rust 游戏引擎架构设计文档
- `docs/ai-architecture.md`：Python AI 模块与网络架构文档
- `docs/action-encoding.md`：状态与动作特征编码说明
- `docs/web-architecture.md`：Web 可视化与对战交互架构说明

## Git 提交规范

- 格式：`<prefix1>[,<prefix2>...]: <description>`（例：`engine,docs: xxxxxxxxx`）
- 允许前缀：`engine`, `docs`, `python`, `web`, `bugfix`, `perf`, `refactor`, `reference`, `chore`

## Agent 工作准则

1. 工作语言：中文沟通，代码与文档中英文均可，参考已有内容保持一致。
2. 项目环境：使用项目根目录下的 `.venv` 虚拟环境，任何 python / cargo 命令均需在激活该虚拟环境（`source .venv/Scripts/activate`）后执行。不要在系统全局环境、其他目录下创建或修改文件。
