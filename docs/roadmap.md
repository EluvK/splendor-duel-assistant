# 《璀璨宝石：对决》(Splendor Duel) 研发路线图 (Roadmap)

本项目致力于为双人桌游《璀璨宝石：对决》构建一套超越人类顶尖水平的 AlphaZero 风格策略分析、自对弈强化学习与实时对局辅助系统。

---

## 整体实施阶段

```
[阶段 0: 已完成] 
  - 官方规则书与 BGA 规则对齐 (docs/splendor-duel-rules.md)
  - 纯 Rust 高性能游戏规则引擎 (6.7万局/秒, 2300万步/秒)
  - 本地单页对局回放与时间轴视窗 (engine/src/bin/replay_web.rs & engine/web/)
  - 725 维状态观察张量 + 256 维离散动作掩码设计 (docs/action-encoding.md)

[阶段 1: 正在进行] Rust ↔ Python 桥接与 Gym 环境
  - PyO3 导出 _engine 模块与 NumPy 零拷贝转换
  - Python 侧 Gymnasium 风格单局环境 (SplendorDuelEnv)
  - pytest 端到端状态转移与张量验证

[阶段 2: 待推进] Policy-Value 神经网络架构与启发式基准
  - 2D ResNet (5x5 棋盘) + 全局上下文 MLP 混合骨干
  - Policy Head (256 logits) + Value Head ([-1, 1] 胜率)
  - 规则启发式 AI (Heuristic AI) 作为基准对比物

[阶段 3: 待推进] 高性能 MCTS 搜索与自博弈流水线
  - PUCT 树搜索算法与 Dirichlet 噪声注入
  - 批量推理队列加速 (Batched Inference)
  - 自动化自对弈样本收集池 (Replay Buffer)

[阶段 4: 待推进] 强化学习训练闭环与实时 AI 决策看板
  - Policy-Value 联合损失优化与调度
  - 门禁对抗评估系统 (Elo 竞技场机制)
  - 本地 Web 回放视窗升级为实时对弈推荐分析看板
```
