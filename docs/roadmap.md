# 《璀璨宝石：对决》(Splendor Duel) 研发路线图 (Roadmap)

本项目致力于为双人桌游《璀璨宝石：对决》构建一套超越人类顶尖水平的 AlphaZero 风格策略分析、自对弈强化学习与实时对局辅助系统。

---

## 整体实施阶段状态 (以当前实现为准)

```
[阶段 0: 已完成 ✅] 规则引擎与回放底座
  - 官方规则书与 BGA 规则严格对齐 (docs/splendor-duel-rules.md)
  - 纯 Rust 高性能游戏规则引擎 (6.7万局/秒, 2300万步/秒)
  - 本地单页对局回放与时间轴视窗 (engine/src/bin/replay_web.rs & engine/web/)
  - 726 维状态观察张量 + 288 维离散动作掩码设计 (docs/action-encoding.md)

[阶段 1: 已完成 ✅] Rust ↔ Python 桥接与 Gym 环境
  - PyO3 导出 _engine 模块与 NumPy 零拷贝转换 (engine/src/bridge/pymod.rs)
  - Python 侧 Gymnasium 风格单局环境 (SplendorDuelEnv)
  - pytest 端到端状态转移、环境复位与张量验证 (python/tests/)

[阶段 2: 已完成 ✅] Policy-Value 神经网络与启发式基准
  - 2D ResNet (5x5 棋盘) + 全局上下文 MLP 混合骨干 (SplendorNet)
  - Policy Head (288 logits) + Value Head ([-1, 1] 胜率预测)
  - 规则启发式专家 AI (HeuristicAI) 作为冷启动样本生成器与对比基准
  - 动态 ONNX 字节流导出 (export_onnx_bytes) 与 tract-onnx 纯 Rust 极速推理

[阶段 3: 已完成 ✅] 高性能 MCTS 搜索与自博弈流水线
  - PUCT 树搜索算法与根节点 Dirichlet 探索噪声注入 (Rust 原生多线程 RustMCTS)
  - 温度轮盘赌采样破除开局盲区
  - 异步流水线推演机制：GPU 梯度更新与 CPU 自博弈并发重叠
  - 向量化显存常驻批加载器 (FastTensorLoader) 与紧凑样本分片 (CompactBatch)

[阶段 4: 已完成 ✅] 强化学习训练闭环与实时 AI 决策看板
  - Policy-Value 联合损失优化与 CosineAnnealing 调度 (Trainer)
  - 严格成对换座门禁对抗评估系统 (Arena & evaluate_neural_match)
  - 本地 Web 原生服务器 (replay_web.rs) 升级为集对局回放、人机实时对战、全卡牌图鉴于一体的交互分析看板
  - Python 实时推理微服务 (server.py) 支持权重文件变动热重载

[阶段 5: 规划中 🚀] 博弈论深化与实时对局助手增强
  - 信息集蒙特卡洛 (ISMCTS) 或 Belief State 采样，消除 MCTS 对暗手牌的完全信息偏差
  - 持续对弈 Elo 评级池与对局树图剪枝分析
  - 浏览器插件或屏幕识别实时战局辅助推荐
```
