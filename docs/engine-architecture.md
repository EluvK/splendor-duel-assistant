# Rust 游戏引擎架构设计文档 (Engine Architecture)

本文档系统阐述《璀璨宝石：对决》(Splendor Duel) Rust 原生游戏引擎的设计目标、模块划分、核心数据结构、规则状态机、高性能推演及 Python 跨语言桥接协议。以当前仓库中的真实实现为准。

---

## 1. 架构总览与设计目标

Rust 引擎承担了整套系统的规则核心、状态转移、合法动作生成、蒙特卡洛树搜索（MCTS）及高并发自博弈样本采样，其设计遵循以下原则：

1. **零堆分配与紧凑内存 (Zero-Cost & Compact Memory)**：
   - 基础类型全面采用 `Copy`/`Clone` 语义与定长栈分配（如 `TokenCollection` 使用 `[u8; 7]`，棋盘使用 `[[Option<GemType>; 5]; 5]`）。
   - 状态复制（State Clone）极速，单步模拟开销控制在微秒级，为 MCTS 前向深度展开提供算力保障。
2. **确定性与随机可重现 (Reproducibility)**：
   - 引擎所有随机事件（初始洗牌、布袋抽取等）均通过强随机数发生器（ChaCha8Rng）及明式 64 位种子完全确定，实现对局的 100% 确定性回放。
3. **细粒度原子状态机 (Fine-grained State Machine)**：
   - 将复杂的多步交互（特权使用、连线拿取、卡牌能力连锁、王室挑选、超限弃牌、预留拿黄金）分解为显式离散阶段（`TurnPhase`），杜绝隐式状态。
4. **嵌入式推理与多线程并发 (Embedded ONNX & Rayon)**：
   - 集成 `tract-onnx` 纯 Rust 推理引擎，在无 Python/CUDA 运行时开销下直接驱动神经网络 MCTS。
   - 借助 `rayon` 并发框架，在释放 GIL 状态下全核并行进行自博弈采样与成对门禁对决。

---

## 2. 模块拓扑与目录组织

```
engine/
├── Cargo.toml               # 引擎依赖配置 (pyo3, tract-onnx, rayon, serde, rand 等)
├── src/
│   ├── lib.rs               # 根库模块、公共 API 导出与 PyO3 #[pymodule] 入口
│   ├── main.rs              # 引擎测试/交互命令行入口
│   ├── model/               # 基础游戏实体与模型定义
│   │   ├── mod.rs
│   │   ├── token.rs         # 宝石标记类型 (GemType) 与集合计数器 (TokenCollection)
│   │   ├── card.rs          # 珠宝卡/王室卡结构、等级、成本与技能枚举
│   │   ├── data.rs          # 静态卡牌全集 (67张珠宝卡 + 4张王室卡常数数据)
│   │   └── action.rs        # 离散游戏动作枚举 (Action)
│   ├── game_state/          # 游戏状态持久与容器结构
│   │   ├── mod.rs
│   │   ├── phase.rs         # 回合状态机阶段 (TurnPhase) 与终局判定 (VictoryReason)
│   │   ├── board.rs         # 5x5 棋盘网格、螺旋轨道 (SPIRAL_ORDER) 与连线检测
│   │   ├── player.rs        # 玩家手牌、标记、特权卷轴与属性缓存 (PlayerState)
│   │   └── state.rs         # 完整对局状态根结构 (GameState) 与初始化洗牌
│   ├── gameplay/            # 核心规则引擎与状态推进
│   │   ├── mod.rs
│   │   ├── payment.rs       # 购卡折抵、珍珠支付与黄金万能抵扣算法
│   │   ├── scoring.rs       # 3 大胜利条件瞬时判定 (check_victory)
│   │   ├── rules.rs         # 合法动作生成器 (RuleEngine::legal_actions)
│   │   └── engine.rs        # 状态转移步进器 (GameEngine::step)
│   ├── ai/                  # 智能体策略与树搜索实现
│   │   ├── mod.rs
│   │   ├── heuristic_ai.rs  # 规则启发式专家 AI (带斩杀、点数与技能加权打分)
│   │   ├── mcts.rs          # 基于先验剪枝与 AlphaZero 探索的高性能 RustMCTS
│   │   ├── neural_evaluator.rs # tract-onnx 纯 Rust 嵌入式神经网络评估器
│   │   ├── neural_ai.rs     # 基于 HTTP 接口的外部推理智能体客户端
│   │   ├── sampling.rs      # Rayon 8 线程并行自博弈采样与严格换座对抗评估
│   │   ├── replay.rs        # 对局轨迹 DTO 序列化与回放会话 (ReplaySession)
│   │   └── interactive.rs   # 面向 Web 交互与人机对战会话 (InteractiveSession)
│   ├── bridge/              # 跨语言接口与特征张量化
│   │   ├── mod.rs
│   │   ├── encode.rs        # 726 维规范化状态观测与 288 维动作空间编解码
│   │   └── pymod.rs         # PyO3 PyGameState 封装与多线程 NumPy 零拷贝转换
│   └── bin/
│       ├── bench.rs         # 引擎微基准与吞吐量压测
│       └── replay_web.rs    # 原生嵌入式 HTTP 服务与 Web 对局回放/对战视窗
└── web/                     # 前端单页应用 (Vanilla JS + HTML5 + CSS)
    ├── replay.html          # 时间轴回放视窗
    ├── play.html            # 交互式人机/AI对战看板
    ├── inspect_cards.html   # 全卡牌资产图鉴查看器
    ├── js/                  # 业务逻辑与控制器
    └── css/                 # 样式表
```

---

## 3. 核心数据结构与游戏状态抽象

### 3.1 标记与集合 (`GemType` & `TokenCollection`)
- `GemType` 包含 7 种离散宝石类型：
  - 基础五色：`White`, `Blue`, `Green`, `Red`, `Black`
  - 特殊标记：`Pearl`（珍珠，稀缺且无永久卡牌 bonus）、`Gold`（黄金，万能折抵，仅能通过预留卡牌获得）
- `TokenCollection` 使用内部 `counts: [u8; 7]` 数组按枚举索引存储，实现了 `add`, `remove`, `remove_collection`, `can_afford` 等基本代数操作，提供极高运算效率。

### 3.2 5x5 棋盘与几何线段检测 (`Board`)
- 棋盘由 `grid: [[Option<GemType>; 5]; 5]` 表示。
- **螺旋轨道填充**：定义官方 `SPIRAL_ORDER: [(usize, usize); 25]` 静态坐标序列，在棋盘重填时顺时针由内向外从布袋摸取填满空位。
- **连线检测**：`find_all_lines` 方法在 4 个方向向量（右、下、右下、左下）进行滑动检测，生成全部包含 1 至 3 枚非黄金标记的连续线段候选（`LineCandidate`）。

### 3.3 玩家状态与缓存 (`PlayerState`)
- 维护玩家资产：标记库存（`tokens`）、已打出卡牌（`cards`）、预留卡（`reserved_cards`，上限 3 张）、王室卡（`royal_cards`，上限 2 张）、特权卷轴数（`privileges`，上限 3 个）。
- **缓存计算属性**：实时同步缓存 `bonuses: [u8; 5]`（基础颜色折扣）、`color_points: [u8; 5]`（各色累计分）、`points_card_points`（纯分卡分）、`total_points` 及 `total_crowns`，避免重复遍历卡牌列表。

### 3.4 根状态机 (`GameState`)
- 维护全局组件：`board`, `bag`, `decks`（3 级未摸出牌堆）, `pyramid`（3 级金字塔明牌展示区）, `royal_cards`（场上 4 张王室卡）, `privilege_pool`（公共特权池）, `players: [PlayerState; 2]`, `current_player`, `phase`, `turn_number`, `winner` 等。

---

## 4. 回合细粒度状态机与规则推进

### 4.1 阶段转换闭环 (`TurnPhase`)
引擎通过 9 种精确的离散阶段对单回合内的微观决策流转进行解耦：

```
[回合开始]
   │
   ▼
TurnPhase::OptionalActions  ◄───────────────────┐ (先用特权：若仍有特权且未补盘可再次使用)
   │                                             │
   ├─► Action::UsePrivilege ─────────────────────┘ (补盘后严禁使用特权)
   ├─► Action::ReplenishBoard ──────────┐ (后补棋盘：对手得 1 特权，可选行动直接结束)
   └─► Action::SkipOptional             │
         │                              │
         ▼                              ▼
TurnPhase::MandatoryAction  ◄───────────┘
   │
   ├─► Action::TakeTokens (拿 1~3 连线非黄金标记) ──────────┐
   ├─► Action::ReserveCard (需盘上有黄金，预留明牌或暗摸) ──────► TurnPhase::SelectReserveGold
   └─► Action::PurchaseCard (打出金字塔明牌或自己预留牌) ─────┐                           │
         │                                               │                      Action::TakeGoldToken
         ├─► 卡牌为 Joker ──► TurnPhase::CardAbilityJoker  │                           │
         │                         │                     │                           ▼
         │                    附着基础颜色                │                  after_action_check
         │                         │                     │                           ▲
         ▼                         ▼                     │                           │
     触发卡牌能力连锁 ─────────────────────────────────────┼───────────────────────────┤
         ├─► ExtraTurn: 赋予额外回合标记                   │                           │
         ├─► TakePrivilege: 获得特权卷轴                   │                           │
         ├─► TakeSameColor ──► TurnPhase::CardAbilitySameColor ─► Action::TakeSameColorToken
         ├─► StealToken ─────► TurnPhase::CardAbilitySteal ────► Action::StealToken ───┘
         └─► 无能力 / 完成能力
                                   │
                                   ▼
                       after_ability_check 结算
                                   │
                     ┌─────────────┴─────────────┐
                     ▼                           ▼
        王冠达标 (3 或 6 顶)                 标记总数 > 10 枚
     TurnPhase::SelectRoyalCard        TurnPhase::DiscardTokens
             │                                   │
     Action::SelectRoyal                Action::DiscardToken (循环弃至 10 枚)
             │                                   │
             └─────────────┬─────────────────────┘
                           │
                           ▼
                      finish_turn
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
       达成胜利条件               未达成胜利条件
TurnPhase::GameOver(Reason)  处理 ExtraTurn 或切换到对手
                               TurnPhase::OptionalActions
```

### 4.2 胜利条件判定 (`check_victory`)
每次动作结算后立即执行常数级胜利检测：
1. **声望总分胜利**：`player.total_points >= 20`。
2. **王冠总数胜利**：`player.total_crowns >= 10`。
3. **单色统治胜利**：存在某种基础颜色使得 `player.color_points[gem] >= 10`。

---

## 5. 智能体与搜索体系

### 5.1 规则启发式专家 (`HeuristicAI`)
- 作为训练冷启动专家与基准对抗基石，无需任何训练即可提供有竞争力的对策。
- **打分加权体系**：
  - 斩杀动作：若直接达成胜利，赋予 `+10000.0` 极大优先级。
  - 卡牌购买：基础价值分（声望 × 30 + 王冠 × 25 + 折扣 × 15）+ 技能加权（额外回合 +40、偷标记 +20、特权 +20）- 黄金消耗惩罚。
  - 标记获取：根据当前金字塔明牌的所需缺口动态打分，对能填补卡牌成本的宝石提高权重，对导致对手获赠特权的惩罚动作进行降权。
  - 随机扰动：在最终得分上注入 `[-0.5, 0.5]` 弱噪声，打破平局死循环。

### 5.2 高性能蒙特卡洛树搜索 (`RustMCTS`)
- **PUCT 树搜索选择**：
  $$U(s, a) = c_{puct} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$
- **AlphaZero 探索机制**：
  - 根节点支持 Dirichlet 噪声注入（$\alpha=0.3, \epsilon=0.25$），强行向次优分支分配搜索预算，打破开局盲区。
  - 支持前 10~15 步温度轮盘赌采样（$T=1.0$），后程切为贪婪决策（$T=0.0$）。
- **零开销快捷路径**：当合法动作仅有 1 个时，跳过搜索直接返回，节约海量无效推演算力。

### 5.3 嵌入式纯 Rust 神经网络评估器 (`TractNeuralEvaluator`)
- 基于 `tract-onnx` 库，直接解析并优化从 PyTorch 导出的 ONNX 二进制模型字节流。
- 构建紧凑运行图（SimplePlan），在纯 Rust 环境下执行前向推理，单步耗时亚毫秒级，脱离 Python 解释器与 CUDA 显存依赖。

### 5.4 并行自博弈与严格换座对抗 (`sampling.rs`)
- 基于 `rayon` 实现线程级并行数据生成：
  - `sample_heuristic_games_parallel`: 8 线程并行生成启发式专家自对弈轨迹，吞吐超 50 万步/秒。
  - `sample_neural_mcts_games_parallel`: 结合 tract-onnx 进行纯神经引导 MCTS 并发采样。
  - `sample_neural_mcts_match_games_parallel`: 结合模型与对手进行多轮对抗采样。
  - `evaluate_neural_match_parallel`: 严格成对换座双向对战评测（种子 S 下分别以先手和后手进行对抗），彻底消除荷官发牌运气方差，在 2~3 秒内完成 20 局成对门禁对抗。

---

## 6. PyO3 桥接层与零拷贝协议

通过 `engine/src/bridge/pymod.rs` 暴露 `_engine` 原生 C 扩展模块：

1. **`PyGameState` 类**：
   - `reset(seed)`: 确定性重置。
   - `observe()`: 输出当前行动方规范视角的 726 维 `Vec<f32>`。
   - `action_mask()`: 输出当前合法动作的 288 维 `Vec<bool>`。
   - `step(action_id)`: 执行动作 ID，推进状态机，返回 `(next_obs, done, winner)`。
   - `clone_state()`: 高效状态深拷贝。
   - `heuristic_action_id()`: 直接调用底层 C 语言级别启发式 AI 决策。
2. **多线程采样导出函数**：
   - `generate_heuristic_samples`, `generate_neural_mcts_samples`, `generate_neural_mcts_match_samples`, `evaluate_neural_match`。
   - 全部在 `py.detach(|| ...)` 中执行，**完全释放 Python 全局解释器锁 (GIL)**，在后台全核跑满 CPU；完成后通过 `numpy::PyArray1::from_vec` 零拷贝交还 Python 托管，杜绝序列化损耗。

---

## 7. Web 回放与交互对战服务器 (`replay_web.rs`)

为了实现对局的可视化分析、调试与实时人机对战，引擎内置了自包含的轻量级原生 HTTP 服务：

- **零外部 Web 框架依赖**：基于标准库 `std::net::TcpListener` 与 `TcpStream` 实现。
- **双工作模式**：
  1. **对局回放模式 (`/replay.html`)**：由 `/api/status`, `/api/history`, `/api/jump` 驱动，支持单步前进、后退、时间轴任意跳转及每步决策候选动作概率打分可视化。
  2. **实时对战模式 (`/play.html`)**：由 `/api/game/action`, `/api/game/restart` 驱动，支持 Human、Neural AI、Heuristic AI、Random 之间的任意对弈与实时 AI 决策提示。
  3. **卡牌资产图鉴 (`/inspect_cards.html`)**：支持 67 张珠宝卡与 4 张王室卡的成本、属性、皇冠、能力的交互式审查。
