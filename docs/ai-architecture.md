# Python AI 模块与网络架构设计文档 (AI Architecture)

本文档系统阐述《璀璨宝石：对决》(Splendor Duel) Python AI 系统的网络架构、特征融合拓扑、自博弈数据生成流水线、分布式分片训练、门禁对抗评估机制及微服务部署设计。以当前仓库中的真实实现为准。

---

## 1. 架构定位与设计哲学

Python AI 模块基于 AlphaZero 范式构建，负责策略与价值联合深度学习。其核心职责包括：

1. **混合多模态骨干网络 (Hybrid Multimodal Backbone)**：
   - 将二维棋盘空间几何关联（3 连相邻）与高维非空间上下文（卡牌市场、手牌、胜负条件进度）解耦为独立特征分支，再行深度融合。
2. **高速向量化张量通道 (Vectorized Tensor Pipeline)**：
   - 摆脱传统逐样本 Python 字典循环，通过 `CompactBatch` 扁平连续数组与 `FastTensorLoader` 显存常驻批加载，消除总线传输开销。
3. **软硬件重叠异步推演 (Pipelined Asynchronous Self-Play)**：
   - 在 GPU 进行当前轮次神经网络前向反向拟合的同时，后台线程并发调度底层 Rust 8 核执行下一轮次的自博弈推演，实现 GPU 计算与 CPU 自博弈的无缝重叠（Zero Idle Time）。
4. **严格成对换座门禁评测 (Paired Seed Gating & Elo)**：
   - 候选模型晋升采用固定种子成对换座对抗，胜率严格达标（如 $\ge 55\%$）方可取代旧基准，有效避免策略退化或震荡。

---

## 2. 模块拓扑与目录组织

```
python/
├── train.py                  # 闭环自博弈与迭代训练主调度入口
├── evaluate.py               # 独立对抗评估与 Elo 基准测试入口
├── profile_ckpt.py           # 模型检查点权重剖析与拓扑参数检查工具
├── splendor_ai/              # AI 核心包
│   ├── __init__.py           # 包导出与 C 扩展桥接
│   ├── env.py                # Gymnasium 规范强化学习环境 (SplendorDuelEnv)
│   ├── net.py                # 混合拓扑 Policy-Value 神经网络 (SplendorNet)
│   ├── mcts.py               # 智能体接口与策略直觉/MCTS 适配层
│   ├── dataset.py            # 紧凑样本块 (CompactBatch) 与显存常驻加载器 (FastTensorLoader)
│   ├── trainer.py            # 混合精度 AMP 训练器与原子性检查点维护 (Trainer)
│   ├── arena.py              # 成对换座竞技场对抗系统 (Arena)
│   ├── selfplay.py           # 高性能自博弈与启发式样本采样流水线
│   ├── server.py             # 带文件变动感知的轻量级 HTTP 推理微服务
│   └── progress.py           # 终端节流进度条工具
└── tests/                    # 单元测试与端到端集成测试
    ├── test_engine.py        # 引擎环境协议与状态转移测试
    ├── test_net.py           # 神经网络输入输出形状与梯度测试
    ├── test_trainer.py       # 训练步与损失收敛测试
    ├── test_mcts_arena.py    # MCTS 与 Arena 对战测试
    └── test_neural_mcts.py   # Rust 神经网络 MCTS、样本缓存与对战测试
```

---

## 3. 策略-价值神经网络架构 (`SplendorNet v3`)

`SplendorNet` 接收当前行动方视角的 **969 维**规范化观测向量，前向输出 **1856 维**跨模态解耦动作对数概率、**1 维**胜负期望（基于 21 桶 Two-Hot 分位数期望内积）、**1 维**预期剩余轮数以及 **3 维**终局胜因预测，并在训练态提供 2 维博弈态势差辅助预测。

### 3.1 网络拓扑图

```
                             输入观测向量 (969 维)
               ┌───────────────────────┼───────────────────────┐
               │                       │                       │
        [0..225] 棋盘分块       [225..657] 市场卡牌分块   [657..969] 标量上下文分块
        (225 维 -> 9×5×5)       (12 市场 + 6 双方手牌)    (4王室+48基础+44全局=96维)
               │               (共 18 实体 × 36 维特征)         │
         Conv2d (9->64)                │                Linear (96->128)
               │              CardEncoder (36->128)            │
           GroupNorm                   │                LayerNorm + ReLU
               │           SetSelfAttention (4 heads)          │
             ReLU                      ├───────────────┐Linear (128->128)
               │                       ▼               │       │
      2× ResidualBlock2D         Mean/Max Pool         │LayerNorm + ReLU
        (GroupNorm 稳定)               │               │       │
               │               Linear (256->128)       │       │
      Conv2d 1x1 (64->16)              │               │       │
        (保持 5×5 网格)         Card Context (128 维)  │Scalar Context (128 维)
               │                       │               │       │
         Flatten (400 维)              │               │       │
               │                       │               │       │
       Linear (400->128)               │               │       │
               └───────────────────────┼───────────────┘       │
                                       │ Concat (384 维)       │
                                       ▼                       ▼
                               Fusion Trunk (Linear(384->256) + LayerNorm + ReLU)
                                       │ [B, 256] (fused)
               ┌───────────────────────┴───────────────────────────────────────┐
               ▼                                                               ▼
  Cross-Entity Policy Heads                                        Multi-Task Valuation
 ┌───────────────────────────┐                           ┌───────────┼───────────┼───────────┐
 │ Token Head (172 维)       │                           ▼           ▼           ▼           ▼
 │ 连线与特权打分            │                        Win Head   Turns Head  Reason Head Lead Head
 ├───────────────────────────┤                        (21-bin     (Sigmoid    (3-way BCE  (2-way Tanh
 │ Reserve Head (375 维)     │                        Two-Hot)    [0, 1])     Logits)     [-1, 1])
 │ 25 黄金网格 + 15 目标卡   │                           │
 ├───────────────────────────┤                           ▼
 │ Buy Head (1260 维)        │                      Expectation
 │ 15 卡意愿 + 84 方案偏好   │                      Dot-Product
 │ + 32 维低秩双线性交互     │                           │
 ├───────────────────────────┤                           ▼
 │ Ability Head (49 维)      │                      Scalar Win
 │ 变色/连击/偷取/王室/弃牌  │                      ([-1, 1])
 └─────────────┬─────────────┘
               ▼
   Assembled Logits [B, 1856]
```

### 3.2 分支实现细节

1. **棋盘 2D 空间卷积主干 (`board_conv_in` & `board_res_blocks`)**：
   - 5×5 棋盘每格包含 8 种标记状态独热 + 第 9 通道顺时针螺旋排位拓扑特征。重排为 `(B, 9, 5, 5)` 张量。
   - 输入卷积将通道数提升至 64，后接 2 个带跳跃连接的 `ResidualBlock2D`。全部空间卷积均采用 `GroupNorm(4, channels)`，彻底消除小批次推理与训练时的 BatchNorm 统计量抖动。
   - 经由 1×1 卷积 (`Conv2d(64->16) + GroupNorm + ReLU`) 保持 5×5 空间拓扑分辨率（用于黄金网格特征），展平后通过 `Linear(400->128) + LayerNorm + ReLU` 输出 128 维全局棋盘表征。
2. **共享卡牌实体与自注意力网络 (`card_encoder` & `card_attention`)**：
   - 覆盖 12 张市场明牌、3 张我方预留手牌与 3 张敌方预留卡（POMDP 暗牌掩蔽），共计 18 个实体。每张卡牌采用 **36 维**规范特征（包含基础费用/产出/点数/皇冠/技能，以及 6 维动态净缺口、黄金冲抵后真实缺口与对手可购威胁）。
   - 共享 `CardEncoder` 将 36 维卡牌投射至 128 维，通过 4 头 `SetSelfAttention` 捕捉卡牌间互相提供加成与抢位争夺关系。
   - 经由 Mean+Max 双路池化生成 128 维全局卡牌态势表征。
3. **标量上下文 MLP 主干 (`context_mlp`)**：
   - 输入包含场上王室卡（4 维）、我方基础资产（24 维）、敌方基础资产（24 维）以及全局环境、博弈差值与胜负威胁（44 维），共计 96 维有效输入。
   - 通过两层 `Linear(128) + LayerNorm + ReLU` 深度提炼当前经济差距与胜负线威胁。
4. **实体解耦跨模态策略打分器 (Cross-Entity Policy Heads)**：
   - `token_head`: 172 维，负责特权使用、单个标记与 2~3 直线连线标记获取。
   - `reserve_head`: 375 维，解耦为 25 黄金网格空间局部特征打分 + 15 预留目标特征打分（12 市场明牌来自对应卡牌 Token，3 盲抽牌堆顶来自全局融合表征）。
   - `buy_head`: 1260 维，解耦为 15 槽位卡牌购买意愿（卡牌 Token 独立打分）+ 84 种支付方案经济偏好（全局融合表征）+ 32 维卡牌因子与支付方案静态嵌入矩阵的双线性低秩交互。
   - `ability_head`: 49 维，处理变色卡、同色取盘、偷对手标记、选王室卡与超限弃牌等后续能力动作。
   - 最终按标准动作空间物理索引拼装回 1856 维完整动作分布。
5. **分位数胜率与多任务评估 (Multi-Task Valuation)**：
   - **Win Head**：采用 21 桶均匀支撑点 $[-1.0, 1.0]$ 的 Two-Hot 分位数分布输出，使用软分类交叉熵训练，杜绝传统 Tanh 饱和区梯度消失与 MSE 均方误差震荡；在模型推理与 ONNX 导出时内部自动与支撑点内积计算标量数学期望 $[-1.0, 1.0]$，对外保持完全一致的标量胜率契约。
   - **Turns Head**：输出归一化剩余轮数预期（$[0.0, 1.0]$，Sigmoid 激活，Smooth L1 损失）。
   - **Reason Head**：输出 3 维独立多标签终局胜因 Logits（$[20\_pts, 10\_crowns, 10\_color]$，BCE 损失）。
   - **Lead Head**：博弈态势差辅助头，输出声望分差与皇冠差估计（Tanh 激活，Smooth L1 损失），强化融合主干对胜负关键差值的表征敏锐度。

### 3.3 ONNX 极速动态导出 (`export_onnx_bytes`)
网络内置 `export_onnx_bytes` 方法，利用 `torch.onnx.export` 将当前 PyTorch 模型直接序列化为内存中的 ONNX 二进制流（Opset 17），开启常量折叠与动态 batch 轴。Two-Hot 支撑点常量折叠后内积生成标准 `win_value` 标量，该字节流可直接无缝传递给 Rust 的 `tract-onnx` 引擎，实现零磁盘 I/O 的跨语言模型传递。

---

## 4. 自博弈与多阶段样本生成流水线

```
[Phase 1: 启发式专家冷启动]
   Rust sample_heuristic_games_parallel (8 线程并发)
   └── 吞吐 > 50 万步/秒 ──► 快速生成 5~10 万步专家对局数据，拟合初始策略网络

[Phase 2: 纯神经网络 AlphaZero 自博弈与联盟对决闭环]
   Rust sample_neural_mcts_games_parallel (结合 Tract ONNX)
   └── 完全脱离规则偏见，纯神经指导 MCTS 自博弈与对抗 ──► 产出高质量探索策略与价值对局
```

### 4.1 样本紧凑表示 (`CompactBatch`)
放弃零散 Python 对象，所有数据以连续 NumPy 数组存储：
- `obs`: `[N, 969]` float32
- `mask`: `[N, 1856]` bool
- `action`: `[N]` int64 (标量动作 ID)
- `value`: `[N, 2]` float32 (纯胜负与归一化剩余轮数)
- `reason`: `[N, 3]` float32 (终局胜因多标签)

单个分片支持 `.npz` 直接持久化，并利用 `FastTensorLoader` 直接一次性 `.to(device)` 驻留 GPU 显存，训练迭代时切片开销降至极限。

### 4.2 异步流水线推演 (`Pipelined Self-Play`)
在 `train.py` 的迭代循环中，自博弈数据生成与模型拟合采用流水线重叠执行：
- **Iter $i$**：GPU 在训练 `ReplayBuffer` 中的已有样本。
- **与此同时**：后台 `ThreadPoolExecutor` 异步启动当前 Baseline 模型的下一批次自博弈对局采样。
- 当 GPU 训练完毕时，下一迭代的数据已在内存中就绪，极大压缩了训练等待时间。

---

## 5. 训练器与优化系统 (`Trainer`)

### 5.1 固化多任务联合损失函数
对局样本的目标动作为 MCTS 访问频次或专家选择动作 $a$，终局胜负目标 $z \in \{-1.0, 1.0\}$ 经 `SplendorNet.to_two_hot` 映射为 21 桶软标签分布 $\mathbf{q}_{two\_hot}$：

$$\mathcal{L}_{total} = w_p \mathcal{L}_{policy} + w_w \mathcal{L}_{win} + w_t \mathcal{L}_{turns} + w_r \mathcal{L}_{reason} + w_l \mathcal{L}_{lead}$$

- $\mathcal{L}_{policy} = -\sum \pi_{target}(a) \log \pi_{pred}(a)$（软分布交叉熵或掩码交叉熵）
- $\mathcal{L}_{win} = -\sum_{k=1}^{21} q_{two\_hot}^{(k)} \log p_{win}^{(k)}$（Two-Hot 软分类交叉熵）
- $\mathcal{L}_{turns} = \text{SmoothL1}(v_{turns}, z_{turns})$
- $\mathcal{L}_{reason} = \text{BCEWithLogits}(logits_{reason}, targets_{reason})$
- $\mathcal{L}_{lead} = \text{SmoothL1}(pred_{lead}, \text{extract\_state\_leads}(obs))$
- 权重固定加权：$w_p = 1.0, w_w = 1.0, w_t = 0.30, w_r = 0.15, w_l = 0.20$，确保价值头与辅助任务拥有稳定的强梯度流。

### 5.2 训练特性
1. **自动混合精度 (AMP)**：针对 Tensor Core 开启 `torch.autocast("cuda")` 与 `GradScaler`，吞吐提升 2~3 倍并节省显存。
2. **梯度裁剪 (Gradient Clipping)**：设置 `max_norm=5.0`，防止深层对抗探索中的梯度爆炸。
3. **余弦退火调度与受挫自适应重置**：学习率自 $10^{-3}$ 退火至 $10^{-5}$。当候选模型连续多轮未达门禁遭淘汰时，通过 `trainer.reset_learning_rate()` 重建调度周期，恢复探索活力。
4. **原子检查点保存**：先写入临时文件，再通过 `os.replace` 原子替换，防止写入中断导致权重损坏。

---

## 6. 竞技场对抗体系与门禁准则 (`Arena`)

### 6.1 成对种子与严格双向换座 (Paired-Seed Swapping)
为彻底剥离《璀璨宝石：对决》中初始牌堆洗牌运气与先手优势对模型水平评判的干扰，竞技场采用成对换座机制：
- 对于测试种子 $S$，固定发牌序列与暗牌堆顺序：
  - **局 1**：候选模型作为先手 (P0)，基准模型作为后手 (P1)。
  - **局 2**：基准模型作为先手 (P0)，候选模型作为后手 (P1)。
- 评估指标包含：总胜率、先手胜率 (P0 Seat Win Rate)、平均获胜步数、3 种胜利原因（20 分、10 冠、10 单色分）分布。

### 6.2 门禁淘汰与晋升闭环
- **晋升条件**：候选模型在多局成对换座中，胜率达到预设阈值（例如 $\ge 55\%$）。
- **处理逻辑**：
  - 达标：晋升为主力模型，保存为 `best.pt`，经验池滚动接收新自博弈对局。
  - 未达标：淘汰候选模型权重，回退到当前基准模型权重重新采样探索，防止策略自相残杀和漂移。

---

## 7. 实时推理微服务 (`server.py`)

为了支持本地 Web 端（如 `play.html` 人机对战视窗或外部分析工具）的实时决策推荐，系统包含一个微秒级 HTTP 推理服务：

- **极简高性能**：基于 Python 标准库 `http.server`，无大型框架依赖。
- **权重热重载 (Hot-Reloading Watcher)**：后台线程每隔 1.5 秒轮询检查 `best.pt` 的修改时间与文件体积（带写入防抖），在训练产生更强模型时自动无缝热更新内存网络。
- **接口端点**：
  - `POST /predict`：接收观测向量与掩码，返回推荐动作、Top-5 候选概率与对局预期胜率。
  - `GET /health`：查看当前托管模型的 Epoch、体积与设备状态。
  - `POST /reload`：手动强制重载权重。
