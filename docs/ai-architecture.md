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

## 3. 策略-价值神经网络架构 (`SplendorNet`)

`SplendorNet` 接收当前行动方视角的 **726 维**规范化观测向量，前向输出 **288 维**动作对数概率和 **1 维**标量对局价值。

### 3.1 网络拓扑图

```
                输入观测向量 (726 维)
               ┌──────────┴──────────┐
               │                     │
        [0..200] 棋盘分块       [200..726] 标量上下文分块
         (200 维 -> 8×5×5)              (526 维)
               │                     │
         Conv2d (8->64)         Linear (526->256)
               │                     │
          BatchNorm             LayerNorm
               │                     │
             ReLU                  ReLU
               │                     │
      2× ResidualBlock2D        Linear (256->256)
         (64 通道)                   │
               │                LayerNorm
        AvgPool2d (3×3)              │
         (输出 2×2 网格)              ReLU
               │                     │
         Flatten (256 维)       Context Embedding (256 维)
               │                     │
               └──────────┬──────────┘
                          │ Concat (512 维)
                          ▼
                  Fusion Trunk (MLP)
                  Linear (512->256)
                      LayerNorm
                        ReLU
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
        Policy Head               Value Head
     Linear (256->256)         Linear (256->64)
           ReLU                      ReLU
     Linear (256->288)          Linear (64->1)
             │                         │
             ▼                         ▼
   Masked Logits [B, 288]       Tanh Value [B, 1]
 (Softmax 得到动作先验 P)       (对局胜率预期 [-1.0, 1.0])
```

### 3.2 分支实现细节

1. **棋盘 2D 空间卷积主干 (`board_conv_in` & `board_res_blocks`)**：
   - 5×5 棋盘每格包含 8 种状态独热（空、5 色宝石、珍珠、黄金）。重排为 `(B, 8, 5, 5)` 张量。
   - 输入卷积将通道数提升至 64，后接 2 个带跳跃连接的 `ResidualBlock2D`，充分提取棋盘在水平、垂直与双对角线上的 3 连宝石几何空间特征。
   - 经由 3×3 核 (stride=2) 空间池化压缩为 2×2，展平为 256 维特征向量。
2. **标量上下文 MLP 主干 (`context_mlp`)**：
   - 输入包含卡牌市场金字塔（15 槽位 × 27 维）、场上王室卡（4 槽位 × 5 维）、双方玩家详细资产与胜负指标（2 × 42 维）以及全局环境进度（17 维），共计 526 维。
   - 通过两层 `Linear(256) + LayerNorm + ReLU` 深度提炼当前经济实力差距与斩杀线威胁。
3. **主干融合与双头输出 (`fusion`, `policy_head`, `value_head`)**：
   - 将棋盘特征 (256) 与上下文特征 (256) 拼接为 512 维，经由 LayerNorm 融合层降维至 256 维。
   - **Policy Head**：输出 288 维原始 Logits。在推理时结合当前合法动作布尔掩码 `action_mask`，将非法动作置为 $-10^4$，经 Softmax 归一化为合法动作概率分布。
   - **Value Head**：经由 64 维隐藏层映射并通过 `Tanh` 激活函数，输出当前行动方预期胜率（$[-1.0, 1.0]$）。

### 3.3 ONNX 极速动态导出 (`export_onnx_bytes`)
网络内置 `export_onnx_bytes` 方法，利用 `torch.onnx.export` 将当前 PyTorch 模型直接序列化为内存中的 ONNX 二进制流（Opset 17），开启常量折叠与动态 batch 轴。该字节流可直接无缝传递给 Rust 的 `tract-onnx` 引擎，实现零磁盘 I/O 的跨语言模型传递。

---

## 4. 自博弈与多阶段样本生成流水线

```
[Phase 1: 启发式专家冷启动]
   Rust sample_heuristic_games_parallel (8 线程并发)
   └── 吞吐 > 50 万步/秒 ──► 快速生成 5~10 万步专家对局数据，拟合初始策略网络

[Phase 2: MCTS 深度推演样本混合]
   Rust sample_mcts_games_parallel_with_config
   └── 融入树搜索与 Dirichlet 探索 ──► 产出更高质量策略分布

[Phase 3: 纯神经网络 AlphaZero 自博弈闭环]
   Rust sample_neural_mcts_games_parallel (结合 Tract ONNX)
   └── 完全脱离规则偏见，网络指导 MCTS 自博弈 ──► 产生超越人类理解的博弈对局
```

### 4.1 样本紧凑表示 (`CompactBatch`)
放弃零散 Python 对象，所有数据以 4 个连续 NumPy 数组存储：
- `obs`: `[N, 726]` float32
- `mask`: `[N, 288]` bool
- `action`: `[N]` int64 (标量动作 ID)
- `value`: `[N, 1]` float32 (终局归属视角值)

单个分片支持 `.npz` 直接持久化，并利用 `FastTensorLoader` 直接一次性 `.to(device)` 驻留 GPU 显存，训练迭代时切片开销降至极限。

### 4.2 异步流水线推演 (`Pipelined Self-Play`)
在 `train.py` 的迭代循环中，自博弈数据生成与模型拟合采用流水线重叠执行：
- **Iter $i$**：GPU 在训练 `ReplayBuffer` 中的已有样本。
- **与此同时**：后台 `ThreadPoolExecutor` 异步启动当前 Baseline 模型的下一批次自博弈对局采样。
- 当 GPU 训练完毕时，下一迭代的数据已在内存中就绪，极大压缩了训练等待时间。

---

## 5. 训练器与优化系统 (`Trainer`)

### 5.1 联合损失函数
对局样本的目标动作为 MCTS 访问频次或专家选择动作 $a$，终局胜负为 $z \in \{-1.0, 1.0\}$：

$$\mathcal{L} = \mathcal{L}_{policy} + c_{value} \cdot \mathcal{L}_{value}$$

- $\mathcal{L}_{policy} = -\sum \log \pi(a | s)$（对掩码后的合法动作计算交叉熵）
- $\mathcal{L}_{value} = \frac{1}{B} \sum (v - z)^2$（均方误差）
- 默认 $c_{value} = 1.0$。

### 5.2 训练特性
1. **自动混合精度 (AMP)**：针对 Tensor Core 开启 `torch.autocast("cuda")` 与 `GradScaler`，吞吐提升 2~3 倍并节省显存。
2. **梯度裁剪 (Gradient Clipping)**：设置 `max_norm=5.0`，防止深层对抗探索中的梯度爆炸。
3. **余弦退火调度 (CosineAnnealingLR)**：学习率自 $10^{-3}$ 平滑退火至 $10^{-5}$。
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
