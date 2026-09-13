# 《璀璨宝石：对决》AI 训练与自博弈实战指南 (Training Guide)

本文档提供从开发环境搭建、启发式数据蒸馏（模仿学习）、AlphaZero 自博弈飞轮、到模型质检与 Web 实时辅助对弈看板的完整实操教程。

---

## 目录

1. [架构总览与演进路径](#1-架构总览与演进路径)
2. [环境准备与扩展编译](#2-环境准备与扩展编译)
3. [第一阶段：启发式数据生成与模仿学习 (Bootstrap)](#3-第一阶段启发式数据生成与模仿学习-bootstrap)
4. [第二阶段：AlphaZero 自博弈强化学习 (Self-Play Loop)](#4-第二阶段alphazero-自博弈强化学习-self-play-loop)
5. [第三阶段：检查点诊断与竞技场全景评估 (Profiling & Benchmarking)](#5-第三阶段检查点诊断与竞技场全景评估-profiling--benchmarking)
6. [实战应用：启动推理微服务与 Web 辅助看板](#6-实战应用启动推理微服务与-web-辅助看板)
7. [常见问题与调优技巧 (FAQ)](#7-常见问题与调优技巧-faq)

---

## 1. 架构总览与演进路径

AI 系统的整体训练流程遵循标准的 AlphaZero 进化范式，并结合底层 Rust 引擎进行高性能加速：

```
[阶段 1: 模仿学习 Bootstrap]
  Rust 8 线程并行模拟 HeuristicAI (50万步/秒) 
      ⬇ 分片流式落盘 (Sharded Compact Batch)
  Policy-Value 神经网络初训 (获得合法动作直觉与基础大局观) ➡ 产出初始 best.pt
      ⬇
[阶段 2: AlphaZero 自博弈飞轮]
  由当前 Champion 模型驱动 MCTS 深度推演进行自我对战 (产生自博弈样本)
      ⬇
  训练候选网络 (Candidate Net)
      ⬇
  竞技场门禁对抗 (Candidate vs Baseline 严格成对换座测试)
      ├── 胜率 >= 55% ──> 晋升为主力！保存 iter_xxx.pt 并刷新 best.pt
      └── 胜率 <  55% ──> 淘汰丢弃，回滚权重，继续探索
      ⬇
[阶段 3: 部署与对弈]
  加载 best.pt 启动本地 HTTP 推理服务 ➡ 驱动 Web 看板实时推荐动作与预测胜率
```

---

## 2. 环境准备与扩展编译

本项目核心引擎采用 Rust 编写，上层模型和自博弈由 Python (PyTorch) 驱动，通过 `maturin` 进行 PyO3 高性能绑定。

### 2.1 激活虚拟环境
根据项目规范，必须使用根目录下的 `.venv` 虚拟环境：

- **Git Bash (推荐)**：
  ```bash
  source .venv/Scripts/activate
  ```
- **Windows PowerShell**：
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

### 2.2 编译 Rust 扩展模块
当修改了 `engine/` 目录中的 Rust 规则、特征编码或 MCTS 逻辑后，需要重新编译 Python 动态库扩展：

```bash
maturin develop --release --features python
```

验证编译成功：
```bash
python -c "import splendor_ai._engine as _engine; print('Rust Engine 加载成功:', dir(_engine))"
```

---

## 3. 第一阶段：启发式数据生成与模仿学习 (Bootstrap)

在强化学习完全从零（Cold Start）开始探索时，由于桌游规则空间庞大，随机行动很难触达终局胜利。因此首先利用内置的规则智能体（`HeuristicAI`）进行全速并行模拟，让网络迅速掌握卡牌价值与基础下棋逻辑。

### 3.1 数据生成机制说明（Rust 引擎底层全速并发）

> **核心解答**：您**不需要**手动去执行一个单独的 Rust 二进制工具来生成数据。
>
> 第一条启动命令本身就是**“Rust 高速并发采样 + 分片流式落盘 + 神经网络训练”一体化**的：
> 1. `train.py` 在未指定 `--reuse-data` 时，会自动通过 PyO3 调用 Rust 底层的 `sample_heuristic_games_parallel`（源码见 `engine/src/ai/sampling.rs`）。
> 2. Rust 使用 `rayon` 线程池驱动 CPU 全部 8 核心全速并发模拟，吞吐量高达 **50 万 ~ 60 万步/秒**（生成 10,000 局对弈、约 170 万步样本仅需约 3 秒）。
> 3. 生成的样本按分片流式存入 `data/shards/shard_xxx.npz`，内存始终受控在 1GB 以内。

### 3.2 两种常见工作流

#### 模式 A：一键端到端运行（自动调用 Rust 并行生成数据 ➡ 紧接着开启训练）
首次训练或需要全新采样时直接执行：
```bash
# 采样 10,000 局启发式对局，每 2500 局落盘为一个分片，随后自动训练 5 个 Epochs
python python/train.py --mode imitation \
    --games 10000 \
    --shard-games 2500 \
    --epochs 5 \
    --batch-size 4096 \
    --lr 1e-3
```

#### 模式 B：两阶段解耦（仅用 Rust 预生成海量数据池 ➡ 多次调参复用训练）
如果希望先离线预生成一个 5 万 ~ 10 万局的超大样本池，之后尝试不同网络参数或学习率反复实验：

1. **第一步：仅调用 Rust 生成海量数据分片（设置 `--epochs 0` 跳过训练）**：
   ```bash
   python python/train.py --mode imitation --games 50000 --shard-games 5000 --epochs 0
   ```
   *控制台会显示 Rust 8 线程并发将数据写入 `data/shards/`，生成落盘完成后自动安全退出。*

2. **第二步：纯训练模式（添加 `--reuse-data`，跳过数据生成，直接复用磁盘分片）**：
   ```bash
   # 直接复用已有分片，尝试不同的学习率和轮次训练
   python python/train.py --mode imitation --reuse-data --epochs 10 --lr 5e-4 --batch-size 4096
   ```

### 3.3 常用控制参数说明
- `--games`：生成对局总数（建议 10,000 ~ 50,000 局）。
- `--shard-games`：单个分片包含的局数（默认 2500，平衡 IO 与内存）。
- `--data-dir`：分片落盘目录（默认 `data/shards`）。
- `--reuse-data`：**复用磁盘上已有分片**，跳过 Rust 数据生成步骤直接开训。
- `--clear-data`：清理 `data/shards` 历史分片，强制重新生成全新样本。
- `--resume <ckpt_path>`：从指定权重断点续训。

### 3.4 关键指标观察
训练过程中控制台输出：
```text
Epoch   Train Loss    Policy Loss   Top-1 Acc     Top-3 Acc     Val Loss    
1       2.1450        1.8320        52.3%         78.6%         2.0812      🌟 (Best)
2       1.7821        1.5120        61.8%         84.2%         1.7450      🌟 (Best)
```
- **Top-1 Acc**：网络直接预测动作与专家动作一致的比例（通常达到 55%~65% 即为优秀）。
- **Top-3 Acc**：网络前 3 概率动作覆盖专家动作的比例（达到 80%~90% 说明动作掩码与大局观正常）。
- 产出产物：最优模型自动保存至 `checkpoints/best.pt`。

---

## 4. 第二阶段：AlphaZero 自博弈强化学习 (Self-Play Loop)

当模型具备基础走子能力（且在 MCTS 指导下已能压制规则 AI）后，切换到 `selfplay` 模式，开启策略自进化飞轮。

### 4.1 启动自博弈闭环 (Rust 8 线程全速原生驱动)
```bash
# 启动 10 轮 Rust 8 线程全速自博弈迭代，每轮自弈 100 局，MCTS 推演 30 次，开启开局探索与经验池
python python/train.py --mode selfplay \
    --iterations 10 \
    --games-per-iter 100 \
    --mcts-sims 30 \
    --selfplay-backend rust \
    --temp-steps 12 \
    --dirichlet-eps 0.25 \
    --buffer-size 50000 \
    --train-epochs 3 \
    --promote-threshold 0.55
```

### 4.2 核心机制运作流程
每轮迭代自动执行以下四步闭环：
1. **Rust 原生多线程高速采样 (Rust 8-Thread MCTS Sampling)**：
   - 彻底摆脱 Python 单步调度与 GIL 瓶颈，底层 Rust 调用 `rayon` 线程池全核并发推演，**100 局 30 次推演仅需约 2 秒**（吞吐量达 7,500 ~ 10,000 步/秒）。
   - **Rust 原生探索机制注入（破除开局盲区）**：在 Rust MCTS 根节点直接注入狄利克雷噪声（$\alpha=0.3, \epsilon=0.25$），前 `--temp-steps 12` 步在 Rust 内部直接以温度 $\tau=1.0$ 进行轮盘赌采样，打破“开局必锁三金”的模式坍塌；12 步之后退火至确定性推演。
2. **经验回放池滑动窗口管理 (ReplayBuffer)**：
   - 每轮采集的新样本存入 `ReplayBuffer`（默认容量 50,000 步）。
   - 训练样本由最新经验与最近几轮高质量历史对局混合组成，防止策略震荡与灾难性遗忘。
3. **候选模型拟合更新 (Candidate Fitting)**：
   - 候选网络在整个 ReplayBuffer 混合池上进行 Policy-Value 联合损失拟合更新。
4. **严格换座门禁对抗 (Arena Promotion)**：
   - 候选模型与现役冠军模型进行双向成对严格换座对抗（消除先后手发牌随机偏差）。
   - 若胜率 $\ge 55\%$：**晋升为主力**，自动归档 `iter_xxx.pt` 并同步覆盖更新 `best.pt`。
   - 若胜率 $< 55\%$：**晋升失败**，丢弃本次权重，候选网络回滚至 Baseline 状态重新下一轮探索。

### 4.3 进阶调优参数说明
- `--selfplay-backend`：自博弈引擎，`rust`（**推荐**，Rust 8 线程原生 MCTS，速度极快）或 `neural`（Python Neural-MCTS）。
- `--temp-steps`：开局探索步数（默认 12 步），此阶段采用 Softmax 概率轮盘赌，打破固定套路。
- `--dirichlet-alpha` 与 `--dirichlet-eps`：根节点狄利克雷探索噪声参数（默认 0.3 和 0.25）。
- `--buffer-size`：经验回放池最大样本容量（默认 50,000 步）。
- `--mcts-sims`：每步 MCTS 推演次数（推演越深样本质量越高，建议 30 ~ 80）。
- `--eval-agent`：门禁测试智能体类型（`policy_net` 快速评估，`neural_mcts` 深度推演评测）。
- `--eval-pairs`：成对门禁评测局数对（默认 5 对 = 10 局）。

---

## 5. 第三阶段：检查点诊断与竞技场全景评估 (Profiling & Benchmarking)

为了准确掌握某个权重文件的真实水准与策略风格，项目提供了专用的诊断剖析工具 `profile_ckpt.py`。

### 5.1 运行权重诊断剖析
```bash
# 默认诊断 checkpoints/best.pt 进行 10 局全景基准测试
python python/profile_ckpt.py --games 10

# 诊断指定检查点并加大对局量
python python/profile_ckpt.py checkpoints/iter_050.pt --games 20

# 开启 MCTS 深度推演进行极限棋力评估
python python/profile_ckpt.py checkpoints/best.pt --games 10 --mcts --sims 50
```

### 5.2 诊断报告核心板块解读
运行后终端将输出全方位的诊断档案：
1. **血统与训练量**：
   ```text
   • 自博弈迭代轮次: Iteration 122
   • 累计训练 Epochs: 78
   • 累计自博弈经历: 约 12,200 局对弈 (~1,952,000 真实决策步)
   ```
2. **基准 1：自我镜像对抗 (Self vs Mirror)**：
   - 先后手胜率平衡度（理想状态下 P0 与 P1 胜率在 45%~55% 之间）。
   - 平均终局步数（通常在 140~180 步左右）。
3. **基准 2：对战启发式 AI (Model vs HeuristicAI)**：
   - 真实对抗胜率（若达到 70%~90%+ 说明已彻底超越基础规则）。
   - 胜因分布（**20声望胜 / 10皇冠胜 / 10单色胜** 的分布，可反映模型风格是否单一化）。

### 5.3 独立竞技场多模式对战 (`evaluate.py`)
如果想要对比任意两个不同模型、或让模型对战随机基线：
```bash
# 让最佳模型 (PolicyNet) 对抗 Rust MCTS (50次推演)
python python/evaluate.py --agent1 net --model1 checkpoints/best.pt --agent2 rust_mcts --sims 50 --pairs 10
```

---

## 6. 实战应用：启动推理微服务与 Web 辅助看板

训练出的最优模型可无缝接入实时 Web 对局看板，为实机下棋提供实时推荐与局势推演。

在另一个终端中启动 Rust Web 服务：
```bash
cargo run --bin replay_web
```
控制台将输出启动信息：
```text
🚀 Splendor Duel Web Server running on http://127.0.0.1:8080
```

在浏览器中打开 [http://127.0.0.1:8080](http://127.0.0.1:8080)：
- 可以在界面上进行人机对战（Human vs AI）或双 AI 观战。
- 点击右上角的 **“神经网络状态”** 查看当前加载的 Epoch、模型修改时间及连接状态。
- 在当前对局回合，系统会实时渲染出 **Top 候选动作推荐（附带执行理由与概率分布）** 及双方的实时胜率曲线。

---

## 7. 常见问题与调优技巧 (FAQ)

### Q1: 发现开局 AI 总是无脑锁 3 张牌拿 3 个黄金，从不拿宝石怎么办？
- **根因**：初始的 `HeuristicAI` 规则对预留 3 级高分牌赋予了极高权重（开局评分 ~105 分，远高于拿宝石的 60 分）。若自博弈探索不足，网络会继承该强硬偏见并误以为这是全局最优解（纳什局部陷阱）。
- **解法**：
  1. **开局温度采样**：在采样逻辑中，前 10~15 步设置温度 $\tau \ge 1.0$ 进行轮盘赌采样，打破开局确定性。
  2. **根节点注入 Dirichlet 噪声**：强行给低先验动作（如拿取连线普通宝石）分配搜索探索机会。
  3. **提升 MCTS 推演深度**：将 `mcts-sims` 提升至 80~150 次。更深的搜索能让 AI 察觉到“盲目锁牌导致前期节奏落后”的深层代价。

### Q2: 出现 CUDA Out Of Memory (OOM) 如何调整？
- `train.py` 默认开启了 AMP（自动混合精度），显存占用很低（通常 < 2GB）。
- 若显存极其紧张，可适当降低 Batch Size：`--batch-size 2048`。
- 模仿学习时调整分片大小：`--shard-games 1000`。

### Q3: 自博弈多轮迭代后胜率停滞，候选模型很难晋升（< 55%）？
- **探索衰减过快**：检查是否在过早的轮次就完全使用了贪婪策略（Argmax）。
- **学习率过大**：进入后期迭代时，降低学习率（例如 `--lr 2e-4` 或 `--lr 1e-4`），避免剧烈震荡洗掉已学得的残局知识。
- **引入历史对手池**：自博弈不仅与当前最新的自己打，还要定期抽选历史检查点（如 `iter_020.pt`）作为对手，防止陷入自我循环克制的单一盲区。
