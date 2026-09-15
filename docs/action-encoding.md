# 《璀璨宝石：对决》(Splendor Duel) 特征与动作编码规格

本文档定义《璀璨宝石：对决》在强化学习（AlphaZero / PPO / MCTS）环境下的状态观察张量（Observation Tensor）与动作空间（Action Space）编码标准。

---

## 一、核心原则

1. **绝对视角规范化 (Canonical Perspective)**：
   - 输入张量永远以**当前行动玩家（Active Player = "我方", 索引 0）**为第一视角，**对手（Opponent = "敌方", 索引 1）**为第二视角。
   - 神经网络无须学习“座位轮换对称性”。
2. **严格消除虚假度量距离**：
   - 离散类别特征（标记类型、卡牌技能、阶段）**严格采用 One-Hot 编码**，杜绝标量小数映射引入的度量偏见。
3. **空间几何拓扑保留**：
   - 5×5 棋盘支持 2D 空间特征图 `(8, 5, 5)` 与展平 200 维两种形式，原生适配卷积神经网络（ResNet）与全连接（MLP）。
4. **物理界限归一化**：
   - 所有连续值均使用游戏规则内的物理最大值进行归一化（如分数/20、皇冠/10、手牌/10、特权/3），确保所有特征严格落在 $[0.0, 1.0]$ 区间内。
5. **胜负条件完全覆盖**：
   - 涵盖总声望分（/20）、皇冠数（/10）、5 种基础颜色各自声望进度（/10）及最大单色声望（/10）。

---

## 二、状态特征张量布局 (Total: 879 Float32)

全局状态由 5 大特征分块组成，展平总维度为 **879 维**：

```
[0..200]    分块 1: 5×5 棋盘空间 (8 通道 × 25 格 = 200 维)
[200..605]  分块 2: 金字塔市场卡牌 (15 槽位 × 27 维 = 405 维)
[605..625]  分块 3: 场上王室卡 (4 槽位 × 5 维 = 20 维)
[625..835]  分块 4: 双方玩家状态仪表板 (2 玩家 × 105 维 = 210 维)
[835..879]  分块 5: 全局环境、博弈差值与胜负威胁 (44 维)
```

### 1. 棋盘空间张量 (200 维)
5×5 网格（行优先 $r \in [0, 4], c \in [0, 4]$），每格 8 通道 One-Hot：
- 通道 0：`Empty`（空格）
- 通道 1：`White`（钻石）
- 通道 2：`Blue`（蓝宝石）
- 通道 3：`Green`（绿宝石）
- 通道 4：`Red`（红宝石）
- 通道 5：`Black`（黑曜石）
- 通道 6：`Pearl`（珍珠）
- 通道 7：`Gold`（黄金）

### 2. 金字塔卡牌张量 (15 槽位 × 27 维 = 405 维)
- Level 3：3 个明牌槽位 + 1 个牌堆余量指示槽 (4 槽位)
- Level 2：4 个明牌槽位 + 1 个牌堆余量指示槽 (5 槽位)
- Level 1：5 个明牌槽位 + 1 个牌堆余量指示槽 (6 槽位)
每个卡牌槽位 27 维：
- `[0]`: `present` (1.0 或 0.0)
- `[1..3]`: `tier` (3-way One-Hot: L1, L2, L3)
- `[4]`: `points / 6.0`
- `[5]`: `crowns / 3.0`
- `[6..11]`: `cost: [w, b, g, r, k] / 8.0`, `pearl / 2.0`
- `[12..17]`: `bonus_color` (6-way One-Hot: W, B, G, R, K, None/Joker)
- `[18]`: `bonus_value / 2.0`
- `[19..25]`: `ability` (7-way One-Hot: None, ExtraTurn, TakePrivilege, TakeSameColor, StealToken, ColorCopy, ColorCopyAndExtraTurn)
- `[26]`: `can_afford` (1.0 或 0.0，当前行动方是否立即买得起)

### 3. 王室卡张量 (4 槽位 × 5 维 = 20 维)
- `[0]`: `available` (1.0 或 0.0)
- `[1]`: `points / 3.0`
- `[2..4]`: `ability` (3-way One-Hot: StealToken, TakePrivilege, ExtraTurn)

### 4. 双方玩家仪表板 (2 玩家 × 105 维 = 210 维)
按 `[我方 (Active), 敌方 (Opponent)]` 排列，每位玩家 105 维：
- `[0..7]`: 标记库存 `[w, b, g, r, k, pearl, gold, total] / 10.0`
- `[8..12]`: 永久 Bonus `[w, b, g, r, k] / 6.0`
- `[13..20]`: 胜负条件进度：
  - 总分 `/ 20.0`
  - 皇冠 `/ 10.0`
  - 5 色声望 `[w, b, g, r, k] / 10.0`
  - 最大单色声望 `/ 10.0`
- `[21..23]`: 特权与王室指标：
  - 特权卷轴数 `/ 3.0`
  - 已获王室卡数 `/ 2.0`
  - 3 冠里程碑是否已达成 (bool)
- `[24..104]`: 预留手牌 (3 槽位 × 27 维 = 81 维)：
  - 我方全部预留手牌与对手公开明牌预留：采用与金字塔一致的完整 27 维卡牌槽位特征规范（含点数、皇冠、费用、Bonus 颜色 One-Hot、技能 One-Hot 与 `can_afford` 支付能力判定）。
  - 敌方盲抽暗牌 (M4 POMDP 设计)：保留 `present = 1.0` 与盲抽操作公开可见的 `tier` (3-way One-Hot: L1, L2, L3)；其余所有私密属性（点数、皇冠、费用、颜色、技能及支付能力）严格清零掩蔽为 `0.0`，杜绝信息泄露。

### 5. 全局环境、博弈差值与胜负威胁 (44 维)
- `[0..8]`: `TurnPhase` (9-way One-Hot: OptionalActions, MandatoryAction, CardAbilityJoker, CardAbilitySameColor, CardAbilitySteal, SelectRoyalCard, DiscardTokens, SelectReserveGold, GameOver)
- `[9]`: `privilege_pool / 3.0`
- `[10]`: `bag_total_count / 25.0`
- `[11..17]`: **布袋 7 色标记各自具体剩余数量 (7 维)**：
  - `[11]`: White `/ 4.0`
  - `[12]`: Blue `/ 4.0`
  - `[13]`: Green `/ 4.0`
  - `[14]`: Red `/ 4.0`
  - `[15]`: Black `/ 4.0`
  - `[16]`: Pearl `/ 2.0`
  - `[17]`: Gold `/ 3.0`
- `[18]`: `board_token_count / 25.0`
- `[19]`: 全局回合数归一化 `min(1.0, turn_number / 80.0)`
- `[20]`: `extra_turn_granted` (1.0 或 0.0)
- `[21..23]`: 牌堆剩余比例 `[deck1/30, deck2/24, deck3/13]`
- `[24]`: `points_diff_norm` (`(cp.points - op.points + 20.0) / 40.0`, 领先为 >0.5, 落后为 <0.5)
- `[25]`: `crowns_diff_norm` (`(cp.crowns - op.crowns + 10.0) / 20.0`)
- `[26]`: `color_diff_norm` (`(cp.max_color - op.max_color + 10.0) / 20.0`)
- `[27]`: `privilege_diff_norm` (`(cp.privileges - op.privileges + 3.0) / 6.0`)
- `[28..29]`: 双方手牌余量 `(10 - total).max(0) / 10.0`
- `[30..31]`: **双方手牌超限数量** `(total - 10).max(0) / 5.0`
- `[32..37]`: 双方胜利距离 (Gap to Win: 分数/20, 皇冠/10, 单色/10)
- `[38..39]`: 双方离胜利的最短归一化差距 `min(points_gap, crowns_gap, color_gap)`
- `[40]`: `cp_has_winning_action` (1.0 或 0.0，我方当前是否有一步致胜买卡斩杀动作)
- `[41]`: `op_has_winning_purchase` (1.0 或 0.0，对手当前在金字塔与公开预留手牌中是否买得起致胜卡；严格遵循 POMDP 掩蔽暗抽牌)
- `[42]`: **本回合是否已补盘 `replenished_this_turn`** (1.0 或 0.0)
- `[43]`: **本回合已消耗特权数** `privileges_used_this_turn / 3.0`

---

## 三、动作空间 (Action Space) 离散编码 (288 维)

```
ID 范围         动作语义
----------------------------------------------------------------------
[0]             SkipOptional (跳过可选行动)
[1..=25]        UsePrivilege (棋盘 25 个坐标)
[26]            ReplenishBoard (补充棋盘)
[27..=51]       TakeTokens: 单个标记 (25 个坐标)
[52..=171]      TakeTokens: 直线相邻 2~3 连线 (120 种几何直线组合)
[172..=183]     ReserveCard: 金字塔明牌 (12 个槽位)
[184..=186]     ReserveCard: 牌堆顶盲抽 (3 个等级)
[187..=198]     PurchaseCard: 金字塔明牌 (12 个槽位)
[199..=201]     PurchaseCard: 自己预留卡 (3 个槽位)
[202..=206]     AssignJokerColor: 变色卡附着颜色 (5 种基础颜色)
[207..=231]     TakeSameColorToken: 盘上取同色 (25 个坐标)
[232..=238]     StealToken: 偷对手标记 (5 种宝石 + 珍珠 + 黄金)
[239..=242]     SelectRoyal: 选择王室卡 (4 个槽位)
[243..=249]     DiscardToken: 超限弃牌 (7 类标记)
[250..=274]     TakeGoldToken: 预留卡牌连锁选择拿取黄金 (棋盘 25 个坐标)
[275..=287]     预留对齐空间 (13 维)
----------------------------------------------------------------------
总动作空间大小 ACTION_SIZE = 288
```
