# Web 可视化与对战交互架构说明 (Web Architecture)

本文档系统阐述《璀璨宝石：对决》(Splendor Duel) Web 可视化界面、实时交互对战系统及复盘分析模块的架构设计、API 规范、前端逻辑与视觉资产渲染体系。

---

## 1. 架构总览与设计目标

Web 模块位于 `engine/web/`，由 Rust 游戏引擎内置的原生轻量 HTTP 服务（`engine/src/bin/replay_web.rs`）驱动，旨在提供无需复杂前端脚手架（No-Bundler/Zero-Build）、开箱即用的高性能交互界面：

1. **零构建轻量化 (No Build Tools Required)**：
   - 采用原生现代 ES Modules、原生 CSS 变量及标准 HTML5 结构，修改文件即时生效，无需 Webpack/Vite 编译打包。
2. **双模式工作流 (Play & Replay)**：
   - **实时交互对战 (`play.html`)**：支持 人类 vs 神经网络、人类 vs 启发式、人类 vs 随机 AI、以及本地双人热座对战。
   - **对局深度复盘 (`replay.html`)**：支持加载 JSON 对局轨迹，逐回合步进、分支探索与价值走势评估。
3. **桌面端高保真视觉体系 (BGA Sprite Sheets & Layout)**：
   - 采用 Board Game Arena (BGA) 级别的原生雪碧图（Sprite Sheet）与切片算法，精准呈现 67 张珠宝卡、4 张王室赞助卡、5×5 螺旋棋盘与各类实体配件。
4. **强状态同步与原子校验 (Authoritative Engine Validation)**：
   - 前端通过 REST API 与 Rust 引擎原子状态机实时同步；一切动作有效性均由 Rust 引擎权威判定，前端仅负责候选高亮指引与合法动作提交。

---

## 2. 目录结构与模块划分

```
engine/web/
├── index.html                 # 默认入口（自动重定向至 play.html）
├── play.html                  # 核心实时人机/双人交互对战页面
├── replay.html                # 核心对局复盘回放与深度分析页面
├── inspect_cards.html         # BGA 雪碧图 (Sprite Sheet) 逐帧排查与校对工具
├── css/
│   ├── style.css              # 主样式表（深色主题、三栏布局、棋盘、卡牌、仪表盘）
│   └── interaction.css        # 交互专用样式（悬停浮层、选取高亮、引导条、动画）
├── js/
│   ├── game-controller.js     # 对战控制器 (GameController)：状态流转、用户输入、API 驱动
│   └── shared-components.js   # 共享渲染库：BGA 雪碧图定位、棋盘绘制、卡牌构建、玩家面板
└── assets/
    └── images/                # BGA 原版贴图素材
        ├── board.jpg          # 5×5 螺旋棋盘底图
        ├── cards1.jpg         # Level 1 珠宝卡雪碧图 (31 帧，含卡背)
        ├── cards2.jpg         # Level 2 珠宝卡雪碧图 (25 帧，含卡背)
        ├── cards3.jpg         # Level 3 珠宝卡雪碧图 (14 帧，含卡背)
        ├── royal-cards.jpg    # 王室赞助卡雪碧图 (4 帧)
        ├── tokens.png         # 宝石与黄金标记图标
        ├── gem-icons.png      # 宝石微标图标
        ├── icons.png          # 动作与皇冠微标
        ├── privilege.png      # 特权卷轴贴图
        ├── bag.png            # 补充布袋贴图
        └── background.jpg     # 全局背景纹理
```

---

## 3. 后端服务与通信协议

静态托管与 API 接口均在 `engine/src/bin/replay_web.rs` 中使用标准库 `std::net::TcpStream` 实现多线程无锁/读写锁 HTTP 服务：

### 3.1 核心对战 API (`/api/game/...`)

| 请求方法 | 路由路径 | 说明 | 输入参数 / Payload |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/game/state` | 获取当前对战完整状态、合法动作及历史步骤摘要 | 无 |
| `POST` | `/api/game/action` | 人类玩家执行合法动作 | JSON 序列化的动作结构体 `{ "action": Action }` |
| `POST` / `GET` | `/api/game/ai_step` | 触发当前席位的 AI 思考并走一步 | 无 |
| `POST` / `GET` | `/api/game/reset` | 重置并开启新对局（返回新状态与重置历史） | Query 参数：`seed` (默认 42), `p0` (身份), `p1` (身份) |
| `GET` | `/api/game/history` | 获取当前对局全部历史步骤摘要列表 | 无 |
| `GET` | `/api/game/step` | 获取指定单步完整数据（含 AI 估值与 Top 候选排行） | Query 参数：`index` (步骤编号，默认最新步) |

#### `GET /api/game/state` 响应结构示例：
```json
{
  "state": {
    "current_player": 0,
    "turn_count": 1,
    "phase": "MandatoryAction",
    "board": [ ... 5x5 二维矩阵 ... ],
    "pyramid": [ ... 3 级金字塔卡牌二维列表 ... ],
    "decks_count": [26, 21, 10],
    "royals": [ ... 待认领王室卡 ... ],
    "privilege_pool": 2,
    "bag_count": 0,
    "players": [ ... 双方玩家声望、王冠、标记、Bonus、预留卡与已购卡 ... ],
    "winner": null
  },
  "player_kinds": ["human", "neural"],
  "current_player": 0,
  "current_player_kind": "Human",
  "is_human": true,
  "legal_actions": [
    {
      "category": "take_tokens",
      "action": { "TakeTokens": { "positions": [[2, 2], [2, 3]], "count": 2 } },
      "desc": "拿取 2 颗标记"
    }
  ],
  "neural_available": true,
  "history_len": 2,
  "history": [
    {
      "index": 0,
      "round": 1,
      "player": 0,
      "action": "Game Started",
      "phase": "OptionalActions",
      "score": null,
      "ai_type": null
    },
    {
      "index": 1,
      "round": 1,
      "player": 0,
      "action": "Skip Optional (Auto)",
      "phase": "OptionalActions",
      "score": null,
      "ai_type": null
    }
  ]
}
```

### 3.2 神经网络与系统状态 API

- `GET /api/neural_status`：检查神经网络微服务（或原生 ONNX）的就绪状态、在线心跳及当前加载模型（如 Epoch 轮次）。
- `POST /api/neural_reload`：热重载最新的 `best.pt` 权重文件，无需重启 Web 或引擎服务。

---

## 4. 前端控制器逻辑 (`js/game-controller.js`)

`GameController` 采用面向对象单例管理整场对局生命周期与异步交互：

### 4.1 核心流转与自动走步机制
1. **轮询与事件驱动**：每次玩家操作成功（`submitAction`）或触发 `stepAi` 后，立即拉取最新状态并重新触发全界面渲染。
2. **AI 自动走步定时器 (`scheduleAiStep`)**：当当前玩家身份非人类（`isHuman === false`）且勾选了【AI 自动走步 ⚡】时，触发延迟微任务调用 `/api/game/ai_step`，保证视觉动画平滑且避免请求风暴。
3. **局势指引条 (`renderGuideBanner`)**：
   - 动态判定当前阶段（可选阶段、连线拿宝石、预留拿黄金、变色绑定、超限弃牌、认领王室等），在版图顶部给出文字指引与快捷动作按钮（如【确认拿取宝石】、【补充棋盘】、【跳过可选阶段】）。

### 4.2 棋盘宝石连线选取算法 (`calculateCandidatePositions`)
玩家在 `MandatoryAction` 点击棋盘宝石时，控制器依据 Rust 引擎传回的合法动作列表 `legal_actions` 进行实时候选约束：
- **首颗选中**：只要存在于任一合法 `TakeTokens` 动作中的非黄金格子，均可作为起点。
- **第二、三颗延伸**：过滤出所有覆盖已选坐标子集的合法 `TakeTokens`，并提取这些动作中未被选中的邻接坐标作为 Candidate，赋予高亮样式；非法格子点击自动忽略。
- **取消选择**：再次点击已选中的宝石格子可反选撤销。

### 4.3 卡牌预留与黄金选取双向自主交互体系
按照《璀璨宝石：对决》规则，预留卡牌强制行动（`ReserveCard`）要求拿取棋盘上 1 枚指定坐标的黄金标记，并预留 1 张明牌或牌堆暗抽。为杜绝系统自动替玩家结算默认黄金、保障策略自主权，前端实现了对称且平滑的双向选取交互：
1. **先选黄金，再选卡牌**：
   - 玩家首先点击棋盘上任意一枚可用黄金，触发选定状态（`selectedGoldPos`），棋盘该黄金高亮聚焦，金字塔明牌与牌堆进入脉冲引导态（`pulse-reservable`）。
   - 随后点击目标明牌【📌 预留卡牌】或牌堆【🎴 盲抽预留】，立即以该选定黄金坐标分发 `ReserveCard` 动作。
2. **先选卡牌，再选黄金（防止自动结算）**：
   - 玩家若未选黄金直接点击金字塔卡牌【📌 预留卡牌】或牌堆【🎴 盲抽预留】，前端绝不自动预支棋盘第 1 颗黄金，而是进入待选黄金流程（`pendingReserveTarget`）。
   - 目标卡牌获得金色高亮微光（`card-pending-reserve`）与角标提示【📌 待选黄金】；
   - 棋盘上所有可用黄金全部进入金色脉冲呼吸态（`cell-gold-candidate`）；顶部指引条提供清晰说明、各黄金坐标快捷按钮（如 `💰 拿取黄金 (1, 2)`）以及【✖ 取消预留】撤销按钮。
   - 玩家在 5×5 棋盘中直接点击任意一颗黄金（或点击顶部对应按钮），即可明确完成该黄金的拿取与预留。
3. **撤销与平滑切换**：
   - 点击【✖ 取消预留】随时退出预留模式；直接点击其他卡牌【💎 购买】或点击棋盘普通宝石，系统自动清除预留待选状态，平滑过渡至购买或连线拿宝石流程，杜绝交互死锁。

### 4.4 模态对话框交互系统 (`showModal`)
针对需要多选一的复杂连锁行动（例如：变色卡指定关联颜色、夺取对手指定颜色标记、超上限弃置手牌等），控制器通过模态遮罩层呈现可视化选项卡，并在用户确认后向引擎分发对应的具体参数动作。

### 4.5 操作历史与 AI 决策可视化体系 (`gameLogContainer` & `stepDetailModal`)
针对人机对战场景中观察 AI 行动轨迹与评估思考的需求，控制器集成了实时的操作历史追踪与下钻分析系统：
1. **实时操作历史面板 (`#gameLogContainer`)**：
   - 位于左栏 5×5 棋盘正下方，实时追加对战双方每一步的轮次（Round）、步数索引（#Index）、行动方（含 `👤P0` / `🧠P1` / `🤖P1` 色标）、以及中文化动作语义（如“拿取 3 颗宝石: 蓝, 红, 白”、“购买 2阶 珠宝卡 #35”等）。
   - **AI 估值标签**：若是 AI 步骤，即时计算并呈现模型预估值（神经网络显示胜率百分比，如 `62%`；启发式 AI 显示评分，如 `+4.5` 或 `斩杀`）。
   - **追踪滚动 (`#btnToggleLogAutoScroll`)**：支持一键开启/关闭自动跟随最新步骤滚动。
2. **AI 决策详情下钻模态框 (`#stepDetailModal`)**：
   - 点击操作历史中的任意步骤，通过 `/api/game/step?index=N` 异步获取该步权威决策上下文。
   - 完整展开决策引擎类型、当前胜率预期或打分、以及当时的候选项列表（Top Candidates）。每项直观展示备选动作、置信概率/评分条，并以金色徽章标明最终采纳的走法。
3. **顶部指引横幅联动**：
   - 当 AI 行动完成轮到人类时，顶部指引条醒目指示 `🤖 AI (P1) 上步: [动作描述] [胜率 62%]`，帮助玩家即刻感知盘面变动。

---

## 5. UI 组件与雪碧图渲染体系 (`js/shared-components.js`)

### 5.1 BGA 原版雪碧图定位算法
全套 67 张卡牌与 4 张王室卡分别分布在单行水平排列的雪碧图文件中：
- **卡背位于 0 号帧**（Face-down card back，`backgroundPosition: 0% 0%`）。
- **正面位于 1..N 号帧**。

```javascript
// 水平雪碧图背景偏移计算公式
const posX = (colIndex / (totalCols - 1)) * 100;
el.style.backgroundImage = `url('${sheetUrl}')`;
el.style.backgroundSize = `${totalCols * 100}% 100%`;
el.style.backgroundPosition = `${posX.toFixed(4)}% 0%`;
```

| 类别 | 资源文件 | 总帧数 (totalCols) | 正面帧映射关系 |
| :--- | :--- | :--- | :--- |
| **Tier 1 (初阶)** | `cards1.jpg` | 31 列 | `BGA_CARD_SPRITE_MAP[0..29]` (1..30 帧) |
| **Tier 2 (中阶)** | `cards2.jpg` | 25 列 | `BGA_CARD_SPRITE_MAP[30..53]` (1..24 帧) |
| **Tier 3 (高阶)** | `cards3.jpg` | 14 列 | `BGA_CARD_SPRITE_MAP[54..66]` (1..13 帧) |
| **王室赞助卡** | `royal-cards.jpg` | 4 列 | `royal.id` (0..3 帧) |

> 备注：卡牌 ID 与 BGA 精灵图物理顺序并非简单线性对应，`shared-components.js` 中内置了经过 `inspect_cards.html` 实机比对校验的精确映射表 `BGA_CARD_SPRITE_MAP`。

### 5.2 核心渲染组件职责
- **`renderBoard(...)`**：生成 5×5 网格单元，放大至 306×306px，根据合法动作动态附加 `cell-clickable`、`cell-selected`、`cell-candidate`、`cell-highlight` 与 `cell-gold-selected`；选中宝石时采用极简优雅的金色聚焦光环与绝对贴合格子右上角的序号角标 (1, 2, 3)，精准清晰且杜绝遮挡原画；支持对手动作幽灵残留发光高亮（Ghost Highlighting）。
- **`createDeckPileElement(...)`**：生成实体牌库卡背，渲染剩余张数角标，当有预留额度时提供【🎴 盲抽预留】快捷悬浮操作与金色闪烁引导态。
- **`renderCardsList(...)`**：渲染金字塔货架上的翻开卡牌；自动挂载 `CardHovercard` 即时高清浮层（包含 1.4x 放大切片、详细技能解释、净花费扣除与差额计算）；判断购买负担能力（`affordable` 绿光高亮），悬停展示【购买】与【预留】按钮；对暗抽预留卡（对手视角）执行背面保密隐藏。
- **`renderRoyalsPool(...)`**：居中展示当前未被认领的王室赞助卡，支持即时 Hovercard 悬停预览。
- **`renderPlayerDashboard(...)`**：渲染单个玩家的声望条 (20分)、王冠条 (10冠)、最高单色分条 (10分)、3D 实体手牌标记 Chips、宝石镶嵌形态的 Bonus 减免徽章、预留卡栏、已获王室卡及已购买卡牌微缩矩阵 (Tableau)；当任意胜利维度达到斩杀临界门槛（≥16声望、≥8王冠、≥8单色）时自动触发赛点危机警示光效。

---

## 6. 样式与视觉体系规范 (`css/style.css` & `css/interaction.css`)

1. **标准三列自适应主战场布局**：
   - **左列 (`.pyramid-section`)**：金字塔珠宝卡展示区（Tier 3、Tier 2、Tier 1 纵向堆叠）及下方王室赞助池。独立成列并享有充裕展示宽度，卡牌规格优化为 84×127px，彻底根绝横向滑动条。
   - **中列 (`.center-stage`, 宽度加宽 20%)**：垂直紧凑上下结构——
     - **上方 (`.board-panel`)**：306×306px 5×5 螺旋棋盘与右侧特权/布袋/存量信息卡紧密贴合居中，移除多余标题，垂直高度压缩至仅约 310px。
     - **下方 (`.players-panel`)**：先手 P0 与后手 P1 仪表盘呈左右并排呈现：
       1. 名字行精简为纯粹身份与特权数量，移除冗余文字标签；
       2. 手中筹码重构为 7 列垂直筹码堆叠（Chip Stacks，精准匹配各色宝石切片，随数量自底向上 3D 拟真层叠）；
       3. 已购卡牌与永久减免合并，卡牌放大至 58×87px，按 5 种基础颜色分列自下而上层叠露出顶部 28px 卡头（清晰呈现声望⭐、王冠👑与加成💎），并在列头整合当前颜色永久减免 `+Bonus`，万能变色卡直观归属于所附着的颜色堆叠；
       4. 最底栏统一横排展示预留卡 (≤3) 与王室卡 (≤2)，全行最多 5 张卡牌。
   - **右列 (`.sidebar-panel`, 300px)**：操作历史日志面板、AI 决策估值与候选动作排行、折叠式快捷操作指南。
2. **通栏横幅与控制设计**：
   - **顶部 Header**：导航 Tab 切换、双方席位类型选择（`先手 P0` 与 `后手 P1`）、一键【🔄 交换先后手】按钮、神经网络状态灯与热重载、控制按钮。
   - **复盘胜率走势天平折线图 (`eval-chart-container`)**：位于 `replay.html` 顶部，基于 Canvas 专门绘制真值神经网络胜率预期曲线：
     1. **纯粹真值：仅对神经网络绘制胜率曲线**：
        - 彻底剥离启发式规则与人类玩家的伪胜率逻辑（启发式属于静态动作规则打分，不具备全局胜率意义，不予绘制折线，界面保持干净纯粹）。
        - 仅在由神经网络执手的行动步上，绘制 Player 0（天蓝线 `#38bdf8`）与 Player 1（粉红线 `#f472b6`）各自独立真实的自我胜率折线（0% ~ 100%）。
        - 若整局无神经网络执手，Canvas 居中展示中性提示文字，绝不凭空伪造数据。
     2. **游标下钻与客观呈现**：
        - 当前查看步垂直高光指示（金色参考线 + 节点微光）；
        - 鼠标悬停画布时实时吸附到最近步骤，Header 客观区分各角色类型与行动：神经网络步显示真实自评胜率，启发式步显示动作启发分值（如 `+12.0分` / `斩杀`），人类步显示玩家决策，并同步展示中文化动作语义；点击曲线上任意点快速跳转至对应局面。
   - **胜利者专区 (`winner-banner`)**：获胜时展开的全宽金色发光通告栏，标明获胜方与获胜条件。
   - **交互指引条 (`turn-guide-banner`)**：当前回合玩家的行动指南与动作确认按钮区（动态匹配当前行动方蓝/粉色标，并醒目提示上步 AI 动作与胜率估值）。

---

## 7. 启动、调试与开发维护

### 7.1 本地启动服务
在激活虚拟环境后，通过 Cargo 启动内置 Web 服务：
```bash
# 根目录下
cargo run --bin replay_web
```
服务将在默认端口 `http://127.0.0.1:8080` 启动，并在控制台输出静态目录与 API 路由监听状态。

### 7.2 浏览器调试技巧
- 访问 `http://127.0.0.1:8080/play.html` 进入交互对战。
- 访问 `http://127.0.0.1:8080/inspect_cards.html` 可视化比对精灵图索引与对应卡牌切片。
- 在浏览器控制台可通过 `window` 查看或调用相关组件函数，利用 Network 选项卡监控 `/api/game/*` 动作请求与返回。
