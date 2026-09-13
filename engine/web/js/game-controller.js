/**
 * Splendor Duel Interactive Game Controller
 * 璀璨宝石：对决 人类交互对战控制器
 */

import {
  renderBoard,
  renderCardsList,
  renderRoyalsPool,
  renderPlayerDashboard,
  COLOR_CLASSES,
  COLOR_NAMES,
  COLOR_KEYS
} from './shared-components.js';

export class GameController {
  constructor() {
    this.state = null;
    this.legalActions = [];
    this.isHuman = false;
    this.currentPlayer = 0;
    this.playerKinds = ['human', 'neural'];
    this.selectedBoardPositions = [];
    this.autoStepAi = true;
    this.aiStepTimer = null;
    this.isActionPending = false;

    this.initDOMElements();
    this.bindEvents();
  }

  initDOMElements() {
    this.guideBanner = document.getElementById('turnGuideBanner');
    this.guideText = document.getElementById('guideText');
    this.guideBadge = document.getElementById('guideBadge');
    this.actionBarButtons = document.getElementById('actionBarButtons');
    this.modalOverlay = document.getElementById('modalOverlay');
    this.modalTitle = document.getElementById('modalTitle');
    this.modalBody = document.getElementById('modalBody');
    this.modalOptions = document.getElementById('modalOptions');
    this.modalFooter = document.getElementById('modalFooter');
  }

  bindEvents() {
    document.getElementById('btnNewGame').onclick = () => this.startNewGame();
    document.getElementById('btnStepAi').onclick = () => this.stepAi();
    document.getElementById('autoAiCheckbox').onchange = (e) => {
      this.autoStepAi = e.target.checked;
      if (this.autoStepAi && !this.isHuman) {
        this.scheduleAiStep();
      }
    };
    document.getElementById('btnToReplay').onclick = () => this.toReplay();
    document.getElementById('p0KindSelect').onchange = () => this.startNewGame();
    document.getElementById('p1KindSelect').onchange = () => this.startNewGame();

    const badge = document.getElementById('neuralModelBadge');
    if (badge) {
      badge.onclick = () => this.reloadNeuralModel();
    }

    this.checkNeuralStatus();
    // 每 5 秒静默同步一次神经网络状态
    setInterval(() => this.checkNeuralStatus(), 5000);
  }

  async checkNeuralStatus() {
    try {
      const res = await fetch('/api/neural_status');
      if (res.ok) {
        const data = await res.json();
        const dot = document.getElementById('neuralStatusDot');
        const badge = document.getElementById('neuralModelBadge');
        const text = document.getElementById('neuralModelText');

        if (data.available) {
          if (dot) {
            dot.style.color = '#22c55e';
            dot.title = '神经网络推理微服务已连接 (127.0.0.1:8088)';
          }
          if (badge && text && data.details) {
            badge.style.display = 'inline-flex';
            text.innerText = `Epoch ${data.details.epoch ?? '--'}`;
          }
        } else {
          if (dot) {
            dot.style.color = '#ef4444';
            dot.title = '神经网络未就绪，使用启发式 AI 兜底';
          }
          if (badge) badge.style.display = 'none';
        }
      }
    } catch (e) {
      // 静默忽略网络探测异常
    }
  }

  async reloadNeuralModel() {
    const text = document.getElementById('neuralModelText');
    if (text) text.innerText = '重载中...';
    try {
      const res = await fetch('/api/neural_reload', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        if (text) text.innerText = `Epoch ${data.epoch ?? '--'}`;
        alert(`✅ 神经网络权重已成功重载！最新版本: Epoch ${data.epoch}`);
      } else {
        alert('❌ 重载神经网络失败，请检查服务日志');
      }
    } catch (e) {
      alert(`❌ 重载网络请求失败: ${e.message}`);
    }
    this.checkNeuralStatus();
  }

  async startNewGame() {
    this.clearModal();
    this.selectedBoardPositions = [];
    const winnerBanner = document.getElementById('winnerBanner');
    if (winnerBanner) winnerBanner.style.display = 'none';
    const p0 = document.getElementById('p0KindSelect').value;
    const p1 = document.getElementById('p1KindSelect').value;
    const seed = Math.floor(Math.random() * 100000);

    try {
      const res = await fetch(`/api/game/new?seed=${seed}&p0=${p0}&p1=${p1}`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        this.updateData(data);
      }
    } catch (e) {
      console.error('startNewGame failed:', e);
    }
  }

  async fetchState() {
    try {
      const res = await fetch('/api/game/state');
      if (res.ok) {
        const data = await res.json();
        this.updateData(data);
      }
    } catch (e) {
      console.error('fetchState failed:', e);
    }
  }

  updateData(data) {
    this.state = data.state;
    this.legalActions = data.legal_actions || [];
    this.isHuman = data.is_human;
    this.currentPlayer = data.current_player;
    this.playerKinds = data.player_kinds || ['human', 'neural'];

    // 同步下拉框
    const p0Select = document.getElementById('p0KindSelect');
    const p1Select = document.getElementById('p1KindSelect');
    if (p0Select) p0Select.value = this.playerKinds[0];
    if (p1Select) p1Select.value = this.playerKinds[1];

    this.selectedBoardPositions = [];
    this.render();
    this.handleTurnFlow();
  }

  render() {
    if (!this.state) return;

    // 1. 棋盘渲染
    const boardOptions = this.getBoardRenderOptions();
    const tokenCount = renderBoard(this.state.board, document.getElementById('boardGrid'), boardOptions);
    document.getElementById('boardTokensCount').innerText = `${tokenCount} 标记`;
    document.getElementById('privilegePool').innerText = this.state.privilege_pool;
    document.getElementById('bagCount').innerText = this.state.bag_count;

    // 2. 金字塔与卡牌渲染
    this.renderPyramidArea();

    // 3. 王室赞助池
    this.renderRoyalsArea();

    // 4. 玩家仪表盘
    this.renderPlayersArea();

    // 5. 操作指引条
    this.renderGuideBanner();
  }

  getBoardRenderOptions() {
    const phase = this.state.phase;
    const isHumanTurn = this.isHuman && !this.state.winner;

    if (!isHumanTurn) {
      return { clickable: false };
    }

    // 技能：从棋盘拿同色宝石
    if (phase.startsWith('CardAbilitySameColor')) {
      const targetColor = phase.replace('CardAbilitySameColor(', '').replace(')', '').toLowerCase();
      const highlightPositions = [];
      for (let r = 0; r < 5; r++) {
        for (let c = 0; c < 5; c++) {
          if (this.state.board[r][c] === targetColor) {
            highlightPositions.push([r, c]);
          }
        }
      }
      return {
        clickable: true,
        highlightPositions,
        onCellClick: (r, c, gem) => {
          if (gem === targetColor) {
            this.submitAction({ TakeSameColorToken: { r, c } });
          }
        }
      };
    }

    // 预留卡牌连锁：从棋盘选择拿取 1 枚黄金
    if (phase === 'SelectReserveGold') {
      const highlightPositions = [];
      for (let r = 0; r < 5; r++) {
        for (let c = 0; c < 5; c++) {
          if (this.state.board[r][c] === 'gold') {
            highlightPositions.push([r, c]);
          }
        }
      }
      return {
        clickable: true,
        highlightPositions,
        onCellClick: (r, c, gem) => {
          if (gem === 'gold') {
            this.submitAction({ TakeGoldToken: { r, c } });
          }
        }
      };
    }

    // 可选行动：使用特权卷轴拿取非黄金宝石
    if (phase === 'OptionalActions' && this.state.players[this.currentPlayer].privileges > 0) {
      const highlightPositions = [];
      for (let r = 0; r < 5; r++) {
        for (let c = 0; c < 5; c++) {
          const g = this.state.board[r][c];
          if (g && g !== 'gold') highlightPositions.push([r, c]);
        }
      }
      return {
        clickable: true,
        highlightPositions,
        onCellClick: (r, c, gem) => {
          if (gem && gem !== 'gold') {
            this.submitAction({ UsePrivilege: { r, c } });
          }
        }
      };
    }

    // 强制行动：连线拿取 1~3 颗宝石
    if (phase === 'MandatoryAction') {
      const takeActions = this.legalActions.filter(a => a.category === 'take_tokens');
      const candidatePositions = this.calculateCandidatePositions(takeActions);

      return {
        clickable: true,
        selectedPositions: this.selectedBoardPositions,
        candidatePositions,
        onCellClick: (r, c, gem) => this.handleBoardCellClickForTokens(r, c, gem, takeActions)
      };
    }

    return { clickable: false };
  }

  calculateCandidatePositions(takeActions) {
    if (this.selectedBoardPositions.length === 0) {
      // 任何存在于合法 takeActions 中的单个坐标都是候选
      const coords = new Set();
      takeActions.forEach(act => {
        const positions = act.action.TakeTokens.positions;
        const count = act.action.TakeTokens.count;
        for (let i = 0; i < count; i++) {
          coords.add(`${positions[i][0]},${positions[i][1]}`);
        }
      });
      return Array.from(coords).map(s => s.split(',').map(Number));
    }

    if (this.selectedBoardPositions.length >= 3) return [];

    // 当前已选部分坐标，找出所有包含当前选中坐标集的合法 TakeTokens
    const nextCandidates = new Set();
    const selSet = new Set(this.selectedBoardPositions.map(([r, c]) => `${r},${c}`));

    takeActions.forEach(act => {
      const positions = act.action.TakeTokens.positions;
      const count = act.action.TakeTokens.count;
      const actCoords = positions.slice(0, count).map(([r, c]) => `${r},${c}`);

      const containsAllSelected = Array.from(selSet).every(s => actCoords.includes(s));
      if (containsAllSelected && count > this.selectedBoardPositions.length) {
        actCoords.forEach(c => {
          if (!selSet.has(c)) nextCandidates.add(c);
        });
      }
    });

    return Array.from(nextCandidates).map(s => s.split(',').map(Number));
  }

  handleBoardCellClickForTokens(r, c, gem, takeActions) {
    if (!gem || gem === 'gold') return;

    const key = `${r},${c}`;
    const idx = this.selectedBoardPositions.findIndex(([pr, pc]) => pr === r && pc === c);

    if (idx >= 0) {
      // 取消选中
      this.selectedBoardPositions.splice(idx, 1);
      this.render();
      return;
    }

    if (this.selectedBoardPositions.length >= 3) return;

    // 检查是否属于合法连线延伸
    const testList = [...this.selectedBoardPositions, [r, c]];
    const testSet = new Set(testList.map(([tr, tc]) => `${tr},${tc}`));

    const isMatch = takeActions.some(act => {
      const positions = act.action.TakeTokens.positions;
      const count = act.action.TakeTokens.count;
      const actCoords = positions.slice(0, count).map(([ar, ac]) => `${ar},${ac}`);
      return Array.from(testSet).every(s => actCoords.includes(s));
    });

    if (isMatch) {
      this.selectedBoardPositions.push([r, c]);
      this.render();
    }
  }

  renderPyramidArea() {
    const isHumanTurn = this.isHuman && !this.state.winner && this.state.phase === 'MandatoryAction';
    const canReserve = isHumanTurn && (this.state.players[this.currentPlayer].reserved_cards.length < 3);

    // 提取所有可购买的金字塔卡牌
    const affordableIds = new Set();
    if (isHumanTurn) {
      this.legalActions
        .filter(a => a.category === 'purchase_card' && !a.action.PurchaseCard.from_reserved)
        .forEach(a => {
          const { tier, slot } = a.action.PurchaseCard;
          const tierIdx = tier === 'Tier3' ? 2 : (tier === 'Tier2' ? 1 : 0);
          const card = this.state.pyramid[tierIdx][slot];
          if (card) affordableIds.add(card.id);
        });
    }

    const makePyramidOptions = (tierIdx, tierName, tierNum) => ({
      interactive: isHumanTurn,
      affordableIds,
      canReserve,
      deckInfo: {
        tier: tierNum,
        count: this.state.decks_count[tierIdx],
      },
      onReserveDeck: () => {
        if (isHumanTurn && canReserve && this.state.decks_count[tierIdx] > 0) {
          this.submitAction({
            ReserveCard: {
              tier: tierName,
              slot: null
            }
          });
        }
      },
      onPurchase: (card) => {
        const slot = this.state.pyramid[tierIdx].findIndex(c => c.id === card.id);
        if (slot >= 0) {
          this.submitAction({
            PurchaseCard: {
              from_reserved: false,
              tier: tierName,
              slot
            }
          });
        }
      },
      onReserve: (card) => {
        const slot = this.state.pyramid[tierIdx].findIndex(c => c.id === card.id);
        if (slot >= 0) {
          this.submitAction({
            ReserveCard: {
              tier: tierName,
              slot
            }
          });
        }
      }
    });

    renderCardsList(this.state.pyramid[2], document.getElementById('tier3Cards'), makePyramidOptions(2, 'Tier3', 3));
    renderCardsList(this.state.pyramid[1], document.getElementById('tier2Cards'), makePyramidOptions(1, 'Tier2', 2));
    renderCardsList(this.state.pyramid[0], document.getElementById('tier1Cards'), makePyramidOptions(0, 'Tier1', 1));

    document.getElementById('deck3Count').innerText = this.state.decks_count[2];
    document.getElementById('deck2Count').innerText = this.state.decks_count[1];
    document.getElementById('deck1Count').innerText = this.state.decks_count[0];

    // 牌堆暗抽预留按钮
    this.updateDeckReserveBtn('deck3ReserveBtn', 'Tier3', 2, isHumanTurn, canReserve);
    this.updateDeckReserveBtn('deck2ReserveBtn', 'Tier2', 1, isHumanTurn, canReserve);
    this.updateDeckReserveBtn('deck1ReserveBtn', 'Tier1', 0, isHumanTurn, canReserve);
  }

  updateDeckReserveBtn(id, tierName, tierIdx, isHumanTurn, canReserve) {
    const btn = document.getElementById(id);
    if (!btn) return;
    const hasDeck = this.state.decks_count[tierIdx] > 0;
    const isLegal = isHumanTurn && canReserve && hasDeck;
    btn.disabled = !isLegal;
    btn.onclick = () => {
      if (isLegal) {
        this.submitAction({
          ReserveCard: {
            tier: tierName,
            slot: null
          }
        });
      }
    };
  }

  renderRoyalsArea() {
    const isSelectingRoyal = this.isHuman && this.state.phase === 'SelectRoyalCard';
    renderRoyalsPool(this.state.royal_cards, document.getElementById('royalsContainer'), {
      selectable: isSelectingRoyal,
      onSelect: (royal) => {
        this.submitAction({ SelectRoyal: { royal_id: royal.id } });
      }
    });
  }

  renderPlayersArea() {
    const isHumanTurn = this.isHuman && !this.state.winner;
    const phase = this.state.phase;

    // 己方手牌中可购买的预留卡
    const affordableReservedIds = new Set();
    if (isHumanTurn && phase === 'MandatoryAction') {
      this.legalActions
        .filter(a => a.category === 'purchase_card' && a.action.PurchaseCard.from_reserved)
        .forEach(a => {
          const slot = a.action.PurchaseCard.slot;
          const card = this.state.players[this.currentPlayer].reserved_cards[slot];
          if (card) affordableReservedIds.add(card.id);
        });
    }

    // 判断暗牌视角掩蔽规则 (若某方为对手，其暗抽预留牌显示为牌背)
    const p0IsOpponent = (this.playerKinds[0] !== 'human' && this.playerKinds[1] === 'human') ||
                         (this.playerKinds[0] === 'human' && this.playerKinds[1] === 'human' && this.currentPlayer !== 0);
    const p1IsOpponent = (this.playerKinds[1] !== 'human' && this.playerKinds[0] === 'human') ||
                         (this.playerKinds[0] === 'human' && this.playerKinds[1] === 'human' && this.currentPlayer !== 1);

    // Player 0 配置
    const p0IsActive = (this.currentPlayer === 0);
    const p0Options = {
      isOpponent: p0IsOpponent,
      interactiveReserved: isHumanTurn && p0IsActive && (phase === 'MandatoryAction'),
      affordableReservedIds,
      onPurchaseReserved: (card) => {
        const slot = this.state.players[0].reserved_cards.findIndex(c => c.id === card.id);
        if (slot >= 0) {
          this.submitAction({
            PurchaseCard: {
              from_reserved: true,
              tier: card.tier === 3 ? 'Tier3' : (card.tier === 2 ? 'Tier2' : 'Tier1'),
              slot
            }
          });
        }
      },
      tokenClickable: isHumanTurn && (
        (p0IsActive && phase === 'DiscardTokens') || (!p0IsActive && phase === 'CardAbilitySteal')
      ),
      onTokenClick: (gem) => {
        const capGem = gem.charAt(0).toUpperCase() + gem.slice(1).toLowerCase();
        if (p0IsActive && phase === 'DiscardTokens') {
          const match = this.legalActions.find(a => a.category === 'discard' && a.action.DiscardToken && (a.action.DiscardToken.gem.toLowerCase() === gem.toLowerCase()));
          this.submitAction(match ? match.action : { DiscardToken: { gem: capGem } });
        } else if (!p0IsActive && phase === 'CardAbilitySteal') {
          const match = this.legalActions.find(a => a.category === 'steal' && a.action.StealToken && (a.action.StealToken.gem.toLowerCase() === gem.toLowerCase()));
          this.submitAction(match ? match.action : { StealToken: { gem: capGem } });
        }
      }
    };

    // Player 1 配置
    const p1IsActive = (this.currentPlayer === 1);
    const p1Options = {
      isOpponent: p1IsOpponent,
      interactiveReserved: isHumanTurn && p1IsActive && (phase === 'MandatoryAction'),
      affordableReservedIds,
      onPurchaseReserved: (card) => {
        const slot = this.state.players[1].reserved_cards.findIndex(c => c.id === card.id);
        if (slot >= 0) {
          this.submitAction({
            PurchaseCard: {
              from_reserved: true,
              tier: card.tier === 3 ? 'Tier3' : (card.tier === 2 ? 'Tier2' : 'Tier1'),
              slot
            }
          });
        }
      },
      tokenClickable: isHumanTurn && (
        (p1IsActive && phase === 'DiscardTokens') || (!p1IsActive && phase === 'CardAbilitySteal')
      ),
      onTokenClick: (gem) => {
        const capGem = gem.charAt(0).toUpperCase() + gem.slice(1).toLowerCase();
        if (p1IsActive && phase === 'DiscardTokens') {
          const match = this.legalActions.find(a => a.category === 'discard' && a.action.DiscardToken && (a.action.DiscardToken.gem.toLowerCase() === gem.toLowerCase()));
          this.submitAction(match ? match.action : { DiscardToken: { gem: capGem } });
        } else if (!p1IsActive && phase === 'CardAbilitySteal') {
          const match = this.legalActions.find(a => a.category === 'steal' && a.action.StealToken && (a.action.StealToken.gem.toLowerCase() === gem.toLowerCase()));
          this.submitAction(match ? match.action : { StealToken: { gem: capGem } });
        }
      }
    };

    renderPlayerDashboard(this.state.players[0], document.getElementById('player0Card'), p0IsActive, false, 'p0', p0Options);
    renderPlayerDashboard(this.state.players[1], document.getElementById('player1Card'), p1IsActive, false, 'p1', p1Options);
  }

  renderGuideBanner() {
    this.actionBarButtons.innerHTML = '';
    const winnerBanner = document.getElementById('winnerBanner');

    if (this.state.winner) {
      if (winnerBanner) {
        winnerBanner.style.display = 'flex';
        winnerBanner.innerHTML = `<span>🏆 <b>对局结束</b> — 获胜者: <span style="color:#fbbf24; font-size:1.06rem; margin-left:4px;">${this.state.winner}</span></span>`;
      }
      this.guideBadge.innerText = '🏆 胜负已分';
      this.guideBadge.style.backgroundColor = '#dc2626';
      this.guideText.innerHTML = `<span style="color:#ef4444; font-weight:800;">${this.state.winner}！</span> 点击右侧按钮可一键将本局对战转入复盘分析。`;
      return;
    }

    if (winnerBanner) {
      winnerBanner.style.display = 'none';
    }

    if (!this.isHuman) {
      const aiKindName = this.playerKinds[this.currentPlayer];
      this.guideBadge.innerText = '🤖 AI 思考中';
      this.guideBadge.style.backgroundColor = '#6366f1';
      this.guideText.innerText = `当前轮到 Player ${this.currentPlayer} (${aiKindName}) 决策...`;

      if (!this.autoStepAi) {
        const btn = document.createElement('button');
        btn.className = 'btn-accent';
        btn.innerText = '🤖 AI 单步行动';
        btn.onclick = () => this.stepAi();
        this.actionBarButtons.appendChild(btn);
      }
      return;
    }

    // 人类行动阶段
    const phase = this.state.phase;
    this.guideBadge.innerText = `👤 玩家 P${this.currentPlayer} 回合`;
    this.guideBadge.style.backgroundColor = '#059669';

    if (phase === 'OptionalActions') {
      this.guideText.innerText = '【可选阶段】可使用特权卷轴点击棋盘拿取宝石，或补充棋盘，或点击右侧跳过直接进入主阶段。';

      const hasReplenish = this.legalActions.some(a => a.category === 'replenish');
      if (hasReplenish) {
        const btnRep = document.createElement('button');
        btnRep.className = 'btn-accent';
        btnRep.innerText = '🌀 补充棋盘 (送对手特权)';
        btnRep.onclick = () => this.submitAction('ReplenishBoard');
        this.actionBarButtons.appendChild(btnRep);
      }

      const btnSkip = document.createElement('button');
      btnSkip.className = 'btn-secondary';
      btnSkip.innerText = '⏭ 跳过可选行动';
      btnSkip.onclick = () => this.submitAction('SkipOptional');
      this.actionBarButtons.appendChild(btnSkip);
      return;
    }

    if (phase === 'MandatoryAction') {
      const takeActions = this.legalActions.filter(a => a.category === 'take_tokens');
      const matchingTakeAction = this.findMatchingTakeAction(takeActions);

      if (matchingTakeAction) {
        const btnTake = document.createElement('button');
        btnTake.className = 'btn-success';
        btnTake.innerText = `✨ 确认拿取 (${this.selectedBoardPositions.length} 颗宝石)`;
        btnTake.onclick = () => this.submitAction(matchingTakeAction.action);
        this.actionBarButtons.appendChild(btnTake);

        const btnClear = document.createElement('button');
        btnClear.className = 'btn-secondary';
        btnClear.innerText = '✖ 清空选择';
        btnClear.onclick = () => {
          this.selectedBoardPositions = [];
          this.render();
        };
        this.actionBarButtons.appendChild(btnClear);

        this.guideText.innerText = `已选 ${this.selectedBoardPositions.length} 颗宝石，点击按钮确认拿取，或点击金字塔卡牌购买/预留。`;
      } else {
        this.guideText.innerText = '【强制行动】在棋盘上连线点选 1~3 颗非黄金宝石，或直接点击金字塔中卡牌进行购买/预留。';
      }
      return;
    }

    if (phase.startsWith('CardAbilityJoker')) {
      this.guideText.innerText = '【变色卡定色】变色卡触发连锁能力，请指定该卡附着的宝石颜色。';
      this.showJokerColorModal();
      return;
    }

    if (phase.startsWith('CardAbilitySameColor')) {
      this.guideText.innerText = '【拿取同色宝石】请在左侧 5x5 棋盘中点击一颗发光的同色宝石完成拿取。';
      return;
    }

    if (phase === 'SelectReserveGold') {
      this.guideText.innerText = '【选择黄金】预留卡牌成功！请在左侧 5x5 棋盘中点击选择你要拿取的 1 枚黄金。';
      return;
    }

    if (phase === 'CardAbilitySteal') {
      this.guideText.innerText = '【偷取宝石】请在对手手牌区域点击任意一颗非黄金宝石进行偷取。';
      return;
    }

    if (phase === 'SelectRoyalCard') {
      this.guideText.innerText = '【认领王室赞助卡】王冠达到里程碑！请在上方王室赞助池中点击一张卡牌认领。';
      return;
    }

    if (phase === 'DiscardTokens') {
      this.guideText.innerText = '【手牌超限】手牌持有标记超过 10 枚，请点击自己手牌中想要弃置的标记。';
      return;
    }
  }

  findMatchingTakeAction(takeActions) {
    if (this.selectedBoardPositions.length === 0) return null;
    const selSet = new Set(this.selectedBoardPositions.map(([r, c]) => `${r},${c}`));
    const len = this.selectedBoardPositions.length;

    return takeActions.find(act => {
      const positions = act.action.TakeTokens.positions;
      const count = act.action.TakeTokens.count;
      if (count !== len) return false;
      const actCoords = positions.slice(0, count).map(([r, c]) => `${r},${c}`);
      return actCoords.every(c => selSet.has(c));
    });
  }

  showJokerColorModal() {
    const jokerActions = this.legalActions.filter(a => a.category === 'joker');
    if (jokerActions.length === 0) return;

    this.modalTitle.innerText = '🃏 变色复制卡 (Joker) 定色选择';
    this.modalBody.innerText = '请选择要将该百搭变色卡附着到哪种已拥有加成的宝石颜色上：';
    this.modalOptions.innerHTML = '';

    jokerActions.forEach(act => {
      const color = act.action.AssignJokerColor.color;
      const btn = document.createElement('div');
      btn.className = 'gem-picker-btn';
      btn.innerHTML = `
        <div class="token ${COLOR_CLASSES[color]}">${color[0].toUpperCase()}</div>
        <span style="font-size:0.75rem; font-weight:700;">${color}</span>
      `;
      btn.onclick = () => {
        this.clearModal();
        this.submitAction(act.action);
      };
      this.modalOptions.appendChild(btn);
    });

    this.modalFooter.innerHTML = '';
    this.modalOverlay.style.display = 'flex';
  }

  clearModal() {
    this.modalOverlay.style.display = 'none';
    this.modalOptions.innerHTML = '';
  }

  async submitAction(action) {
    if (this.isActionPending) return;
    this.isActionPending = true;

    try {
      const res = await fetch('/api/game/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action })
      });
      if (res.ok) {
        const data = await res.json();
        this.updateData(data);
      } else {
        const err = await res.json();
        console.error('Action rejected:', err);
        alert(`操作无效: ${err.error || '未知错误'}`);
      }
    } catch (e) {
      console.error('submitAction failed:', e);
    } finally {
      this.isActionPending = false;
    }
  }

  async stepAi() {
    if (this.isActionPending) return;
    this.isActionPending = true;

    try {
      const res = await fetch('/api/game/ai_step', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        if (data.ok) {
          this.updateData(data);
        }
      }
    } catch (e) {
      console.error('stepAi failed:', e);
    } finally {
      this.isActionPending = false;
    }
  }

  handleTurnFlow() {
    if (this.state.winner) return;

    if (!this.isHuman && this.autoStepAi) {
      this.scheduleAiStep();
    } else if (this.isHuman && this.state.phase === 'OptionalActions') {
      // 若无可操作的可选行动（无特权且不可补盘），自动跳过进入强制行动阶段
      if (this.legalActions.length === 1 && this.legalActions[0].category === 'skip_optional') {
        this.submitAction('SkipOptional');
      }
    }
  }

  scheduleAiStep() {
    if (this.aiStepTimer) clearTimeout(this.aiStepTimer);
    this.aiStepTimer = setTimeout(() => {
      if (!this.isHuman && this.autoStepAi && !this.state.winner) {
        this.stepAi();
      }
    }, 380);
  }

  async toReplay() {
    try {
      const res = await fetch('/api/game/to_replay', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        window.location.href = data.redirect || 'replay.html';
      }
    } catch (e) {
      console.error('toReplay failed:', e);
    }
  }
}
