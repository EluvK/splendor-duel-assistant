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

function translateColor(color) {
  const map = {
    Blue: '蓝', Red: '红', Green: '绿', White: '白', Black: '黑', Pearl: '珍珠', Gold: '黄金',
    blue: '蓝', red: '红', green: '绿', white: '白', black: '黑', pearl: '珍珠', gold: '黄金'
  };
  return map[color] || color;
}

export function formatFriendlyAction(desc) {
  if (!desc) return '';
  // Take N Tokens: (r, c), ...
  if (desc.startsWith('Take') && desc.includes('Tokens:')) {
    const m = desc.match(/Take (\d+) Tokens: (.*)/);
    if (m) {
      return `拿取 ${m[1]} 颗宝石: ${m[2]}`;
    }
  }
  if (desc.startsWith('Take') && desc.includes('gems:')) {
    const m = desc.match(/Take (\d+) gems: \[(.*?)\]/);
    if (m) {
      const count = m[1];
      const gems = m[2].split(',').map(s => translateColor(s.trim())).join(', ');
      return `拿取 ${count} 颗宝石: ${gems}`;
    }
  }
  // Reserve Tier Tier1 slot 1 (gold at (r,c)) 或 Reserve Tier Tier2 from Deck
  if (desc.startsWith('Reserve Tier')) {
    const goldMatch = desc.match(/gold at \((\d+),(\d+)\)/);
    const goldHint = goldMatch ? ` (+1金(${goldMatch[1]},${goldMatch[2]}))` : ' (+1金)';
    if (desc.includes('from Deck')) {
      const m = desc.match(/Reserve Tier Tier(\d+) from Deck/);
      return m ? `盲抽预留 等级${m[1]}牌库${goldHint}` : `盲抽预留卡牌${goldHint}`;
    }
    const m = desc.match(/Reserve Tier Tier(\d+) slot (\d+)/);
    if (m) {
      return `预留 等级${m[1]} 卡位#${Number(m[2]) + 1}${goldHint}`;
    }
  }
  // Purchase Tier Tier1 slot 0 [plan]
  if (desc.startsWith('Purchase Tier')) {
    const m = desc.match(/Purchase Tier Tier(\d+) slot (\d+)(?: \[(.*?)\])?/);
    if (m) {
      const planHint = m[3] && m[3] !== 'default pay' ? ` (${m[3]})` : '';
      return `购买 等级${m[1]} 卡位#${Number(m[2]) + 1}${planHint}`;
    }
  }
  // Purchase Reserved card #0 [plan]
  if (desc.startsWith('Purchase Reserved card')) {
    const m = desc.match(/#(\d+)(?: \[(.*?)\])?/);
    const planHint = m && m[2] && m[2] !== 'default pay' ? ` (${m[2]})` : '';
    return m ? `购买预留卡 #${Number(m[1]) + 1}${planHint}` : '购买预留卡';
  }
  if (desc.startsWith('Purchase Reserved')) {
    const m = desc.match(/Purchase Reserved (Tier\d+) Card #(\d+)/);
    if (m) {
      const tier = m[1].replace('Tier', '');
      return `购买预留 ${tier}阶 卡牌 #${m[2]}`;
    }
  }
  if (desc.startsWith('Purchase')) {
    const m = desc.match(/Purchase (Tier\d+) Card #(\d+)/);
    if (m) {
      const tier = m[1].replace('Tier', '');
      return `购买 ${tier}阶 卡牌 #${m[2]}`;
    }
  }
  if (desc.startsWith('Reserve')) {
    const m = desc.match(/Reserve (Tier\d+) Card #(\d+)(.*)/);
    if (m) {
      const tier = m[1].replace('Tier', '');
      const goldHint = m[3].includes('got gold') ? ' (+1金)' : '';
      return `预留 ${tier}阶 卡牌 #${m[2]}${goldHint}`;
    }
  }
  if (desc.startsWith('Blind Reserve')) {
    const m = desc.match(/Blind Reserve (Tier\d+)(.*)/);
    if (m) {
      const tier = m[1].replace('Tier', '');
      const goldHint = m[2].includes('got gold') ? ' (+1金)' : '';
      return `盲抽预留 ${tier}阶${goldHint}`;
    }
  }
  if (desc.includes('Replenish Board')) return '🌀 补充棋盘 (赠对手特权)';
  if (desc.includes('Skip Optional (Auto)')) return '⏭ 自动跳过可选行动';
  if (desc.includes('Skip Optional')) return '⏭ 跳过可选阶段';
  if (desc.includes('Use Privilege')) {
    const m = desc.match(/\((\d+),\s*(\d+)\)/);
    return m ? `📜 特权拿取宝石 (${m[1]},${m[2]})` : '📜 使用特权卷轴';
  }
  if (desc.startsWith('Claim Royal Card')) {
    const m = desc.match(/#(\d+)/);
    return m ? `👑 认领王室卡 #${m[1]}` : '👑 认领王室卡';
  }
  if (desc.startsWith('Discard')) {
    const m = desc.match(/Discard (\w+)/);
    return m ? `弃置 1 标记 (${translateColor(m[1])})` : '弃置超限标记';
  }
  if (desc.startsWith('Steal')) {
    const m = desc.match(/Steal (\w+) from Opponent/);
    return m ? `夺取对手标记 (${translateColor(m[1])})` : '夺取对手标记';
  }
  if (desc.startsWith('Take Same Color token')) {
    const m = desc.match(/\((\d+),\s*(\d+)\)/);
    return m ? `拿取同色宝石 (${m[1]},${m[2]})` : '连击拿取同色宝石';
  }
  if (desc === 'Confirm Payment') return '✔ 确认支付方案';
  if (desc.startsWith('Use Gold to Preserve')) {
    const m = desc.match(/Use Gold to Preserve (\w+)/);
    const cn = m ? translateColor(m[1].toLowerCase()) : '';
    return `💰 黄金代付 (保留${cn}宝石)`;
  }
  if (desc.startsWith('Take Gold token')) {
    const m = desc.match(/\((\d+),\s*(\d+)\)/);
    return m ? `拿取黄金预留 (${m[1]},${m[2]})` : '预留拿取 1 黄金';
  }
  if (desc.startsWith('Joker attach to')) {
    const m = desc.match(/attach to (\w+)/);
    return m ? `变色卡附着为 ${translateColor(m[1])}` : desc;
  }
  if (desc.includes('Game Started')) return '🎮 对局开始';
  return desc;
}

export class GameController {
  constructor() {
    this.state = null;
    this.prevState = null;
    this.legalActions = [];
    this.isHuman = false;
    this.currentPlayer = 0;
    this.playerKinds = ['human', 'neural'];
    this.selectedBoardPositions = [];
    this.selectedGoldPos = null;
    this.pendingReserveTarget = null;
    this.autoStepAi = true;
    this.aiStepTimer = null;
    this.isActionPending = false;

    // 动作幽灵残留追踪 (Ghost Highlighting)
    this.ghostTokens = [];
    this.ghostTimer = null;

    // 对战操作历史状态
    this.history = [];
    this.lastRenderedRound = 0;
    this.autoScrollLog = true;

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

    // 历史面板与单步详情 DOM
    this.logContainer = document.getElementById('gameLogContainer');
    this.logStepBadge = document.getElementById('gameLogStepBadge');
    this.btnToggleLogAutoScroll = document.getElementById('btnToggleLogAutoScroll');
    this.stepDetailModal = document.getElementById('stepDetailModal');
    this.stepDetailTitle = document.getElementById('stepDetailTitle');
    this.stepDetailBody = document.getElementById('stepDetailBody');
    this.btnStepDetailClose = document.getElementById('btnStepDetailClose');
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
    document.getElementById('p0KindSelect').onchange = () => this.onPlayerKindChange(0);
    document.getElementById('p1KindSelect').onchange = () => this.onPlayerKindChange(1);

    // 交换先后手
    const btnSwap = document.getElementById('btnSwapPlayers');
    if (btnSwap) {
      btnSwap.onclick = () => this.swapPlayers();
    }

    // 自动滚动控制
    if (this.btnToggleLogAutoScroll) {
      this.btnToggleLogAutoScroll.onclick = () => {
        this.autoScrollLog = !this.autoScrollLog;
        if (this.autoScrollLog) {
          this.btnToggleLogAutoScroll.classList.add('active');
          if (this.logContainer) {
            this.logContainer.scrollTop = this.logContainer.scrollHeight;
          }
        } else {
          this.btnToggleLogAutoScroll.classList.remove('active');
        }
      };
    }

    // 步骤详情弹窗关闭
    if (this.btnStepDetailClose) {
      this.btnStepDetailClose.onclick = () => this.closeStepDetail();
    }
    if (this.stepDetailModal) {
      this.stepDetailModal.onclick = (e) => {
        if (e.target === this.stepDetailModal) this.closeStepDetail();
      };
    }

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

  onPlayerKindChange(slot) {
    const p0Select = document.getElementById('p0KindSelect');
    const p1Select = document.getElementById('p1KindSelect');
    if (!p0Select || !p1Select) return;

    // 若用户修改一方席位，在单人对战人机情境下智能联动，避免误触双 AI
    if (slot === 0) {
      if (p0Select.value !== 'human' && p1Select.value !== 'human') {
        p1Select.value = 'human';
      }
    } else {
      if (p1Select.value !== 'human' && p0Select.value !== 'human') {
        p0Select.value = 'human';
      }
    }
    this.startNewGame();
  }

  async swapPlayers() {
    const p0Select = document.getElementById('p0KindSelect');
    const p1Select = document.getElementById('p1KindSelect');
    if (!p0Select || !p1Select) return;
    const temp = p0Select.value;
    p0Select.value = p1Select.value;
    p1Select.value = temp;
    await this.startNewGame();
  }

  async startNewGame() {
    this.clearModal();
    this.selectedBoardPositions = [];
    this.selectedGoldPos = null;
    this.pendingReserveTarget = null;
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
    const prevBoard = this.state ? this.state.board : null;
    this.prevState = this.state;
    this.state = data.state;
    this.legalActions = data.legal_actions || [];
    this.isHuman = data.is_human;
    this.currentPlayer = data.current_player;
    if (data.player_kinds && Array.isArray(data.player_kinds) && data.player_kinds.length === 2) {
      this.playerKinds = data.player_kinds;
    } else if (!this.playerKinds || this.playerKinds.length !== 2) {
      this.playerKinds = ['human', 'neural'];
    }

    // 动作幽灵残留追踪提取 (Ghost Highlighting)
    const latestStep = data.step || (data.history && data.history.length > 0 ? data.history[data.history.length - 1] : null);
    if (latestStep && (latestStep.action || latestStep.action_desc)) {
      const actTxt = latestStep.action_desc || latestStep.action;
      this.extractGhostTokensFromAction(actTxt, prevBoard);
    }

    // 同步下拉框
    const p0Select = document.getElementById('p0KindSelect');
    const p1Select = document.getElementById('p1KindSelect');
    if (p0Select && this.playerKinds[0]) p0Select.value = this.playerKinds[0];
    if (p1Select && this.playerKinds[1]) p1Select.value = this.playerKinds[1];

    this.selectedBoardPositions = [];
    this.selectedGoldPos = null;
    this.pendingReserveTarget = null;

    // 处理对战操作历史
    if (data.history) {
      this.syncHistoryList(data.history);
    } else if (data.step) {
      this.appendHistoryStep(data.step, true);
    }

    this.render();
    this.handleTurnFlow();
  }

  extractGhostTokensFromAction(desc, prevBoard) {
    if (!desc) return;
    const matches = desc.matchAll(/\((\d+),\s*(\d+)\)/g);
    const coords = [];
    for (const m of matches) {
      coords.push([parseInt(m[1], 10), parseInt(m[2], 10)]);
    }

    if (coords.length > 0 && prevBoard) {
      const ghosts = [];
      coords.forEach(([r, c]) => {
        if (r >= 0 && r < 5 && c >= 0 && c < 5) {
          const prevGem = prevBoard[r][c];
          if (prevGem) {
            ghosts.push({ r, c, gem: prevGem, label: '取走' });
          }
        }
      });
      if (ghosts.length > 0) {
        this.ghostTokens = ghosts;
        if (this.ghostTimer) clearTimeout(this.ghostTimer);
        this.ghostTimer = setTimeout(() => {
          this.ghostTokens = [];
          if (this.state) {
            const boardOptions = this.getBoardRenderOptions();
            renderBoard(this.state.board, document.getElementById('boardGrid'), boardOptions);
          }
        }, 2600);
      }
    }
  }

  /* 历史记录同步与展示 */
  syncHistoryList(steps) {
    if (!steps || !Array.isArray(steps)) return;
    this.history = steps;

    if (!this.logContainer) return;
    this.logContainer.innerHTML = '';
    this.lastRenderedRound = 0;

    steps.forEach((s, idx) => {
      const isLatest = (idx === steps.length - 1);
      this.appendLogItem(s, false);
    });

    if (this.autoScrollLog && this.logContainer) {
      this.logContainer.scrollTop = this.logContainer.scrollHeight;
    }
  }

  appendHistoryStep(step, isNew = false) {
    if (!step) return;
    const summary = {
      index: step.step_index ?? this.history.length,
      round: step.round_number ?? 1,
      player: step.player,
      action: step.action_desc,
      phase: step.phase,
      score: step.decision?.chosen_score,
      ai_type: step.decision?.ai_type,
    };

    if (!this.history.some(h => h.index === summary.index)) {
      this.history.push(summary);
    }

    this.appendLogItem(summary, isNew);
  }

  appendLogItem(s, isNew = false) {
    const container = this.logContainer;
    if (!container) return;
    if (container.querySelector(`[data-index="${s.index}"]`)) return;

    const round = s.round || 1;
    if (round > this.lastRenderedRound) {
      this.lastRenderedRound = round;
      const divider = document.createElement('div');
      divider.className = 'log-round-divider';
      divider.innerHTML = `<span>⏳ 第 ${round} 轮 (Round ${round})</span>`;
      container.appendChild(divider);
    }

    const item = document.createElement('div');
    const isAi = (s.player !== undefined && this.playerKinds[s.player] !== 'human') || (s.ai_type && s.ai_type !== 'human');
    item.className = `log-item ${isAi ? 'ai-step' : 'human-step'} ${isNew ? 'just-executed' : ''}`;
    item.setAttribute('data-index', s.index);

    const friendlyAct = formatFriendlyAction(s.action);
    const rawScore = s.score;
    let scoreBadgeHtml = '';

    if (rawScore !== undefined && rawScore !== null) {
      const isNeural = (s.ai_type && s.ai_type.includes('neural'));
      if (isNeural) {
        const winrate = (rawScore + 100.0) / 2.0;
        scoreBadgeHtml = `<span class="log-score-tag" title="神经网络预测胜率: ${winrate.toFixed(1)}%">${winrate.toFixed(0)}%</span>`;
      } else {
        const sc = rawScore;
        const txt = sc >= 1000 ? '斩杀' : (sc >= 0 ? `+${sc.toFixed(1)}` : sc.toFixed(1));
        scoreBadgeHtml = `<span class="log-score-tag" title="启发式估值: ${txt}">${txt}</span>`;
      }
    }

    const pColor = (s.player === 0 ? '#38bdf8' : '#f472b6');
    const aiIcon = isAi ? (s.ai_type?.includes('neural') ? '🧠' : '🤖') : '👤';

    item.title = `第 ${round} 轮 #${s.index} [P${s.player}] ${s.action} (${s.phase || ''})\n💡 点击查看详细决策与评估候选`;

    item.innerHTML = `
      <span class="log-idx">#${s.index}</span>
      <span class="log-p" style="color:${pColor};" title="Player ${s.player} (${isAi ? 'AI' : '人类'})">${aiIcon}P${s.player}</span>
      <span class="log-act" title="${friendlyAct}">${friendlyAct}</span>
      ${scoreBadgeHtml}
    `;

    item.onclick = () => this.showStepDetail(s.index);
    container.appendChild(item);

    if (this.logStepBadge) {
      this.logStepBadge.innerText = `${s.index + 1} 步`;
    }

    const prevActive = container.querySelector('.active-step');
    if (prevActive) prevActive.classList.remove('active-step');
    item.classList.add('active-step');

    if (this.autoScrollLog) {
      container.scrollTop = container.scrollHeight;
    }
  }

  async showStepDetail(stepIndex) {
    try {
      const res = await fetch(`/api/game/step?index=${stepIndex}`);
      if (!res.ok) return;
      const step = await res.json();
      this.renderStepDetailModal(step);
    } catch (e) {
      console.error('showStepDetail failed:', e);
    }
  }

  renderStepDetailModal(step) {
    if (!this.stepDetailModal || !this.stepDetailBody) return;

    const round = step.round_number || 1;
    const isP0 = (step.player === 0);
    const pColor = isP0 ? '#38bdf8' : '#f472b6';
    const playerKind = this.playerKinds[step.player] || '未知';
    const friendlyAction = formatFriendlyAction(step.action_desc);

    this.stepDetailTitle.innerHTML = `<span>第 ${round} 轮 · 步骤 #${step.step_index} 决策详情</span>`;

    let decisionHtml = '';
    const decision = step.decision;

    if (decision) {
      const isNeural = decision.ai_type && decision.ai_type.includes('neural');
      const isRandom = decision.ai_type === 'random';
      const isHuman = decision.ai_type === 'human';

      let badgeText = '';
      let badgeColor = '';
      let evalText = '';

      if (isNeural) {
        badgeText = `🧠 神经网络 AI (${decision.ai_type})`;
        badgeColor = '#22c55e';
        if (decision.chosen_score !== null && decision.chosen_score !== undefined) {
          const winrate = (decision.chosen_score + 100.0) / 2.0;
          evalText = `<b style="color:${winrate >= 50 ? '#22c55e' : '#f87171'}; font-size:1.05rem;">${winrate.toFixed(1)}%</b> 胜率预期`;
        }
      } else if (isRandom) {
        badgeText = '🎲 随机 AI (Random)';
        badgeColor = '#f59e0b';
        evalText = '随机均匀抽样，无估值打分';
      } else if (isHuman) {
        badgeText = '👤 人类玩家自主操作';
        badgeColor = '#38bdf8';
        evalText = '由人类根据盘面策略手动选择';
      } else {
        badgeText = `🤖 启发式 AI (${decision.ai_type})`;
        badgeColor = 'var(--primary)';
        if (decision.chosen_score !== null && decision.chosen_score !== undefined) {
          const sc = decision.chosen_score;
          evalText = `<b style="color:#22c55e; font-size:1.05rem;">${sc >= 1000 ? '+9999 斩杀' : (sc >= 0 ? `+${sc.toFixed(1)}` : sc.toFixed(1))}</b> 分`;
        }
      }

      let candidatesHtml = '';
      if (decision.top_candidates && decision.top_candidates.length > 0) {
        candidatesHtml = `
          <div style="margin-top:12px;">
            <div style="font-size:0.75rem; color:var(--accent-gold); font-weight:700; margin-bottom:6px; display:flex; justify-content:space-between;">
              <span>${isNeural ? '🎯 神经网络策略分布 (Policy Head)' : '📊 备选动作估值排名 (Top Candidates)'}</span>
              <span style="font-size:0.68rem; color:var(--text-muted); font-weight:normal;">共 ${decision.top_candidates.length} 项</span>
            </div>
            <div style="background:#111520; border-radius:6px; border:1px solid rgba(255,255,255,0.08); padding:4px; max-height:220px; overflow-y:auto;">
              ${decision.top_candidates.map((c, i) => {
                const friendlyCand = formatFriendlyAction(c.action_desc);
                const isChosen = c.is_chosen;
                let scoreStr = '';
                if (isNeural) {
                  scoreStr = `${c.score.toFixed(1)}%`;
                } else {
                  scoreStr = c.score >= 1000 ? '斩杀' : (c.score >= 0 ? `+${c.score.toFixed(1)}` : c.score.toFixed(1));
                }
                return `
                  <div class="candidate-row ${isChosen ? 'chosen' : ''}" style="display:flex; justify-content:space-between; align-items:center; padding:5px 8px; border-radius:4px; margin-bottom:2px; font-size:0.73rem; background:${isChosen ? 'rgba(99, 102, 241, 0.2)' : 'transparent'};">
                    <span style="color:#64748b; font-size:0.68rem; min-width:18px;">${i + 1}.</span>
                    <span class="candidate-act" style="flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; margin:0 6px;" title="${c.action_desc}">${friendlyCand}</span>
                    ${isChosen ? '<span class="candidate-chosen-badge" style="background:var(--accent-gold); color:#000; font-size:0.65rem; padding:1px 5px; border-radius:3px; font-weight:700; margin-right:6px;">★ 采纳</span>' : ''}
                    <span class="candidate-score" style="color:${isChosen ? '#22c55e' : '#38bdf8'}; font-family:monospace; font-weight:700;">${scoreStr}</span>
                  </div>
                `;
              }).join('')}
            </div>
          </div>
        `;
      }

      decisionHtml = `
        <div style="background:#141926; border-radius:6px; border:1px solid rgba(255,255,255,0.08); padding:10px; margin-top:8px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <span style="font-size:0.75rem; color:var(--text-muted);">决策引擎:</span>
            <span style="font-size:0.72rem; color:${badgeColor}; background:rgba(255,255,255,0.05); padding:2px 8px; border-radius:4px; border:1px solid ${badgeColor}; font-weight:700;">${badgeText}</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:0.75rem; color:var(--text-muted);">局面评估:</span>
            <span style="font-size:0.85rem; color:var(--text-main);">${evalText}</span>
          </div>
          ${candidatesHtml}
        </div>
      `;
    }

    this.stepDetailBody.innerHTML = `
      <div style="display:flex; flex-direction:column; gap:8px;">
        <div style="background:#111520; border-radius:6px; padding:10px; border:1px solid rgba(255,255,255,0.06);">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span style="font-size:0.75rem; color:var(--text-muted);">行动方:</span>
            <b style="color:${pColor}; font-size:0.85rem;">Player ${step.player} (${playerKind})</b>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span style="font-size:0.75rem; color:var(--text-muted);">所属阶段:</span>
            <span style="font-size:0.75rem; color:var(--accent-gold); font-family:monospace;">${step.phase}</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:0.75rem; color:var(--text-muted);">执行动作:</span>
            <span style="font-size:0.85rem; font-weight:700; color:#fff;" title="${step.action_desc}">${friendlyAction}</span>
          </div>
        </div>
        ${decisionHtml}
      </div>
    `;

    this.stepDetailModal.style.display = 'flex';
  }

  closeStepDetail() {
    if (this.stepDetailModal) {
      this.stepDetailModal.style.display = 'none';
    }
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
    const baseOptions = {
      ghostTokens: this.ghostTokens,
      goldSelectedPos: this.selectedGoldPos,
    };

    if (!isHumanTurn) {
      return { clickable: false, ...baseOptions };
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
        ...baseOptions,
        clickable: true,
        highlightPositions,
        onCellClick: (r, c, gem) => {
          if (gem === targetColor) {
            this.submitAction({ TakeSameColorToken: { r, c } });
          }
        }
      };
    }

    // 可选行动：使用特权卷轴拿取非黄金宝石（必须在补盘前使用）
    if (phase === 'OptionalActions' && !this.state.replenished_this_turn && this.state.players[this.currentPlayer].privileges > 0) {
      const highlightPositions = [];
      for (let r = 0; r < 5; r++) {
        for (let c = 0; c < 5; c++) {
          const g = this.state.board[r][c];
          if (g && g !== 'gold') highlightPositions.push([r, c]);
        }
      }
      return {
        ...baseOptions,
        clickable: true,
        highlightPositions,
        onCellClick: (r, c, gem) => {
          if (gem && gem !== 'gold') {
            this.submitAction({ UsePrivilege: { r, c } });
          }
        }
      };
    }

    // 强制行动：连线拿取 1~3 颗非黄金宝石，或直接点击黄金开启预留
    if (phase === 'MandatoryAction') {
      const takeActions = this.legalActions.filter(a => a.category === 'take_tokens');
      const reserveActions = this.legalActions.filter(a => a.category === 'reserve_card');
      const goldPositions = [...new Set(reserveActions.map(a => `${a.action.ReserveCard.gold_pos[0]},${a.action.ReserveCard.gold_pos[1]}`))].map(s => s.split(',').map(Number));

      // 若处于“已选定卡牌/牌堆、等待挑选棋盘黄金”的待选阶段
      if (this.pendingReserveTarget) {
        return {
          ...baseOptions,
          clickable: true,
          candidatePositions: goldPositions.slice(),
          goldCandidatePositions: goldPositions.slice(),
          onCellClick: (r, c, gem) => {
            if (gem === 'gold') {
              const isAvailableGold = goldPositions.some(([gr, gc]) => gr === r && gc === c);
              if (isAvailableGold) {
                const target = this.pendingReserveTarget;
                this.pendingReserveTarget = null;
                this.selectedGoldPos = null;
                this.submitAction({
                  ReserveCard: {
                    gold_pos: [r, c],
                    tier: target.tier,
                    slot: target.slot
                  }
                });
                return;
              }
            }
            // 若用户点击了非黄金宝石，放弃待预留状态，平滑转入连线拿宝石逻辑
            if (gem && gem !== 'gold') {
              this.pendingReserveTarget = null;
              this.selectedGoldPos = null;
              this.handleBoardCellClickForTokens(r, c, gem, takeActions);
            }
          }
        };
      }

      const candidatePositions = this.calculateCandidatePositions(takeActions);
      // 未选定连线时，棋盘上的可用黄金同样作为可点击候选
      if (this.selectedBoardPositions.length === 0) {
        goldPositions.forEach(([gr, gc]) => candidatePositions.push([gr, gc]));
      }

      const activeSelected = this.selectedBoardPositions.slice();
      if (this.selectedGoldPos) {
        activeSelected.push(this.selectedGoldPos);
      }

      return {
        ...baseOptions,
        clickable: true,
        selectedPositions: activeSelected,
        candidatePositions,
        onCellClick: (r, c, gem) => {
          if (gem === 'gold') {
            if (this.selectedBoardPositions.length === 0 && goldPositions.some(([gr, gc]) => gr === r && gc === c)) {
              this.selectedGoldPos = (this.selectedGoldPos && this.selectedGoldPos[0] === r && this.selectedGoldPos[1] === c) ? null : [r, c];
              this.render();
            }
            return;
          }
          this.selectedGoldPos = null;
          this.handleBoardCellClickForTokens(r, c, gem, takeActions);
        }
      };
    }

    return { clickable: false, ...baseOptions };
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
    const isHumanTurn = this.isHuman && !this.state.winner;
    const isMandatory = isHumanTurn && this.state.phase === 'MandatoryAction';
    const reserveActions = isMandatory ? this.legalActions.filter(a => a.category === 'reserve_card') : [];
    const canReserve = isMandatory && reserveActions.length > 0;
    const interactive = isMandatory;

    // 提取所有可购买的金字塔卡牌
    const affordableIds = new Set();
    if (isMandatory) {
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
      interactive,
      affordableIds,
      canReserve,
      pendingReserveTarget: this.pendingReserveTarget,
      isReserveGuidance: Boolean(this.selectedGoldPos) || Boolean(this.pendingReserveTarget),
      currentPlayerState: this.state.players ? this.state.players[this.currentPlayer] : null,
      deckInfo: {
        tier: tierNum,
        count: this.state.decks_count[tierIdx],
      },
      onReserveDeck: () => {
        if (!canReserve || this.state.decks_count[tierIdx] <= 0) return;
        // 若已在棋盘上先选定了黄金，直接完成盲抽预留
        if (this.selectedGoldPos) {
          const gold_pos = this.selectedGoldPos;
          this.selectedGoldPos = null;
          this.pendingReserveTarget = null;
          this.submitAction({
            ReserveCard: {
              gold_pos,
              tier: tierName,
              slot: null
            }
          });
        } else {
          // 未选黄金时：开启/切换待预留牌堆目标，由用户自主在棋盘上选择要拿哪颗黄金
          if (this.pendingReserveTarget && this.pendingReserveTarget.isDeck && this.pendingReserveTarget.tier === tierName) {
            this.pendingReserveTarget = null;
          } else {
            this.pendingReserveTarget = {
              tier: tierName,
              slot: null,
              card: null,
              isDeck: true
            };
          }
          this.render();
        }
      },
      onPurchase: (card) => {
        if (!isMandatory) return;
        this.pendingReserveTarget = null;
        this.selectedGoldPos = null;
        const slot = this.state.pyramid[tierIdx].findIndex(c => c.id === card.id);
        if (slot >= 0) {
          const buyActs = this.legalActions.filter(a =>
            a.category === 'purchase_card' &&
            !a.action.PurchaseCard.from_reserved &&
            a.action.PurchaseCard.tier === tierName &&
            a.action.PurchaseCard.slot === slot
          );
          if (buyActs.length === 1) {
            this.submitAction(buyActs[0].action);
          } else if (buyActs.length > 1) {
            this.showPurchasePlanModal(card, buyActs);
          }
        }
      },
      onReserve: (card) => {
        if (!canReserve) return;
        const slot = this.state.pyramid[tierIdx].findIndex(c => c.id === card.id);
        if (slot < 0) return;

        // 若已在棋盘上先选定了黄金，直接完成该卡牌预留
        if (this.selectedGoldPos) {
          const gold_pos = this.selectedGoldPos;
          this.selectedGoldPos = null;
          this.pendingReserveTarget = null;
          this.submitAction({
            ReserveCard: {
              gold_pos,
              tier: tierName,
              slot
            }
          });
        } else {
          // 未选黄金时：开启/切换待预留卡牌目标，由用户自主在棋盘上选择要拿哪颗黄金
          if (this.pendingReserveTarget && !this.pendingReserveTarget.isDeck && this.pendingReserveTarget.card?.id === card.id) {
            this.pendingReserveTarget = null;
          } else {
            this.pendingReserveTarget = {
              tier: tierName,
              slot,
              card,
              isDeck: false
            };
          }
          this.render();
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

    const isPendingDeck = Boolean(
      this.pendingReserveTarget &&
      this.pendingReserveTarget.isDeck &&
      this.pendingReserveTarget.tier === tierName
    );

    if (isPendingDeck) {
      btn.classList.add('active-pending-reserve');
      btn.title = '📌 正在为此牌堆选择棋盘黄金，点击可取消';
    } else {
      btn.classList.remove('active-pending-reserve');
      if (!canReserve && isHumanTurn && this.state.phase === 'MandatoryAction') {
        btn.title = '当前不可预留（预留手牌已满 3 张或棋盘上无黄金）';
      } else if (this.selectedGoldPos) {
        btn.title = `✨ 已选定黄金 (${this.selectedGoldPos[0]},${this.selectedGoldPos[1]})，点击盲抽预留 1 张牌堆顶暗牌放入手牌`;
      } else if (canReserve) {
        btn.title = '点击开启该牌堆盲抽预留，随后在棋盘上选定要拿取的黄金';
      }
    }

    btn.onclick = () => {
      if (!isLegal) return;
      if (this.selectedGoldPos) {
        const gold_pos = this.selectedGoldPos;
        this.selectedGoldPos = null;
        this.pendingReserveTarget = null;
        this.submitAction({
          ReserveCard: {
            gold_pos,
            tier: tierName,
            slot: null
          }
        });
      } else {
        if (isPendingDeck) {
          this.pendingReserveTarget = null;
        } else {
          this.pendingReserveTarget = {
            tier: tierName,
            slot: null,
            card: null,
            isDeck: true
          };
        }
        this.render();
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

    // 判断暗牌视角掩蔽规则 (人类永远看清自己的暗抽卡，对手 AI 的暗抽卡为牌背)
    let p0IsOpponent = false;
    let p1IsOpponent = false;

    const p0IsHuman = (this.playerKinds[0] === 'human');
    const p1IsHuman = (this.playerKinds[1] === 'human');

    if (p0IsHuman && p1IsHuman) {
      // 双人热座：非当前行动方为对手，暗抽牌背
      p0IsOpponent = (this.currentPlayer !== 0);
      p1IsOpponent = (this.currentPlayer !== 1);
    } else if (p0IsHuman && !p1IsHuman) {
      // P0 为人类，P1 为 AI：人类永远看自己牌，对手 AI 暗抽为牌背
      p0IsOpponent = false;
      p1IsOpponent = true;
    } else if (!p0IsHuman && p1IsHuman) {
      // P0 为 AI，P1 为人类：人类永远看自己牌，对手 AI 暗抽为牌背
      p0IsOpponent = true;
      p1IsOpponent = false;
    } else {
      // 双 AI 自博弈：全公开观战
      p0IsOpponent = false;
      p1IsOpponent = false;
    }

    // Player 0 配置
    const p0IsActive = (this.currentPlayer === 0);
    const p0Options = {
      isOpponent: p0IsOpponent,
      playerKind: this.playerKinds[0],
      interactiveReserved: isHumanTurn && p0IsActive && (phase === 'MandatoryAction'),
      affordableReservedIds,
      onPurchaseReserved: (card) => {
        const slot = this.state.players[0].reserved_cards.findIndex(c => c.id === card.id);
        if (slot >= 0) {
          const tierStr = card.tier === 3 ? 'Tier3' : (card.tier === 2 ? 'Tier2' : 'Tier1');
          const buyActs = this.legalActions.filter(a =>
            a.category === 'purchase_card' &&
            a.action.PurchaseCard.from_reserved &&
            a.action.PurchaseCard.slot === slot
          );
          if (buyActs.length === 1) {
            this.submitAction(buyActs[0].action);
          } else if (buyActs.length > 1) {
            this.showPurchasePlanModal(card, buyActs);
          }
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
      playerKind: this.playerKinds[1],
      interactiveReserved: isHumanTurn && p1IsActive && (phase === 'MandatoryAction'),
      affordableReservedIds,
      onPurchaseReserved: (card) => {
        const slot = this.state.players[1].reserved_cards.findIndex(c => c.id === card.id);
        if (slot >= 0) {
          const tierStr = card.tier === 3 ? 'Tier3' : (card.tier === 2 ? 'Tier2' : 'Tier1');
          const buyActs = this.legalActions.filter(a =>
            a.category === 'purchase_card' &&
            a.action.PurchaseCard.from_reserved &&
            a.action.PurchaseCard.slot === slot
          );
          if (buyActs.length === 1) {
            this.submitAction(buyActs[0].action);
          } else if (buyActs.length > 1) {
            this.showPurchasePlanModal(card, buyActs);
          }
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
        const curRound = this.state.round_number || this.state.turn_number;
        winnerBanner.innerHTML = `<span>🏆 <b>对局结束</b> — 获胜者: <span style="color:#fbbf24; font-size:1.06rem; margin-left:4px;">${this.state.winner}</span> <span style="font-size:0.8rem; color:#94a3b8; margin-left:14px;">(耗时: 共 ${curRound} 轮)</span></span>`;
      }
      this.guideBadge.innerText = '🏆 胜负已分';
      this.guideBadge.style.backgroundColor = '#dc2626';
      this.guideText.innerHTML = `<span style="color:#ef4444; font-weight:800;">${this.state.winner}！</span> 点击右侧按钮可一键将本局对战转入复盘分析。`;
      return;
    }

    if (winnerBanner) {
      winnerBanner.style.display = 'none';
    }

    const curRound = this.state.round_number || this.state.turn_number;
    const isP0 = (this.currentPlayer === 0);
    const orderText = isP0 ? '先手' : '后手';

    if (!this.isHuman) {
      const aiKindName = this.playerKinds[this.currentPlayer];
      let aiIcon = '🤖';
      if (aiKindName === 'neural') aiIcon = '🧠';
      else if (aiKindName === 'random') aiIcon = '🎲';

      this.guideBadge.innerText = `${aiIcon} AI 思考中 (P${this.currentPlayer} ${orderText} · 第 ${curRound} 轮)`;
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
    this.guideBadge.innerText = `👤 玩家 P${this.currentPlayer} (${orderText} · 第 ${curRound} 轮)`;
    this.guideBadge.style.backgroundColor = isP0 ? '#0284c7' : '#db2777';

    // 检查上一条记录是否为对方 AI 的操作，若是，在指引条提示
    let lastAiHint = '';
    const lastStep = this.history.length > 0 ? this.history[this.history.length - 1] : null;
    if (lastStep && lastStep.player !== this.currentPlayer && (this.playerKinds[lastStep.player] !== 'human' || (lastStep.ai_type && lastStep.ai_type !== 'human'))) {
      const friendlyLast = formatFriendlyAction(lastStep.action);
      let scoreTxt = '';
      if (lastStep.score !== undefined && lastStep.score !== null) {
        if (lastStep.ai_type && lastStep.ai_type.includes('neural')) {
          scoreTxt = ` [预期胜率 ${((lastStep.score + 100) / 2).toFixed(0)}%]`;
        }
      }
      lastAiHint = `<span style="display:inline-flex; align-items:center; gap:3px; margin-right:8px; padding:1px 6px; background:rgba(99,102,241,0.2); border:1px solid rgba(99,102,241,0.4); border-radius:4px; font-size:0.75rem; color:#a5b4fc;">🤖 AI (P${lastStep.player}) 上步: <b>${friendlyLast}</b>${scoreTxt}</span>`;
    }

    if (phase === 'OptionalActions') {
      this.guideText.innerHTML = `${lastAiHint}【可选阶段】可先使用特权卷轴点击棋盘拿取宝石；若补充棋盘则可选行动结束并进入强制行动；亦可直接跳过。`;

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

        this.guideText.innerHTML = `${lastAiHint}已选 ${this.selectedBoardPositions.length} 颗宝石，点击按钮确认拿取，或点击金字塔卡牌购买。`;
      } else if (this.pendingReserveTarget) {
        this.guideBadge.innerText = `📌 预留选金 (P${this.currentPlayer})`;
        this.guideBadge.style.backgroundColor = '#d97706';

        let targetDesc = '';
        if (this.pendingReserveTarget.isDeck) {
          const tNum = this.pendingReserveTarget.tier.replace('Tier', '');
          targetDesc = `【等级 ${tNum} 牌堆暗牌】`;
        } else {
          const c = this.pendingReserveTarget.card;
          targetDesc = `【等级 ${c.tier} · 卡牌 #${c.id}】`;
        }

        this.guideText.innerHTML = `${lastAiHint}已选定预留 <b style="color:var(--accent-gold);">${targetDesc}</b>。请在右侧 5×5 棋盘中<b>点击你要拿取的那一颗黄金</b>，或点击快捷按钮：`;

        const reserveActions = this.legalActions.filter(a => a.category === 'reserve_card');
        const goldPositions = [...new Set(reserveActions.map(a => `${a.action.ReserveCard.gold_pos[0]},${a.action.ReserveCard.gold_pos[1]}`))].map(s => s.split(',').map(Number));

        goldPositions.forEach(([gr, gc]) => {
          const btnGold = document.createElement('button');
          btnGold.className = 'btn-accent';
          btnGold.innerHTML = `💰 拿取黄金 (${gr}, ${gc})`;
          btnGold.title = `从棋盘坐标 (${gr}, ${gc}) 拿取黄金并预留 ${targetDesc}`;
          btnGold.onclick = () => {
            const target = this.pendingReserveTarget;
            this.pendingReserveTarget = null;
            this.selectedGoldPos = null;
            this.submitAction({
              ReserveCard: {
                gold_pos: [gr, gc],
                tier: target.tier,
                slot: target.slot
              }
            });
          };
          this.actionBarButtons.appendChild(btnGold);
        });

        const btnCancel = document.createElement('button');
        btnCancel.className = 'btn-secondary';
        btnCancel.innerText = '✖ 取消预留';
        btnCancel.onclick = () => {
          this.pendingReserveTarget = null;
          this.render();
        };
        this.actionBarButtons.appendChild(btnCancel);
      } else if (this.selectedGoldPos) {
        const btnCancelGold = document.createElement('button');
        btnCancelGold.className = 'btn-secondary';
        btnCancelGold.innerText = '✖ 取消选定黄金';
        btnCancelGold.onclick = () => {
          this.selectedGoldPos = null;
          this.render();
        };
        this.actionBarButtons.appendChild(btnCancelGold);

        this.guideText.innerHTML = `${lastAiHint}【已选定黄金 (${this.selectedGoldPos[0]},${this.selectedGoldPos[1]})】请点击下方发光的金字塔卡牌或牌堆直接预留，或点击取消。`;
      } else {
        this.guideText.innerHTML = `${lastAiHint}【强制行动】在棋盘上连线点选 1~3 颗非黄金宝石，或点击黄金预留卡牌，或直接点击卡牌购买/预留。`;
      }
      return;
    }

    if (phase.startsWith('CardAbilityJoker')) {
      this.guideText.innerHTML = `${lastAiHint}【变色卡定色】变色卡触发连锁能力，请指定该卡附着的宝石颜色。`;
      this.showJokerColorModal();
      return;
    }

    if (phase.startsWith('CardAbilitySameColor')) {
      this.guideText.innerHTML = `${lastAiHint}【拿取同色宝石】请在右侧 5x5 棋盘中点击一颗发光的同色宝石完成拿取。`;
      return;
    }

    if (phase === 'CardAbilitySteal') {
      this.guideText.innerHTML = `${lastAiHint}【偷取宝石】请在对手手牌区域点击任意一颗非黄金宝石进行偷取。`;
      return;
    }

    if (phase === 'SelectRoyalCard') {
      this.guideText.innerHTML = `${lastAiHint}【认领王室赞助卡】王冠达到里程碑！请在上方王室赞助池中点击一张卡牌认领。`;
      return;
    }

    if (phase === 'DiscardTokens') {
      this.guideText.innerHTML = `${lastAiHint}【手牌超限】手牌持有标记超过 10 枚，请点击自己手牌中想要弃置的标记。`;
      return;
    }
  }

  showPurchasePlanModal(card, buyActs) {
    if (!buyActs || buyActs.length === 0) return;

    this.modalTitle.innerText = '💰 卡牌购买支付方案选择';
    this.modalBody.innerText = '检测到你持有自由黄金，你可以选择默认天然支付，或消耗自由黄金替代以保留指定天然宝石：';
    this.modalOptions.innerHTML = '';

    buyActs.forEach(act => {
      const planId = act.action.PurchaseCard.plan_id;
      const desc = act.desc || '';
      const friendlyDesc = formatFriendlyAction(desc);

      const cardBox = document.createElement('div');
      cardBox.className = 'visual-plan-card';

      let planTitle = (planId === 0) ? '✨ 方案 1: 默认天然支付' : `💰 方案 ${planId + 1}: 黄金代付`;
      let planSub = (planId === 0) ? '优先使用手中天然宝石，保留全部自由黄金' : (friendlyDesc.includes('保留') ? friendlyDesc : '消耗 1 枚自由黄金以节省天然宝石');
      let badgeColor = (planId === 0) ? 'var(--text-main)' : 'var(--accent-gold)';

      cardBox.innerHTML = `
        <div class="visual-plan-header">
          <span class="visual-plan-title" style="color:${badgeColor};">${planTitle}</span>
          <span style="font-size:0.68rem; color:var(--text-muted); font-family:monospace;">plan_id: ${planId}</span>
        </div>
        <div class="visual-plan-rows">
          <div class="visual-plan-row">
            <span style="color:var(--text-muted); min-width:48px;">说明:</span>
            <span style="color:#f1f5f9; font-weight:600;">${planSub}</span>
          </div>
        </div>
      `;

      cardBox.onclick = () => {
        this.clearModal();
        this.submitAction(act.action);
      };
      this.modalOptions.appendChild(cardBox);
    });

    this.modalFooter.innerHTML = '';
    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'btn-secondary';
    cancelBtn.innerText = '取消';
    cancelBtn.onclick = () => this.clearModal();
    this.modalFooter.appendChild(cancelBtn);

    this.modalOverlay.style.display = 'flex';
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
    this.modalBody.innerText = '请选择要将该万能变色卡附着到哪种已拥有加成的宝石颜色上：';
    this.modalOptions.innerHTML = '';

    jokerActions.forEach(act => {
      const rawColor = act.action.AssignJokerColor.color;
      const normColor = String(rawColor).toLowerCase();
      const tokenClass = COLOR_CLASSES[normColor] || `token-${normColor}`;
      const cnColor = translateColor(normColor);

      const btn = document.createElement('div');
      btn.className = 'gem-picker-btn';
      btn.title = `附着为 ${cnColor}宝石 (${rawColor})`;
      btn.innerHTML = `
        <div class="token ${tokenClass}"></div>
        <span style="font-size:0.8rem; font-weight:700; color:var(--text-main); margin-top:2px;">${cnColor}宝石</span>
        <span style="font-size:0.68rem; color:var(--text-muted); text-transform:capitalize;">${rawColor}</span>
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

  async syncFullHistory() {
    try {
      const res = await fetch('/api/game/history');
      if (res.ok) {
        const data = await res.json();
        if (data.steps && data.steps.length !== this.history.length) {
          this.syncHistoryList(data.steps);
        }
      }
    } catch (e) {
      // 静默忽略
    }
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
        this.syncFullHistory();
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
          this.syncFullHistory();
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
