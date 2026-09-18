/**
 * 璀璨宝石：对决 - 交互式复盘回放控制器 (ReplayController)
 * 封装复盘分析、单步推演、AI 决策树与神经网络胜率评估走势图
 */
import {
  renderBoard,
  renderCardsList,
  renderRoyalsPool,
  renderPlayerDashboard,
  updateNeuralModelBadge
} from './shared-components.js';
import { formatFriendlyAction } from './game-controller.js';
import { soundManager } from './sound-manager.js';
import { replayService } from './game-service.js';

export class ReplayController {
  constructor() {
    this.currentStep = 0;
    this.totalSteps = 0;
    this.totalRounds = 1;
    this.isPlaying = false;
    this.playTimer = null;
    this.isFetching = false;
    this.lastSoundStepIndex = -1;

    // 客户端步骤缓存，避免重复拉取
    this.stepCache = new Map();
    this.historySteps = [];
    this.allChartStepCoords = [];
    this.hoveredChartStep = null;
    this.lastRenderedRound = 0;

    this.initDOMElements();
    this.bindEvents();
  }

  initDOMElements() {
    this.chartCanvas = document.getElementById('evalChartCanvas');
    this.stepSlider = document.getElementById('stepSlider');
    this.btnPrev = document.getElementById('btnPrev');
    this.btnNext = document.getElementById('btnNext');
    this.btnPlay = document.getElementById('btnPlay');
    this.btnPlayToEnd = document.getElementById('btnPlayToEnd');
    this.btnReset = document.getElementById('btnReset');
    this.p0TypeSelect = document.getElementById('p0TypeSelect');
    this.p1TypeSelect = document.getElementById('p1TypeSelect');
    this.playSpeedSelect = document.getElementById('playSpeed');
    this.logContainer = document.getElementById('replayLogContainer') || document.getElementById('logContainer');
  }

  bindEvents() {
    if (this.btnNext) this.btnNext.onclick = () => this.stepNext();
    if (this.btnPrev) this.btnPrev.onclick = () => this.stepPrev();
    if (this.btnPlay) this.btnPlay.onclick = () => this.togglePlay();
    if (this.btnPlayToEnd) this.btnPlayToEnd.onclick = () => this.playToEnd();
    if (this.btnReset) this.btnReset.onclick = () => this.resetGame();
    if (this.p0TypeSelect) this.p0TypeSelect.onchange = () => this.updateAiConfig();
    if (this.p1TypeSelect) this.p1TypeSelect.onchange = () => this.updateAiConfig();

    if (this.stepSlider) {
      this.stepSlider.oninput = (e) => {
        if (this.isPlaying) this.togglePlay(false);
        this.fetchStep(parseInt(e.target.value, 10));
      };
    }

    if (this.chartCanvas) {
      this.chartCanvas.onclick = (e) => {
        if (!this.historySteps || this.historySteps.length <= 1) return;
        const rect = this.chartCanvas.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const ratio = Math.max(0, Math.min(1, (clickX - 16) / (rect.width - 32)));
        const targetStep = Math.round(ratio * (this.historySteps.length - 1));
        this.fetchStep(targetStep);
      };

      this.chartCanvas.onmousemove = (e) => {
        if (!this.allChartStepCoords || this.allChartStepCoords.length <= 1) return;
        const rect = this.chartCanvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const ratio = Math.max(0, Math.min(1, (mouseX - 16) / (rect.width - 32)));
        const hoverIdx = Math.round(ratio * (this.allChartStepCoords.length - 1));
        if (hoverIdx >= 0 && hoverIdx < this.allChartStepCoords.length) {
          if (!this.hoveredChartStep || this.hoveredChartStep.idx !== hoverIdx) {
            this.hoveredChartStep = { idx: hoverIdx };
            this.renderEvaluationChart(this.historySteps, this.currentStep);
            this.updateChartHeaderBadge(this.allChartStepCoords[hoverIdx], true);
          }
        }
      };

      this.chartCanvas.onmouseleave = () => {
        if (this.hoveredChartStep !== null) {
          this.hoveredChartStep = null;
          this.renderEvaluationChart(this.historySteps, this.currentStep);
          if (this.allChartStepCoords && this.allChartStepCoords[this.currentStep]) {
            this.updateChartHeaderBadge(this.allChartStepCoords[this.currentStep], false);
          }
        }
      };
    }

    // 窗口尺寸自适应重绘走势图
    window.addEventListener('resize', () => {
      if (this.isViewMounted && this.historySteps && this.historySteps.length > 0) {
        this.renderEvaluationChart(this.historySteps, this.currentStep);
      }
    });
  }

  mountView() {
    this.isViewMounted = true;
    this.fetchStatus();
    this.syncNeuralStatus();
  }

  unmountView() {
    this.isViewMounted = false;
    if (this.isPlaying) {
      this.togglePlay(false);
    }
  }

  async loadFromReplayData(replayData) {
    if (!replayData) return;
    try {
      this.stepCache.clear();
      this.historySteps = [];
      this.lastSoundStepIndex = -1;
      this.currentStep = 0;

      await replayService.loadReplayData(replayData);
      await this.fetchStatus();
    } catch (e) {
      console.error('[ReplayController] loadFromReplayData 失败:', e);
    }
  }

  playActionSound(step) {
    if (!step) return;
    if (this.isPlaying && this.playSpeedSelect) {
      const speed = parseInt(this.playSpeedSelect.value, 10);
      if (speed < 100) return;
    }

    const desc = (typeof step === 'string' ? step : (step.action_desc || step.action || '')).toString();
    if (!desc) return;

    if (desc.startsWith('Take') || desc.includes('Take') || desc.startsWith('Steal') || desc.includes('拿取') || desc.includes('偷取')) {
      if (desc.includes('Gold') || desc.includes('gold') || desc.includes('黄金') || desc.includes('金')) {
        soundManager.play('gold_clink');
      } else {
        soundManager.play('gem_clink');
      }
    } else if (desc.startsWith('Reserve') || desc.includes('Reserve') || desc.includes('预留')) {
      if (desc.includes('gold') || desc.includes('金')) {
        soundManager.play('gold_clink');
      } else {
        soundManager.play('card_flip');
      }
    } else if (desc.startsWith('Purchase') || desc.includes('Purchase') || desc.includes('购买')) {
      soundManager.play('card_buy');
    } else if (desc.includes('Privilege') || desc.includes('特权')) {
      soundManager.play('privilege');
    } else if (desc.includes('Replenish') || desc.includes('补充')) {
      soundManager.play('replenish');
    } else if (desc.includes('Royal') || desc.includes('王室')) {
      soundManager.play('royal_claim');
    }
  }

  async fetchStatus() {
    try {
      const data = await replayService.getStatus();
      if (!data) return;

      // 更新驱动徽章
      const badge = document.getElementById('driverBadge');
      if (badge) {
        if (replayService.driverType === 'wasm') {
          badge.innerText = '⚡ 纯静态 WASM';
          badge.style.color = '#34d399';
          badge.style.borderColor = 'rgba(52,211,153,0.3)';
          badge.style.background = 'rgba(52,211,153,0.1)';
          badge.title = '运行模式：浏览器本地 WebAssembly 规则与复盘引擎';
        } else {
          badge.innerText = '🔌 服务端 REST';
          badge.style.color = '#38bdf8';
          badge.title = '运行模式：本地 Rust HTTP 服务端';
        }
      }

      // 同步双方 AI 类型下拉选择器
      if (data.player_types) {
        if (this.p0TypeSelect) this.p0TypeSelect.value = data.player_types[0] || 'heuristic';
        if (this.p1TypeSelect) this.p1TypeSelect.value = data.player_types[1] || 'heuristic';
      }

      if (data.step) {
        this.stepCache.set(data.step.step_index, data.step);
        this.currentStep = data.step.step_index;
        this.renderState(data.state, data.step);
        this.renderDecision(data.step);
      } else {
        this.renderState(data.state, null);
      }

      await this.reloadFullHistory();
    } catch (e) {
      console.error('[ReplayController] fetchStatus failed:', e);
    }
  }

  async reloadFullHistory() {
    try {
      const data = await replayService.getHistory();
      if (data && data.steps) {
        this.totalSteps = data.total_steps;
        this.totalRounds = data.total_rounds || 1;
        this.historySteps = data.steps;
        this.rebuildLogDom(this.historySteps);
        this.updateControlUI();
      }
    } catch (e) {
      console.error('[ReplayController] reloadFullHistory failed:', e);
    }
  }

  async fetchStep(index) {
    if (index < 0 || (this.totalSteps > 0 && index >= this.totalSteps)) return;
    this.currentStep = index;

    if (this.stepCache.has(index)) {
      const step = this.stepCache.get(index);
      this.renderState(step.state, step);
      this.renderDecision(step);
      this.updateControlUI();
      this.updateActiveLogInView(index);
      return;
    }

    try {
      const step = await replayService.getStep(index);
      if (step && !step.error) {
        this.stepCache.set(index, step);
        this.renderState(step.state, step);
        this.renderDecision(step);
        this.updateControlUI();
        this.updateActiveLogInView(index);
      }
    } catch (e) {
      console.error('[ReplayController] fetchStep failed:', e);
    }
  }

  async stepNext() {
    if (this.isFetching) return;

    if (this.currentStep < this.totalSteps - 1) {
      await this.fetchStep(this.currentStep + 1);
      return;
    }

    this.isFetching = true;
    try {
      const data = await replayService.stepForward(1);
      if (data && data.step) {
        const prevTotal = this.totalSteps;
        this.totalSteps = data.total_steps;
        if (data.total_rounds) this.totalRounds = data.total_rounds;
        this.currentStep = this.totalSteps - 1;

        this.stepCache.set(this.currentStep, data.step);
        this.renderState(data.step.state, data.step);
        this.renderDecision(data.step);

        if (data.new_steps && data.new_steps.length > 0) {
          data.new_steps.forEach(s => {
            this.appendLogItem(s);
            this.historySteps.push(s);
          });
          this.renderEvaluationChart(this.historySteps, this.currentStep);
          this.updateControlUI();
        } else if (this.totalSteps > prevTotal) {
          await this.reloadFullHistory();
        }

        this.updateActiveLogInView(this.currentStep);
        this.updateControlUI();

        if (!data.advanced && this.isPlaying) {
          this.togglePlay(false);
        }
      }
    } catch (e) {
      console.error('[ReplayController] stepNext failed:', e);
    } finally {
      this.isFetching = false;
    }
  }

  async stepPrev() {
    if (this.currentStep > 0) {
      await this.fetchStep(this.currentStep - 1);
    }
  }

  async playToEnd() {
    if (this.isPlaying) this.togglePlay(false);
    try {
      const data = await replayService.playToEnd();
      if (data && data.step) {
        this.totalSteps = data.total_steps;
        if (data.total_rounds) this.totalRounds = data.total_rounds;
        this.currentStep = this.totalSteps - 1;
        this.stepCache.set(this.currentStep, data.step);
        this.renderState(data.step.state, data.step);
        this.renderDecision(data.step);
        await this.reloadFullHistory();
        this.updateControlUI();
      }
    } catch (e) {
      console.error('[ReplayController] playToEnd failed:', e);
    }
  }

  async resetGame() {
    if (this.isPlaying) this.togglePlay(false);
    this.stepCache.clear();
    this.historySteps = [];
    this.lastSoundStepIndex = -1;
    this.currentStep = 0;
    this.totalSteps = 1;
    this.totalRounds = 1;
    this.rebuildLogDom([]);
    this.updateControlUI();

    const seed = Math.floor(Math.random() * 100000);
    const p0 = this.p0TypeSelect ? this.p0TypeSelect.value : 'heuristic';
    const p1 = this.p1TypeSelect ? this.p1TypeSelect.value : 'heuristic';

    try {
      const data = await replayService.reset(seed, p0, p1);
      if (data) {
        await this.fetchStatus();
      }
    } catch (e) {
      console.error('[ReplayController] resetGame failed:', e);
    }
  }

  async updateAiConfig() {
    if (!this.p0TypeSelect || !this.p1TypeSelect) return;
    const p0 = this.p0TypeSelect.value;
    const p1 = this.p1TypeSelect.value;
    try {
      await replayService.setPlayers(p0, p1);
      if (this.stepCache.has(this.currentStep)) {
        const s = this.stepCache.get(this.currentStep);
        this.renderState(s.state, s);
      }
    } catch (e) {
      console.error('[ReplayController] updateAiConfig failed:', e);
    }
  }

  updateControlUI() {
    let curRound = 1;
    if (this.stepCache.has(this.currentStep)) {
      const s = this.stepCache.get(this.currentStep);
      curRound = s.round_number || s.state?.round_number || s.state?.turn_number || 1;
    } else if (this.historySteps[this.currentStep] && this.historySteps[this.currentStep].round) {
      curRound = this.historySteps[this.currentStep].round;
    }

    const curRoundEl = document.getElementById('currentRoundText');
    const totRoundsEl = document.getElementById('totalRoundsText');
    const curStepEl = document.getElementById('currentStepText');
    const totStepsEl = document.getElementById('totalStepsText');

    if (curRoundEl) curRoundEl.innerText = curRound;
    if (totRoundsEl) totRoundsEl.innerText = this.totalRounds;
    if (curStepEl) curStepEl.innerText = this.currentStep;
    if (totStepsEl) totStepsEl.innerText = Math.max(0, this.totalSteps - 1);

    if (this.stepSlider) {
      this.stepSlider.max = Math.max(0, this.totalSteps - 1);
      this.stepSlider.value = this.currentStep;
    }
    if (this.btnPrev) this.btnPrev.disabled = (this.currentStep === 0);
    const phaseText = document.getElementById('gamePhase')?.innerText || '';
    if (this.btnNext) {
      this.btnNext.disabled = (this.totalSteps > 0 && this.currentStep >= this.totalSteps - 1 && phaseText.includes('获胜'));
    }

    this.renderEvaluationChart(this.historySteps, this.currentStep);
  }

  getStepSourceType(s) {
    if (!s || s.score === undefined || s.score === null) return 'human';
    const aiType = (s.ai_type || '').toLowerCase();
    if (aiType.includes('neural') && !aiType.includes('fallback')) {
      return 'neural';
    }
    if (aiType.includes('heuristic') || aiType.includes('fallback')) {
      return 'heuristic';
    }
    if (aiType === 'random') {
      return 'random';
    }
    return 'human';
  }

  renderEvaluationChart(steps, currentIdx) {
    const canvas = this.chartCanvas || document.getElementById('evalChartCanvas');
    if (!canvas || !steps || steps.length === 0) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = rect.width || 600;
    const h = rect.height || 85;

    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const total = steps.length;
    if (total <= 1) {
      this.allChartStepCoords = [];
      const badge = document.getElementById('chartCurrentStepBadge');
      if (badge) badge.innerHTML = `当前步: <b style="color:var(--accent-gold);">#0</b> <span style="color:#64748b;">(对局初始)</span>`;
      return;
    }

    // 50% 均势基准线
    const midY = h / 2;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(w, midY);
    ctx.stroke();
    ctx.setLineDash([]);

    // 刻度文本
    ctx.fillStyle = 'rgba(255, 255, 255, 0.22)';
    ctx.font = '9px system-ui, sans-serif';
    ctx.fillText('50%', 4, midY - 3);

    const p0NeuralPoints = [];
    const p1NeuralPoints = [];
    this.allChartStepCoords = [];

    steps.forEach((s, idx) => {
      const x = (idx / Math.max(1, total - 1)) * (w - 32) + 16;
      const sourceType = this.getStepSourceType(s);
      let winrate = null;
      let y = null;

      if (sourceType === 'neural' && s.score !== undefined && s.score !== null) {
        const rawVal = Number(s.score);
        const winratePct = rawVal < 0 ? (rawVal + 100.0) / 2.0 : rawVal;
        winrate = Math.max(0, Math.min(100, winratePct));
        y = h - 8 - ((winrate / 100.0) * (h - 16));
      }

      const ptInfo = {
        idx,
        x,
        y,
        winrate,
        rawScore: s.score,
        sourceType,
        player: s.player,
        step: s,
      };

      this.allChartStepCoords.push(ptInfo);

      if (winrate !== null) {
        if (s.player === 0) {
          p0NeuralPoints.push(ptInfo);
        } else if (s.player === 1) {
          p1NeuralPoints.push(ptInfo);
        }
      }
    });

    const hasP0Neural = p0NeuralPoints.length > 0;
    const hasP1Neural = p1NeuralPoints.length > 0;

    const legendLinesEl = document.getElementById('chartLegendLines');
    if (legendLinesEl) {
      if (hasP0Neural && hasP1Neural) {
        legendLinesEl.innerHTML = `<span style="color:#38bdf8; font-weight:700;">■ P0 神经网络</span> · <span style="color:#f472b6; font-weight:700;">■ P1 神经网络</span>`;
      } else if (hasP0Neural && !hasP1Neural) {
        legendLinesEl.innerHTML = `<span style="color:#38bdf8; font-weight:700;">■ P0 神经网络</span> · <span style="color:#64748b;">P1 无胜率模型 (启发式/人类)</span>`;
      } else if (!hasP0Neural && hasP1Neural) {
        legendLinesEl.innerHTML = `<span style="color:#64748b;">P0 无胜率模型 (人类/启发式)</span> · <span style="color:#f472b6; font-weight:700;">■ P1 神经网络</span>`;
      } else {
        legendLinesEl.innerHTML = `<span style="color:#64748b;">本局无神经网络评估 (启发式与人类不输出胜率)</span>`;
      }
    }

    if (!hasP0Neural && !hasP1Neural) {
      ctx.fillStyle = 'rgba(148, 163, 184, 0.45)';
      ctx.font = '11px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('当前对局无神经网络评估数据（启发式 AI 与人类玩家不输出胜率模型）', w / 2, midY + 4);
      ctx.textAlign = 'start';
    }

    // 1. P0 胜率曲线 (蓝 #38bdf8)
    if (hasP0Neural) {
      if (p0NeuralPoints.length >= 2) {
        ctx.beginPath();
        ctx.moveTo(p0NeuralPoints[0].x, p0NeuralPoints[0].y);
        for (let i = 1; i < p0NeuralPoints.length; i++) {
          ctx.lineTo(p0NeuralPoints[i].x, p0NeuralPoints[i].y);
        }
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2.2;
        ctx.shadowColor = 'rgba(56, 189, 248, 0.4)';
        ctx.shadowBlur = 5;
        ctx.stroke();
        ctx.shadowBlur = 0;
      }

      p0NeuralPoints.forEach(pt => {
        if (pt.idx === currentIdx || (this.hoveredChartStep && this.hoveredChartStep.idx === pt.idx)) return;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 2.8, 0, Math.PI * 2);
        ctx.fillStyle = '#38bdf8';
        ctx.fill();
      });
    }

    // 2. P1 胜率曲线 (粉 #f472b6)
    if (hasP1Neural) {
      if (p1NeuralPoints.length >= 2) {
        ctx.beginPath();
        ctx.moveTo(p1NeuralPoints[0].x, p1NeuralPoints[0].y);
        for (let i = 1; i < p1NeuralPoints.length; i++) {
          ctx.lineTo(p1NeuralPoints[i].x, p1NeuralPoints[i].y);
        }
        ctx.strokeStyle = '#f472b6';
        ctx.lineWidth = 2.2;
        ctx.shadowColor = 'rgba(244, 114, 182, 0.4)';
        ctx.shadowBlur = 5;
        ctx.stroke();
        ctx.shadowBlur = 0;
      }

      p1NeuralPoints.forEach(pt => {
        if (pt.idx === currentIdx || (this.hoveredChartStep && this.hoveredChartStep.idx === pt.idx)) return;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 2.8, 0, Math.PI * 2);
        ctx.fillStyle = '#f472b6';
        ctx.fill();
      });
    }

    // 3. 悬停游标
    if (this.hoveredChartStep && this.hoveredChartStep.idx >= 0 && this.hoveredChartStep.idx < this.allChartStepCoords.length && this.hoveredChartStep.idx !== currentIdx) {
      const hovPt = this.allChartStepCoords[this.hoveredChartStep.idx];
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(hovPt.x, 0);
      ctx.lineTo(hovPt.x, h);
      ctx.stroke();
      ctx.setLineDash([]);

      if (hovPt.y !== null) {
        const hovCol = (hovPt.player === 0) ? '#38bdf8' : '#f472b6';
        ctx.beginPath();
        ctx.arc(hovPt.x, hovPt.y, 4.5, 0, Math.PI * 2);
        ctx.fillStyle = hovCol;
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.fill();
        ctx.stroke();
      }
    }

    // 4. 当前步游标
    if (currentIdx >= 0 && currentIdx < this.allChartStepCoords.length) {
      const curPt = this.allChartStepCoords[currentIdx];
      ctx.strokeStyle = 'rgba(251, 191, 36, 0.9)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(curPt.x, 0);
      ctx.lineTo(curPt.x, h);
      ctx.stroke();
      ctx.setLineDash([]);

      if (curPt.y !== null) {
        const curCol = (curPt.player === 0) ? '#38bdf8' : '#f472b6';
        ctx.beginPath();
        ctx.arc(curPt.x, curPt.y, 5.2, 0, Math.PI * 2);
        ctx.fillStyle = '#fbbf24';
        ctx.shadowColor = curCol;
        ctx.shadowBlur = 8;
        ctx.fill();
        ctx.shadowBlur = 0;
      } else {
        ctx.beginPath();
        ctx.arc(curPt.x, midY, 3.8, 0, Math.PI * 2);
        ctx.fillStyle = '#0b0f17';
        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 1.5;
        ctx.fill();
        ctx.stroke();
      }

      if (!this.hoveredChartStep) {
        this.updateChartHeaderBadge(curPt, false);
      }
    }
  }

  updateChartHeaderBadge(pt, isHover = false) {
    const badge = document.getElementById('chartCurrentStepBadge');
    if (!badge || !pt) return;

    const pCol = (pt.player === 0) ? '#38bdf8' : '#f472b6';
    let infoHtml = '';

    if (pt.sourceType === 'neural' && pt.winrate !== null) {
      infoHtml = `<span style="color:#10b981; font-weight:700;">[🧠 神经网络]</span> <span style="color:${pCol}; font-weight:800;">胜率预期: ${pt.winrate.toFixed(0)}%</span>`;
    } else if (pt.sourceType === 'heuristic') {
      const sc = pt.rawScore;
      const scTxt = (sc !== undefined && sc !== null)
        ? (sc >= 1000 ? '+9999 斩杀' : (sc >= 0 ? `+${sc.toFixed(1)}分` : `${sc.toFixed(1)}分`))
        : '无估值';
      infoHtml = `<span style="color:#818cf8; font-weight:700;">[🤖 启发式 AI]</span> <span style="color:#cbd5e1;">动作评分: <b style="color:#f1f5f9;">${scTxt}</b></span>`;
    } else {
      infoHtml = `<span style="color:#94a3b8;">[👤 人类决策]</span> <span style="color:#64748b;">(无模型胜率)</span>`;
    }

    const prefix = isHover ? '👉 悬停' : '当前步';
    const actDesc = pt.step ? formatFriendlyAction(pt.step.action) : '';
    const actHtml = actDesc ? ` · <span style="color:#e2e8f0; font-weight:normal;">${actDesc}</span>` : '';

    badge.innerHTML = `
      ${prefix}: <b style="color:#fbbf24;">#${pt.idx}</b> (<b style="color:${pCol};">P${pt.player}</b>)
      ${infoHtml}${actHtml}
    `;
  }

  rebuildLogDom(steps) {
    const container = this.logContainer || document.getElementById('replayLogContainer') || document.getElementById('logContainer');
    if (!container) return;
    container.innerHTML = '';
    this.lastRenderedRound = 0;
    steps.forEach(s => this.appendLogItem(s));
    this.updateActiveLogInView(this.currentStep);
  }

  appendLogItem(s) {
    const container = this.logContainer || document.getElementById('replayLogContainer') || document.getElementById('logContainer');
    if (!container || container.querySelector(`[data-index="${s.index}"]`)) return;

    const round = s.round || 1;
    if (round > this.lastRenderedRound) {
      this.lastRenderedRound = round;
      const divider = document.createElement('div');
      divider.className = 'log-round-divider';
      divider.innerHTML = `<span>⏳ 第 ${round} 轮 (Round ${round})</span>`;
      container.appendChild(divider);
    }

    const item = document.createElement('div');
    item.className = 'log-item';
    item.setAttribute('data-index', s.index);
    const scoreHint = (s.score !== undefined && s.score !== null) ? ` [估值: ${s.score.toFixed(1)}]` : '';
    item.title = `第${round}轮 #${s.index} [P${s.player}] ${s.action} (${s.phase})${scoreHint}`;

    const friendlyAct = formatFriendlyAction(s.action);
    item.innerHTML = `
      <span class="log-idx">#${s.index}</span>
      <span class="log-p" style="color:${s.player === 0 ? '#38bdf8' : '#f472b6'};">P${s.player}</span>
      <span class="log-act" title="${friendlyAct}">${friendlyAct}</span>
      ${(s.score !== undefined && s.score !== null) ? `<span class="log-score-tag">${s.score.toFixed(0)}</span>` : ''}
    `;
    item.onclick = () => this.fetchStep(s.index);
    container.appendChild(item);
  }

  updateActiveLogInView(index) {
    const container = this.logContainer || document.getElementById('replayLogContainer') || document.getElementById('logContainer');
    if (!container) return;
    const prev = container.querySelector('.active-step');
    if (prev) prev.classList.remove('active-step');

    const target = container.querySelector(`[data-index="${index}"]`);
    if (target) {
      target.classList.add('active-step');
      const itemTop = target.offsetTop - container.offsetTop;
      const itemHeight = target.offsetHeight;
      const containerHeight = container.clientHeight;
      const currentScroll = container.scrollTop;

      if (itemTop < currentScroll || (itemTop + itemHeight) > (currentScroll + containerHeight)) {
        container.scrollTop = itemTop - Math.floor((containerHeight - itemHeight) / 2);
      }
    }
  }

  renderDecision(step) {
    if (!step) return;
    const stepIdxEl = document.getElementById('decisionStepIndex');
    if (stepIdxEl) stepIdxEl.innerText = `#${step.step_index}`;

    const pEl = document.getElementById('decisionPlayer');
    if (pEl) {
      pEl.innerText = `Player ${step.player}`;
      pEl.style.color = (step.player === 0 ? '#38bdf8' : '#f472b6');
    }

    const actEl = document.getElementById('decisionAction');
    if (actEl) {
      actEl.innerText = step.action_desc;
      actEl.title = step.action_desc;
    }

    const scoreEl = document.getElementById('decisionScore');
    const scoreLabelEl = document.getElementById('decisionScoreLabel');
    const badgeEl = document.getElementById('decisionAiTypeBadge');
    const candidatesList = document.getElementById('candidatesList');
    const rankTitle = document.getElementById('candidatesRankingTitle');
    const rankSub = document.getElementById('candidatesRankingSubtitle');
    if (!candidatesList) return;
    candidatesList.innerHTML = '';

    const decision = step.decision;
    if (decision) {
      const isNeural = decision.ai_type.includes('neural');
      const isRandom = decision.ai_type === 'random';

      if (isNeural) {
        if (badgeEl) {
          badgeEl.innerText = `🧠 神经网络 AI (${decision.ai_type})`;
          badgeEl.style.borderColor = '#22c55e';
          badgeEl.style.backgroundColor = 'rgba(34, 197, 94, 0.15)';
        }
        if (scoreLabelEl) scoreLabelEl.innerText = '胜率估值:';
        if (rankTitle) rankTitle.innerText = '神经网络策略概率分布 P(a|s)';
        if (rankSub) rankSub.innerText = 'Policy Head 输出的高置信动作';
      } else if (isRandom) {
        if (badgeEl) {
          badgeEl.innerText = '🎲 随机 AI';
          badgeEl.style.borderColor = '#94a3b8';
          badgeEl.style.backgroundColor = 'rgba(148, 163, 184, 0.15)';
        }
        if (scoreLabelEl) scoreLabelEl.innerText = '随机评分:';
        if (rankTitle) rankTitle.innerText = '随机合法动作抽取 (Uniform Random)';
        if (rankSub) rankSub.innerText = '无策略权重，等概率随机';
      } else {
        if (badgeEl) {
          badgeEl.innerText = '🤖 启发式 AI';
          badgeEl.style.borderColor = '#38bdf8';
          badgeEl.style.backgroundColor = 'rgba(56, 189, 248, 0.15)';
        }
        if (scoreLabelEl) scoreLabelEl.innerText = '启发式得分:';
        if (rankTitle) rankTitle.innerText = '候选合法动作打分排名 (Top Candidates)';
        if (rankSub) rankSub.innerText = '启发式估值越高越优先';
      }

      if (decision.chosen_score !== null && decision.chosen_score !== undefined) {
        const sc = Number(decision.chosen_score);
        if (scoreEl) {
          if (isNeural) {
            const winrateProb = sc < 0 ? (sc + 100.0) / 2.0 : sc;
            const clamped = Math.max(0, Math.min(100, winrateProb));
            scoreEl.innerText = `${clamped.toFixed(1)}% 胜率`;
            scoreEl.style.color = clamped >= 50 ? '#22c55e' : '#f87171';
          } else {
            scoreEl.innerText = sc >= 1000 ? `+${sc.toFixed(0)} (斩杀优先)` : `+${sc.toFixed(1)}`;
            scoreEl.style.color = '#22c55e';
          }
          scoreEl.style.display = 'inline-block';
        }
      } else if (scoreEl) {
        scoreEl.innerText = '无打分';
        scoreEl.style.display = 'inline-block';
      }

      if (decision.top_candidates && decision.top_candidates.length > 0) {
        decision.top_candidates.forEach((cand, idx) => {
          const row = document.createElement('div');
          row.className = `candidate-row ${cand.is_chosen ? 'chosen' : ''}`;
          let scoreStr = '';
          if (isNeural) {
            scoreStr = `${cand.score.toFixed(1)}%`;
          } else {
            const isKill = cand.score >= 1000;
            scoreStr = isKill ? `+${cand.score.toFixed(0)} 斩杀` : (cand.score >= 0 ? `+${cand.score.toFixed(1)}` : `${cand.score.toFixed(1)}`);
          }

          const chosenBadge = isNeural ? '<span class="candidate-chosen-badge">★ 采纳</span>' : '<span class="candidate-chosen-badge">★ 选定</span>';

          row.innerHTML = `
            <span style="color:#64748b; font-size:0.68rem; min-width:16px;">${idx + 1}.</span>
            <span class="candidate-act" title="${cand.action_desc}">${cand.action_desc}</span>
            ${cand.is_chosen ? chosenBadge : ''}
            <span class="candidate-score" style="color:${cand.is_chosen ? '#22c55e' : '#38bdf8'};">${scoreStr}</span>
          `;
          candidatesList.appendChild(row);
        });
      } else {
        candidatesList.innerHTML = '<span style="font-size:0.7rem; color:var(--text-muted); padding:4px;">当前步骤由随机 AI 执行，无动作估值排行</span>';
      }
    } else {
      if (badgeEl) {
        badgeEl.innerText = '对局初始化';
        badgeEl.style.borderColor = '#64748b';
        badgeEl.style.backgroundColor = 'rgba(100, 116, 139, 0.15)';
      }
      if (scoreEl) scoreEl.innerText = '-';
      candidatesList.innerHTML = '<span style="font-size:0.7rem; color:var(--text-muted); padding:4px;">初始棋盘就绪，先手玩家准备行动</span>';
    }
  }

  renderState(state, step) {
    if (!state) return;
    const actor = (step && step.step_index > 0) ? step.player : state.current_player;
    const nextPlayer = state.current_player;

    const phaseEl = document.getElementById('gamePhase');
    const winnerBanner = document.getElementById('winnerBanner');
    const phaseBar = document.getElementById('phaseBar');
    const phaseBarText = document.getElementById('phaseBarText');
    const phaseBarSub = document.getElementById('phaseBarSub');

    const curRound = state.round_number || state.turn_number;
    if (state.winner) {
      if (phaseEl) phaseEl.innerText = `🏆 获胜者: ${state.winner}`;
      if (winnerBanner) {
        winnerBanner.style.display = 'flex';
        const finalRound = curRound;
        const finalStep = step ? step.step_index : Math.max(0, this.totalSteps - 1);
        winnerBanner.innerHTML = `<span>🏆 <b>对局结束</b> — 获胜者: <span style="color:#fbbf24; font-size:1.06rem; margin-left:4px;">${state.winner}</span> <span style="font-size:0.8rem; color:#94a3b8; margin-left:14px;">(耗时: 共 ${finalRound} 轮 / ${finalStep} 步)</span></span>`;
      }
      if (phaseBar) phaseBar.style.display = 'none';
    } else {
      if (winnerBanner) winnerBanner.style.display = 'none';
      if (phaseBar) phaseBar.style.display = 'flex';

      if (step && step.step_index > 0) {
        const turnTip = (actor !== nextPlayer) ? `(下次回合由 P${nextPlayer} 接手)` : `(持续行动)`;
        if (phaseEl) phaseEl.innerText = `第 ${curRound} 轮 | P${actor} 执行动作 | 阶段: ${state.phase}`;
        if (phaseBarText) phaseBarText.innerText = `第 ${curRound} 轮 | P${actor} 执行动作 | 阶段: ${state.phase}`;
        if (phaseBarSub) phaseBarSub.innerText = turnTip;
      } else {
        if (phaseEl) phaseEl.innerText = `第 ${curRound} 轮 | 先手行动: P${state.current_player} | 阶段: ${state.phase}`;
        if (phaseBarText) phaseBarText.innerText = `第 ${curRound} 轮 | 先手行动: P${state.current_player} | 阶段: ${state.phase}`;
        if (phaseBarSub) phaseBarSub.innerText = '初始准备就绪';
      }
    }

    // 幽灵残留
    const ghostTokens = [];
    if (step && step.action_desc) {
      const matches = step.action_desc.matchAll(/\((\d+),\s*(\d+)\)/g);
      for (const m of matches) {
        ghostTokens.push({
          r: parseInt(m[1], 10),
          c: parseInt(m[2], 10),
          gem: 'blue',
          label: '取走'
        });
      }
    }

    const boardEl = document.getElementById('boardGrid');
    if (boardEl) {
      const tokenCount = renderBoard(state.board, boardEl, { ghostTokens });
      const boardTokensCountEl = document.getElementById('boardTokensCount');
      if (boardTokensCountEl) boardTokensCountEl.innerText = `${tokenCount} 标记`;
    }

    const privEl = document.getElementById('privilegePool');
    if (privEl) privEl.innerText = state.privilege_pool;
    const bagEl = document.getElementById('bagCount');
    if (bagEl) bagEl.innerText = state.bag_count;

    renderRoyalsPool(state.royal_cards, document.getElementById('royalsContainer'));

    const curP = state.players[actor] || state.players[0];
    renderCardsList(state.pyramid[2], document.getElementById('tier3Cards'), {
      deckInfo: { tier: 3, count: state.decks_count[2] },
      currentPlayerState: curP
    });
    renderCardsList(state.pyramid[1], document.getElementById('tier2Cards'), {
      deckInfo: { tier: 2, count: state.decks_count[1] },
      currentPlayerState: curP
    });
    renderCardsList(state.pyramid[0], document.getElementById('tier1Cards'), {
      deckInfo: { tier: 1, count: state.decks_count[0] },
      currentPlayerState: curP
    });

    const d3 = document.getElementById('deck3Count');
    if (d3) d3.innerText = state.decks_count[2];
    const d2 = document.getElementById('deck2Count');
    if (d2) d2.innerText = state.decks_count[1];
    const d1 = document.getElementById('deck1Count');
    if (d1) d1.innerText = state.decks_count[0];

    const p0Acting = (actor === 0);
    const p1Acting = (actor === 1);
    const p0Next = (step && step.step_index > 0 && actor !== nextPlayer && nextPlayer === 0);
    const p1Next = (step && step.step_index > 0 && actor !== nextPlayer && nextPlayer === 1);

    const p0Type = this.p0TypeSelect ? this.p0TypeSelect.value : 'heuristic';
    const p1Type = this.p1TypeSelect ? this.p1TypeSelect.value : 'heuristic';

    renderPlayerDashboard(state.players[0], document.getElementById('player0Card'), p0Acting, p0Next, 'p0', { playerKind: p0Type });
    renderPlayerDashboard(state.players[1], document.getElementById('player1Card'), p1Acting, p1Next, 'p1', { playerKind: p1Type });

    if (step && step.step_index !== this.lastSoundStepIndex && step.step_index > 0) {
      this.lastSoundStepIndex = step.step_index;
      if (state.winner) {
        soundManager.play('victory');
      } else {
        this.playActionSound(step);
      }
    }
  }

  togglePlay(start) {
    this.isPlaying = (start !== undefined) ? start : !this.isPlaying;
    if (this.btnPlay) {
      this.btnPlay.innerText = this.isPlaying ? '暂停 ⏸' : '自动播放 ⏯';
      this.btnPlay.style.backgroundColor = this.isPlaying ? '#dc2626' : '';
    }
    if (this.isPlaying) {
      this.scheduleNextTick();
    } else {
      if (this.playTimer) clearTimeout(this.playTimer);
    }
  }

  scheduleNextTick() {
    if (!this.isPlaying) return;
    const speed = parseInt(this.playSpeedSelect?.value, 10) || 120;
    this.playTimer = setTimeout(async () => {
      if (!this.isPlaying) return;
      if (this.currentStep < this.totalSteps - 1) {
        await this.fetchStep(this.currentStep + 1);
        this.scheduleNextTick();
      } else {
        await this.stepNext();
        if (this.isPlaying) this.scheduleNextTick();
      }
    }, speed);
  }

  async syncNeuralStatus() {
    try {
      const data = await replayService.getNeuralStatus();
      if (data) {
        const dot = document.getElementById('neuralStatusDot');
        const badge = document.getElementById('neuralModelBadge');
        const text = document.getElementById('neuralModelText');
        if (dot) {
          if (data.available) {
            dot.style.color = '#22c55e';
            dot.title = data.model_type === 'onnx-web'
              ? '浏览器本地 ONNX 神经网络引擎已就绪 (WebAssembly 加速)'
              : '神经网络推理微服务在线';
          } else if (data.loading) {
            dot.style.color = '#eab308';
            dot.title = '神经网络模型正在加载中...';
          } else {
            dot.style.color = '#94a3b8';
            dot.title = '神经网络推理微服务未连接 (当前复盘使用启发式 AI)';
          }
        }
        updateNeuralModelBadge(badge, text, data);
      }
    } catch(e) {}
  }
}
