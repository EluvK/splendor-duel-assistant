/**
 * Splendor Duel Game Service
 * 璀璨宝石：对决 统一对战服务抽象层
 * 
 * 支持双模驱动：
 * 1. WasmGameDriver: 纯静态运行模式 (GitHub Pages / 离线)，基于 WebAssembly + ONNX Runtime Web，零后端依赖
 * 2. HttpGameDriver: 本地服务端运行模式 (开发/训练调试)，通过 REST JSON API 与 Rust 后端通信
 */

import initWasm, { WasmGameSession, WasmReplaySession } from '../pkg/splendor_duel_wasm.js';

/**
 * 浏览器端 ONNX 神经网络推理器
 */
class OnnxPredictor {
  constructor() {
    this.session = null;
    this.loading = false;
    this.ready = false;
    this.error = null;
    this.modelPath = 'assets/models/best.onnx';
    this.metadata = null;
  }

  async loadModel() {
    if (this.ready || this.loading) return this.ready;
    if (typeof window.ort === 'undefined') {
      this.error = 'ONNX Runtime Web 未加载';
      console.warn('[OnnxPredictor]', this.error);
      return false;
    }

    this.loading = true;
    try {
      console.log('[OnnxPredictor] 开始加载神经网络模型:', this.modelPath);
      // 配置 ONNX Web 运行时选项 (优先使用 WASM 执行器)
      window.ort.env.wasm.numThreads = 1;
      this.session = await window.ort.InferenceSession.create(this.modelPath, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      this.ready = true;
      this.error = null;
      console.log('[OnnxPredictor] 神经网络模型加载成功！');

      // 异步尝试获取模型元数据与部署版本清单
      this.fetchMetadata().catch(() => {});

      return true;
    } catch (err) {
      this.error = err.message || String(err);
      console.warn('[OnnxPredictor] 加载模型失败，将回退至启发式 AI:', err);
      return false;
    } finally {
      this.loading = false;
    }
  }

  async fetchMetadata() {
    try {
      const [metaRes, verRes] = await Promise.allSettled([
        fetch('assets/models/best.json'),
        fetch('assets/version.json'),
      ]);
      const meta = (metaRes.status === 'fulfilled' && metaRes.value.ok) ? await metaRes.value.json() : {};
      const ver = (verRes.status === 'fulfilled' && verRes.value.ok) ? await verRes.value.json() : {};
      this.metadata = { ...ver, ...meta };
    } catch (e) {
      this.metadata = null;
    }
  }

  /**
   * 执行前向推理
   * @param {Float32Array} obs 969维观测向量
   * @param {Array<number>} legalActionIds 合法动作ID集合
   * @param {Array<Object>} legalActionDtos 合法动作展示DTO集合
   * @returns {Object} { bestActionId, winrate, topCandidates }
   */
  async predict(obs, legalActionIds, legalActionDtos = []) {
    if (!this.ready) {
      const ok = await this.loadModel();
      if (!ok) return null;
    }

    try {
      const inputTensor = new window.ort.Tensor('float32', obs, [1, obs.length]);
      const feeds = { obs: inputTensor };
      const output = await this.session.run(feeds);

      const policyLogits = output.policy_logits.data; // Float32Array [1856]
      const winValue = output.win_value ? output.win_value.data[0] : 0.0; // [-1.0, 1.0]
      const winrate = Math.max(0.0, Math.min(1.0, (winValue + 1.0) / 2.0));

      if (!legalActionIds || legalActionIds.length === 0) {
        return null;
      }

      // 对合法动作计算 Masked Softmax
      let maxLogit = -Infinity;
      for (const id of legalActionIds) {
        if (id < policyLogits.length) {
          const l = policyLogits[id];
          if (l > maxLogit) maxLogit = l;
        }
      }

      let expSum = 0.0;
      const candidates = [];
      for (let i = 0; i < legalActionIds.length; i++) {
        const id = legalActionIds[i];
        const logit = id < policyLogits.length ? policyLogits[id] : -100.0;
        const expVal = Math.exp(logit - maxLogit);
        expSum += expVal;
        const dto = legalActionDtos[i];
        candidates.push({
          action_id: id,
          expVal,
          desc: dto ? dto.desc : `Action #${id}`,
        });
      }

      // 计算概率并排序
      for (const c of candidates) {
        c.prob = expSum > 0 ? c.expVal / expSum : 1.0 / candidates.length;
      }
      candidates.sort((a, b) => b.prob - a.prob);

      const bestActionId = candidates[0].action_id;
      const topCandidates = candidates.slice(0, 8).map(c => ({
        action_desc: c.desc,
        score: c.prob * 100.0,
        is_chosen: c.action_id === bestActionId,
      }));

      return {
        bestActionId,
        winrate,
        topCandidates,
      };
    } catch (e) {
      console.error('[OnnxPredictor] 推理计算异常:', e);
      return null;
    }
  }
}

/**
 * WASM 纯前端驱动器 (GitHub Pages 模式)
 */
class WasmGameDriver {
  constructor() {
    this.session = null;
    this.wasmInitialized = false;
    this.predictor = new OnnxPredictor();
    this.seed = 42;
    this.playerKinds = ['human', 'neural'];
    this.mctsSims = 0;
  }

  async init() {
    if (!this.wasmInitialized) {
      await initWasm();
      this.wasmInitialized = true;
      console.log('[WasmGameDriver] Rust WASM 核心规则引擎加载成功！');
    }
    // 异步后台拉取 ONNX 模型
    this.predictor.loadModel().then(ready => {
      if (this.session) {
        this.session.set_neural_ready(ready);
      }
    });
  }

  ensureSession() {
    if (!this.session) {
      // 尝试从 sessionStorage 恢复已有对战局势（避免切换复盘页面后丢失对局进度）
      const saved = sessionStorage.getItem('splendor_duel_play_session');
      if (saved) {
        try {
          this.session = WasmGameSession.from_saved_state(saved);
          this.session.set_mcts_simulations(this.mctsSims);
          this.session.set_neural_ready(this.predictor.ready);

          // 依据恢复的会话同步 driver 内部参数，保证先后手与配置一致
          try {
            const raw = this.session.get_state_json();
            const data = JSON.parse(raw);
            if (data && Array.isArray(data.player_kinds) && data.player_kinds.length === 2) {
              this.playerKinds = data.player_kinds;
            }
            if (data && data.state && data.state.rng_seed !== undefined) {
              this.seed = data.state.rng_seed;
            }
          } catch (err) {}

          console.log('[WasmGameDriver] 成功从会话缓存恢复已有对战局势！');
          return;
        } catch (e) {
          console.error('[WasmGameDriver] 恢复对战局势异常:', e);
        }
      }
      this.session = new WasmGameSession(BigInt(this.seed), this.playerKinds[0], this.playerKinds[1]);
      this.session.set_mcts_simulations(this.mctsSims);
      this.session.set_neural_ready(this.predictor.ready);
      this.persistSession();
    }
  }

  persistSession() {
    if (this.session) {
      try {
        const exported = this.session.export_saved_state();
        if (exported && exported.length > 0) {
          sessionStorage.setItem('splendor_duel_play_session', exported);
        }
      } catch (e) {
        console.warn('[WasmGameDriver] persistSession 异常:', e);
      }
    }
  }

  async getState() {
    await this.init();
    this.ensureSession();
    const raw = this.session.get_state_json();
    const data = JSON.parse(raw);
    data.neural_available = this.predictor.ready;
    data.driver_type = 'wasm';
    return data;
  }

  async newGame(seed, p0, p1, sims) {
    await this.init();
    this.seed = seed || 42;
    this.playerKinds = [p0 || 'human', p1 || 'neural'];
    this.mctsSims = Number(sims) || 0;
    this.session = new WasmGameSession(BigInt(this.seed), this.playerKinds[0], this.playerKinds[1]);
    this.session.set_mcts_simulations(this.mctsSims);
    this.session.set_neural_ready(this.predictor.ready);
    this.persistSession();
    const data = JSON.parse(this.session.get_state_json());
    data.neural_available = this.predictor.ready;
    data.driver_type = 'wasm';
    return data;
  }

  async stepHuman(actionPayload) {
    await this.init();
    this.ensureSession();
    const raw = this.session.step_human(JSON.stringify(actionPayload));
    const data = JSON.parse(raw);
    data.neural_available = this.predictor.ready;
    if (data.ok) {
      this.persistSession();
    }
    return data;
  }

  async stepAi(sims) {
    await this.init();
    this.ensureSession();
    const currentSims = sims !== undefined ? Number(sims) : this.mctsSims;

    const state = await this.getState();
    const currentKind = state.current_player_kind;

    if (currentKind === 'neural') {
      // 检查 ONNX 模型是否可用
      if (this.predictor.ready) {
        const obsArray = this.session.encode_observation();
        const legalIds = Array.from(this.session.get_legal_action_ids());
        const legalsDto = state.legal_actions;

        const pred = await this.predictor.predict(new Float32Array(obsArray), legalIds, legalsDto);
        if (pred) {
          const raw = this.session.apply_neural_step(
            pred.bestActionId,
            pred.winrate,
            JSON.stringify(pred.topCandidates)
          );
          const result = JSON.parse(raw);
          if (result.ok) {
            this.persistSession();
          }
          return result;
        }
      }
      // 若 ONNX 未就绪或推理失败，平滑降级至启发式 AI
      console.warn('[WasmGameDriver] 神经网络尚未就绪，使用启发式 AI 替代落子');
    }

    // Heuristic 或 Random 直接由 Rust WASM 极速推演
    const raw = this.session.step_ai(currentSims > 0 ? currentSims : undefined);
    const result = JSON.parse(raw);
    if (result.ok) {
      this.persistSession();
    }
    return result;
  }

  async getStep(index) {
    await this.init();
    this.ensureSession();
    const raw = this.session.get_step_json(index);
    return JSON.parse(raw);
  }

  async getHistory() {
    await this.init();
    this.ensureSession();
    const raw = this.session.get_history_json();
    return JSON.parse(raw);
  }

  async exportReplayData() {
    await this.init();
    this.ensureSession();
    const raw = this.session.export_replay_data();
    return JSON.parse(raw);
  }

  async getNeuralStatus() {
    return {
      available: this.predictor.ready,
      loading: this.predictor.loading,
      error: this.predictor.error,
      model_type: 'onnx-web',
      details: this.predictor.metadata,
    };
  }

  async reloadNeural() {
    this.predictor.ready = false;
    const ok = await this.predictor.loadModel();
    if (this.session) {
      this.session.set_neural_ready(ok);
    }
    return { ok };
  }

  setMctsSimulations(sims) {
    this.mctsSims = Number(sims) || 0;
    if (this.session) {
      this.session.set_mcts_simulations(this.mctsSims);
    }
  }
}

/**
 * HTTP REST 驱动器 (本地开发模式)
 */
class HttpGameDriver {
  async getState() {
    const res = await fetch('/api/game/state');
    const data = await res.json();
    data.driver_type = 'http';
    return data;
  }

  async newGame(seed, p0, p1, sims) {
    const res = await fetch(`/api/game/new?seed=${seed}&p0=${p0}&p1=${p1}&sims=${sims}`, {
      method: 'POST',
    });
    const data = await res.json();
    data.driver_type = 'http';
    return data;
  }

  async stepHuman(actionPayload) {
    const res = await fetch('/api/game/action', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(actionPayload),
    });
    return await res.json();
  }

  async stepAi(sims) {
    const res = await fetch(`/api/game/ai_step?sims=${sims}`, { method: 'POST' });
    return await res.json();
  }

  async getStep(index) {
    const res = await fetch(`/api/game/step?index=${index}`);
    return await res.json();
  }

  async getHistory() {
    const res = await fetch('/api/game/history');
    return await res.json();
  }

  async getNeuralStatus() {
    const res = await fetch('/api/neural_status');
    return await res.json();
  }

  async reloadNeural() {
    const res = await fetch('/api/neural_reload', { method: 'POST' });
    return await res.json();
  }

  async setMctsSimulations(sims) {
    try {
      const res = await fetch(`/api/game/set_sims?sims=${sims}`, { method: 'POST' });
      return await res.json();
    } catch (e) {
      return null;
    }
  }
}

/**
 * 统一门面服务
 */
class GameService {
  constructor() {
    this.driver = null;
    this.driverType = 'unknown';
    this.readyPromise = this.detectAndInitDriver();
  }

  async detectAndInitDriver() {
    // 1. 如果在 GitHub Pages 环境下 (包含 github.io 或指定了 wasm 参数)，直接使用 WASM
    const isGithubPages = window.location.hostname.includes('github.io');
    const forceWasm = new URLSearchParams(window.location.search).has('wasm');
    const forceHttp = new URLSearchParams(window.location.search).has('http');

    if (forceWasm || isGithubPages) {
      console.log('[GameService] 环境为静态托管或指定了 wasm，启用 WasmGameDriver');
      this.driver = new WasmGameDriver();
      this.driverType = 'wasm';
      await this.driver.init();
      return;
    }

    if (forceHttp) {
      console.log('[GameService] 用户指定了 http，启用 HttpGameDriver');
      this.driver = new HttpGameDriver();
      this.driverType = 'http';
      return;
    }

    // 2. 自动探测本地 HTTP 服务是否存在
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 800);
      const res = await fetch('/api/game/state', { signal: controller.signal });
      clearTimeout(timeoutId);
      if (res.ok) {
        console.log('[GameService] 探测到本地 Rust HTTP 接口在线，启用 HttpGameDriver');
        this.driver = new HttpGameDriver();
        this.driverType = 'http';
        return;
      }
    } catch (e) {
      // 网络不可达或请求超时
    }

    // 3. 本地无 HTTP 服务，安全回退到 WASM 驱动
    console.log('[GameService] 本地 HTTP 服务不可用，安全切换至 WasmGameDriver');
    this.driver = new WasmGameDriver();
    this.driverType = 'wasm';
    await this.driver.init();
  }

  async ensureReady() {
    await this.readyPromise;
  }

  async getState() {
    await this.ensureReady();
    return await this.driver.getState();
  }

  async newGame(seed, p0, p1, sims) {
    await this.ensureReady();
    return await this.driver.newGame(seed, p0, p1, sims);
  }

  async stepHuman(actionPayload) {
    await this.ensureReady();
    return await this.driver.stepHuman(actionPayload);
  }

  async stepAi(sims) {
    await this.ensureReady();
    return await this.driver.stepAi(sims);
  }

  async getStep(index) {
    await this.ensureReady();
    return await this.driver.getStep(index);
  }

  async getHistory() {
    await this.ensureReady();
    return await this.driver.getHistory();
  }

  async exportReplayData() {
    await this.ensureReady();
    if (this.driver.exportReplayData) {
      return await this.driver.exportReplayData();
    }
    return null;
  }

  persistSession() {
    if (this.driver && typeof this.driver.persistSession === 'function') {
      this.driver.persistSession();
    }
  }

  async getNeuralStatus() {
    await this.ensureReady();
    return await this.driver.getNeuralStatus();
  }

  async reloadNeural() {
    await this.ensureReady();
    return await this.driver.reloadNeural();
  }

  async setMctsSimulations(sims) {
    await this.ensureReady();
    if (this.driver.setMctsSimulations) {
      return await this.driver.setMctsSimulations(sims);
    }
    return null;
  }
}

export const gameService = new GameService();

/**
 * WASM 纯前端复盘驱动器 (GitHub Pages 模式)
 */
class WasmReplayDriver {
  constructor() {
    this.session = null;
    this.wasmInitialized = false;
    this.seed = 42;
    this.playerTypes = ['heuristic', 'heuristic'];
    this.predictor = new OnnxPredictor();
  }

  async init() {
    if (!this.wasmInitialized) {
      await initWasm();
      this.wasmInitialized = true;
    }
    // 异步加载 ONNX 模型
    this.predictor.loadModel().then(ready => {
      if (this.session && typeof this.session.set_neural_ready === 'function') {
        this.session.set_neural_ready(ready);
      }
    });
  }

  ensureSession() {
    if (!this.session) {
      // 检查是否有来自对战界面的历史转储
      const transferData = sessionStorage.getItem('splendor_duel_replay_transfer');
      if (transferData) {
        try {
          this.session = WasmReplaySession.from_replay_data(transferData);
          if (typeof this.session.set_neural_ready === 'function') {
            this.session.set_neural_ready(this.predictor.ready);
          }
          console.log('[WasmReplayDriver] 成功从对战会话恢复历史局势！');
          sessionStorage.removeItem('splendor_duel_replay_transfer');
          return;
        } catch (e) {
          console.warn('[WasmReplayDriver] 恢复转储局势失败，新建空局:', e);
        }
      }
      this.session = new WasmReplaySession(BigInt(this.seed), this.playerTypes[0], this.playerTypes[1]);
      if (typeof this.session.set_neural_ready === 'function') {
        this.session.set_neural_ready(this.predictor.ready);
      }
    }
  }

  async loadReplayData(replayData) {
    await this.init();
    try {
      const jsonStr = typeof replayData === 'string' ? replayData : JSON.stringify(replayData);
      this.session = WasmReplaySession.from_replay_data(jsonStr);
      if (typeof this.session.set_neural_ready === 'function') {
        this.session.set_neural_ready(this.predictor.ready);
      }
      return true;
    } catch (e) {
      console.error('[WasmReplayDriver] loadReplayData 失败:', e);
      return false;
    }
  }

  async getStatus() {
    await this.init();
    this.ensureSession();
    const raw = this.session.get_status_json();
    const data = JSON.parse(raw);
    data.driver_type = 'wasm';
    data.neural_available = this.predictor.ready;
    return data;
  }

  async getHistory() {
    await this.init();
    this.ensureSession();
    const raw = this.session.get_history_json();
    return JSON.parse(raw);
  }

  async getStep(index) {
    await this.init();
    this.ensureSession();
    const raw = this.session.get_step_json(index);
    return JSON.parse(raw);
  }

  async stepForward(count = 1) {
    await this.init();
    this.ensureSession();

    let lastResult = null;
    const targetCount = Math.max(1, count);
    for (let c = 0; c < targetCount; c++) {
      // 检查当前走步方是否为神经网络
      const currentType = this.session.current_player_type();
      if (currentType === 'neural') {
        if (this.predictor.ready) {
          const obsArray = this.session.encode_observation();
          const legalIds = Array.from(this.session.get_legal_action_ids());
          const legalsDto = JSON.parse(this.session.get_legal_actions_dto());

          const pred = await this.predictor.predict(new Float32Array(obsArray), legalIds, legalsDto);
          if (pred) {
            const raw = this.session.step_with_neural(
              pred.bestActionId,
              pred.winrate,
              JSON.stringify(pred.topCandidates)
            );
            lastResult = JSON.parse(raw);
            if (!lastResult.advanced) break;
            continue;
          }
        }
        console.warn('[WasmReplayDriver] 神经网络未就绪，使用启发式 AI 走步');
      }

      const raw = this.session.step_forward(1);
      lastResult = JSON.parse(raw);
      if (!lastResult.advanced) break;
    }

    return lastResult || JSON.parse(this.session.get_status_json());
  }

  async playToEnd(maxSteps = 2000) {
    await this.init();
    this.ensureSession();

    const p0Type = this.playerTypes[0];
    const p1Type = this.playerTypes[1];
    const hasNeural = (p0Type === 'neural' || p1Type === 'neural');

    // 若包含神经网络玩家且 ONNX 模型已就绪，循环异步驱动 stepForward，
    // 确保每一步都经过真实的 ONNX 前向推理，生成完整的胜率过程与候选概率分布
    if (hasNeural && this.predictor.ready) {
      let lastResult = null;
      for (let i = 0; i < maxSteps; i++) {
        lastResult = await this.stepForward(1);
        if (!lastResult || !lastResult.advanced) break;
      }
      return lastResult || JSON.parse(this.session.get_status_json());
    }

    // 纯启发式/随机对弈或模型未就绪时，直接交由 Rust WASM 极速推演
    const raw = this.session.play_to_end(maxSteps);
    return JSON.parse(raw);
  }

  async reset(seed, p0, p1) {
    await this.init();
    this.seed = seed || 42;
    this.playerTypes = [p0 || 'heuristic', p1 || 'heuristic'];
    this.session = new WasmReplaySession(BigInt(this.seed), this.playerTypes[0], this.playerTypes[1]);
    if (typeof this.session.set_neural_ready === 'function') {
      this.session.set_neural_ready(this.predictor.ready);
    }
    const data = JSON.parse(this.session.get_status_json());
    data.driver_type = 'wasm';
    data.neural_available = this.predictor.ready;
    return data;
  }

  async setPlayers(p0, p1) {
    await this.init();
    this.ensureSession();
    this.playerTypes = [p0, p1];
    const raw = this.session.set_players(p0, p1);
    return JSON.parse(raw);
  }

  async getNeuralStatus() {
    return {
      available: this.predictor.ready,
      loading: this.predictor.loading,
      error: this.predictor.error,
      model_type: 'onnx-web',
      details: this.predictor.metadata,
    };
  }

  async reloadNeural() {
    this.predictor.ready = false;
    const ok = await this.predictor.loadModel();
    if (this.session && typeof this.session.set_neural_ready === 'function') {
      this.session.set_neural_ready(ok);
    }
    return { ok };
  }
}

/**
 * HTTP REST 复盘驱动器 (本地开发模式)
 */
class HttpReplayDriver {
  async getStatus() {
    const res = await fetch('/api/status');
    const data = await res.json();
    data.driver_type = 'http';
    return data;
  }

  async getHistory() {
    const res = await fetch('/api/history');
    return await res.json();
  }

  async getStep(index) {
    const res = await fetch(`/api/step?index=${index}`);
    return await res.json();
  }

  async stepForward(count = 1) {
    const res = await fetch(`/api/step?count=${count}`, { method: 'POST' });
    return await res.json();
  }

  async playToEnd() {
    const res = await fetch('/api/play_to_end', { method: 'POST' });
    return await res.json();
  }

  async reset(seed, p0, p1) {
    const res = await fetch(`/api/reset?seed=${seed}&p0=${p0}&p1=${p1}`, { method: 'POST' });
    return await res.json();
  }

  async setPlayers(p0, p1) {
    const res = await fetch(`/api/set_players?p0=${p0}&p1=${p1}`, { method: 'POST' });
    return await res.json();
  }

  async getNeuralStatus() {
    const res = await fetch('/api/neural_status');
    return await res.json();
  }

  async reloadNeural() {
    const res = await fetch('/api/neural_reload', { method: 'POST' });
    return await res.json();
  }

  async loadReplayData(_replayData) {
    return true;
  }
}

/**
 * 统一复盘门面服务
 */
class ReplayService {
  constructor() {
    this.driver = null;
    this.driverType = 'unknown';
    this.readyPromise = this.detectAndInitDriver();
  }

  async detectAndInitDriver() {
    const isGithubPages = window.location.hostname.includes('github.io');
    const forceWasm = new URLSearchParams(window.location.search).has('wasm');
    const forceHttp = new URLSearchParams(window.location.search).has('http');

    if (forceWasm || isGithubPages) {
      this.driver = new WasmReplayDriver();
      this.driverType = 'wasm';
      await this.driver.init();
      return;
    }

    if (forceHttp) {
      this.driver = new HttpReplayDriver();
      this.driverType = 'http';
      return;
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 800);
      const res = await fetch('/api/status', { signal: controller.signal });
      clearTimeout(timeoutId);
      if (res.ok) {
        this.driver = new HttpReplayDriver();
        this.driverType = 'http';
        return;
      }
    } catch (e) {}

    this.driver = new WasmReplayDriver();
    this.driverType = 'wasm';
    await this.driver.init();
  }

  async ensureReady() {
    await this.readyPromise;
  }

  async getStatus() {
    await this.ensureReady();
    return await this.driver.getStatus();
  }

  async getHistory() {
    await this.ensureReady();
    return await this.driver.getHistory();
  }

  async getStep(index) {
    await this.ensureReady();
    return await this.driver.getStep(index);
  }

  async stepForward(count = 1) {
    await this.ensureReady();
    return await this.driver.stepForward(count);
  }

  async playToEnd() {
    await this.ensureReady();
    return await this.driver.playToEnd();
  }

  async reset(seed, p0, p1) {
    await this.ensureReady();
    return await this.driver.reset(seed, p0, p1);
  }

  async setPlayers(p0, p1) {
    await this.ensureReady();
    return await this.driver.setPlayers(p0, p1);
  }

  async getNeuralStatus() {
    await this.ensureReady();
    return await this.driver.getNeuralStatus();
  }

  async reloadNeural() {
    await this.ensureReady();
    return await this.driver.reloadNeural();
  }

  async loadReplayData(replayData) {
    await this.ensureReady();
    if (this.driver && typeof this.driver.loadReplayData === 'function') {
      return await this.driver.loadReplayData(replayData);
    }
    return false;
  }
}

export const replayService = new ReplayService();

