/**
 * Splendor Duel Sound Manager
 * 璀璨宝石：对决 沉浸式音效管理器
 * 基于 Web Audio API 程序化物理合成拟真音效，零外部资源依赖，支持外部音频无缝升级覆盖。
 */

export class SoundManager {
  constructor() {
    this.enabled = localStorage.getItem('splendor_sfx_enabled') !== 'false';
    this.volume = parseFloat(localStorage.getItem('splendor_sfx_volume') || '0.35');
    this.audioCtx = null;
    this.customAudioMap = new Map();
    this.lastPlayTime = new Map();

    // 绑定全局首次交互解锁 AudioContext
    if (typeof window !== 'undefined') {
      const unlockAudio = () => {
        this.initAudioContext();
        window.removeEventListener('pointerdown', unlockAudio);
        window.removeEventListener('keydown', unlockAudio);
      };
      window.addEventListener('pointerdown', unlockAudio, { passive: true });
      window.addEventListener('keydown', unlockAudio, { passive: true });
    }
  }

  /**
   * 初始化或唤醒 AudioContext
   */
  initAudioContext() {
    if (!this.audioCtx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (AudioContextClass) {
        this.audioCtx = new AudioContextClass();
      }
    }
    if (this.audioCtx && this.audioCtx.state === 'suspended') {
      this.audioCtx.resume().catch(() => {});
    }
  }

  /**
   * 切换静音/开启状态
   * @returns {boolean} 当前音效开启状态
   */
  toggleSound() {
    this.enabled = !this.enabled;
    localStorage.setItem('splendor_sfx_enabled', this.enabled ? 'true' : 'false');
    if (this.enabled) {
      this.initAudioContext();
      this.play('gem_select');
    }
    return this.enabled;
  }

  /**
   * 设置主音量 (0.0 ~ 1.0)
   */
  setVolume(vol) {
    this.volume = Math.max(0, Math.min(1, vol));
    localStorage.setItem('splendor_sfx_volume', this.volume.toString());
  }

  /**
   * 注册外部高保真音频资源覆盖默认合成音
   * @param {string} type 音效类型
   * @param {string} audioUrl 音频文件地址 (mp3/wav/ogg)
   */
  registerCustomAudio(type, audioUrl) {
    const audio = new Audio(audioUrl);
    audio.preload = 'auto';
    this.customAudioMap.set(type, audio);
  }

  /**
   * 播放音效
   * @param {string} type 音效类型
   */
  play(type) {
    if (!this.enabled) return;

    // 防止同类音效过于密集爆音（最小间隔 35ms）
    const nowMs = performance.now();
    const last = this.lastPlayTime.get(type) || 0;
    if (nowMs - last < 35) return;
    this.lastPlayTime.set(type, nowMs);

    this.initAudioContext();
    if (!this.audioCtx) return;

    // 如果注册了外部音频文件，优先播放
    if (this.customAudioMap.has(type)) {
      try {
        const audio = this.customAudioMap.get(type).cloneNode();
        audio.volume = this.volume;
        audio.play().catch(() => {});
        return;
      } catch (e) {}
    }

    // 默认执行 Web Audio API 物理拟真程序化合成
    try {
      switch (type) {
        case 'gem_select':
          // 宝石轻点：清脆微击声，带微小音调扰动避免机械单调
          this._synthTone(1250 + (Math.random() * 160 - 80), 0.035, 'sine', 0.16 * this.volume);
          break;

        case 'gem_clink':
          // 宝石入袋碰撞：模拟亚克力硬质筹码碰击 (多频叠加)
          this._synthTone(1550 + Math.random() * 120, 0.06, 'triangle', 0.28 * this.volume);
          setTimeout(() => {
            this._synthTone(2100 + Math.random() * 150, 0.05, 'sine', 0.2 * this.volume);
          }, 25);
          break;

        case 'gold_clink':
          // 金币碰撞：明亮金属双频铃音 + 较长延音 (2400Hz + 3800Hz)
          this._synthTone(2400, 0.35, 'sine', 0.25 * this.volume);
          this._synthTone(3850, 0.45, 'sine', 0.18 * this.volume);
          break;

        case 'card_flip':
          // 抽牌/翻牌/预留：粉红/白噪声带通滤波，模拟纸牌滑出摩擦声
          this._synthNoise(0.09, 1300, 0.22 * this.volume);
          break;

        case 'card_buy':
          // 购牌：纸牌盖桌轻击 + 悦耳双音点数结算声
          this._synthNoise(0.05, 500, 0.25 * this.volume);
          this._synthTone(523.25, 0.25, 'sine', 0.15 * this.volume); // C5
          setTimeout(() => {
            this._synthTone(659.25, 0.3, 'sine', 0.2 * this.volume); // E5
          }, 60);
          break;

        case 'privilege':
          // 特权卷轴：羊皮纸摩擦的沙沙声 + 微妙回响
          this._synthNoise(0.12, 1800, 0.18 * this.volume);
          setTimeout(() => {
            this._synthTone(880, 0.15, 'triangle', 0.12 * this.volume);
          }, 40);
          break;

        case 'replenish':
          // 棋盘补充宝石：连续多颗宝石哗啦啦落盘
          for (let i = 0; i < 4; i++) {
            setTimeout(() => {
              this._synthTone(1400 + Math.random() * 600, 0.05, 'triangle', 0.2 * this.volume);
            }, i * 45);
          }
          break;

        case 'turn_notify':
          // 轮到人类玩家回合：柔和上行双音
          this._synthTone(440, 0.12, 'sine', 0.12 * this.volume); // A4
          setTimeout(() => {
            this._synthTone(659.25, 0.22, 'sine', 0.15 * this.volume); // E5
          }, 80);
          break;

        case 'royal_claim':
          // 获得王室赞助：庄严号角三和弦
          [587.33, 739.99, 880].forEach((freq, i) => { // D5, F#5, A5
            setTimeout(() => this._synthTone(freq, 0.35, 'triangle', 0.22 * this.volume), i * 90);
          });
          break;

        case 'victory':
          // 胜利大圆满：大三和弦向上琶音 (C5 -> E5 -> G5 -> C6)
          [523.25, 659.25, 783.99, 1046.50].forEach((freq, i) => {
            setTimeout(() => this._synthTone(freq, 0.45, 'triangle', 0.3 * this.volume), i * 110);
          });
          break;

        default:
          break;
      }
    } catch (e) {
      console.warn('[SoundManager] play error:', e);
    }
  }

  /**
   * 正弦/三角波简单音调合成
   */
  _synthTone(freq, duration, waveType = 'sine', gainVal = 0.2) {
    if (!this.audioCtx) return;
    const osc = this.audioCtx.createOscillator();
    const gain = this.audioCtx.createGain();
    const t = this.audioCtx.currentTime;

    osc.type = waveType;
    osc.frequency.setValueAtTime(freq, t);
    gain.gain.setValueAtTime(Math.max(0.0001, gainVal), t);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + duration);

    osc.connect(gain);
    gain.connect(this.audioCtx.destination);
    osc.start(t);
    osc.stop(t + duration);
  }

  /**
   * 带通滤波白噪声合成 (模拟纸张滑动、摩擦、盖桌击打)
   */
  _synthNoise(duration, filterFreq = 1000, gainVal = 0.2) {
    if (!this.audioCtx) return;
    const sampleRate = this.audioCtx.sampleRate;
    const bufferSize = Math.max(1, Math.floor(sampleRate * duration));
    const buffer = this.audioCtx.createBuffer(1, bufferSize, sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      data[i] = Math.random() * 2 - 1;
    }

    const noise = this.audioCtx.createBufferSource();
    noise.buffer = buffer;

    const filter = this.audioCtx.createBiquadFilter();
    filter.type = 'bandpass';
    filter.frequency.setValueAtTime(filterFreq, this.audioCtx.currentTime);

    const gain = this.audioCtx.createGain();
    const t = this.audioCtx.currentTime;
    gain.gain.setValueAtTime(Math.max(0.0001, gainVal), t);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + duration);

    noise.connect(filter);
    filter.connect(gain);
    gain.connect(this.audioCtx.destination);
    noise.start(t);
  }
}

export const soundManager = new SoundManager();
