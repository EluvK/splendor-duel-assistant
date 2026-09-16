/**
 * Splendor Duel Shared UI Components
 * 璀璨宝石：对决 共享组件与渲染函数
 */

export const COLOR_CLASSES = {
  white: 'token-white',
  blue: 'token-blue',
  green: 'token-green',
  red: 'token-red',
  black: 'token-black',
  pearl: 'token-pearl',
  gold: 'token-gold',
  White: 'token-white',
  Blue: 'token-blue',
  Green: 'token-green',
  Red: 'token-red',
  Black: 'token-black',
  Pearl: 'token-pearl',
  Gold: 'token-gold',
};

export const COLOR_KEYS = ['white', 'blue', 'green', 'red', 'black', 'pearl', 'gold'];
export const COLOR_NAMES = ['白', '蓝', '绿', '红', '黑', '珍珠', '黄金'];

export const REPLACEABLE_GEM_KEYS = ['white', 'blue', 'green', 'red', 'black', 'pearl'];
export const REPLACEABLE_GEM_NAMES = ['白', '蓝', '绿', '红', '黑', '珍珠'];

/**
 * 生成全量 84 种自由黄金静态支付替代方案
 * 映射顺序严格与 Rust 引擎 crate::gameplay::payment::PAYMENT_PLANS 对齐
 */
export function generatePaymentPlans() {
  const plans = [];
  // k = 0 (1 种): 全部为 0
  plans.push([0, 0, 0, 0, 0, 0]);
  // k = 1 (6 种): 6 种颜色各取 1 枚
  for (let c0 = 0; c0 < 6; c0++) {
    const p = [0, 0, 0, 0, 0, 0];
    p[c0] += 1;
    plans.push(p);
  }
  // k = 2 (21 种): 6 种颜色选 2 枚 (带放回)
  for (let c0 = 0; c0 < 6; c0++) {
    for (let c1 = c0; c1 < 6; c1++) {
      const p = [0, 0, 0, 0, 0, 0];
      p[c0] += 1;
      p[c1] += 1;
      plans.push(p);
    }
  }
  // k = 3 (56 种): 6 种颜色选 3 枚 (带放回)
  for (let c0 = 0; c0 < 6; c0++) {
    for (let c1 = c0; c1 < 6; c1++) {
      for (let c2 = c1; c2 < 6; c2++) {
        const p = [0, 0, 0, 0, 0, 0];
        p[c0] += 1;
        p[c1] += 1;
        p[c2] += 1;
        plans.push(p);
      }
    }
  }
  return plans;
}

export const PAYMENT_PLANS = generatePaymentPlans();

/**
 * 解析支付方案 plan_id，提取消耗黄金数及保留的天然宝石
 */
export function getPaymentPlanDetails(planId) {
  if (!planId || planId === 0 || !PAYMENT_PLANS[planId]) {
    return {
      planId: 0,
      isDefault: true,
      goldCount: 0,
      preserved: [],
    };
  }
  const plan = PAYMENT_PLANS[planId];
  const preserved = [];
  let goldCount = 0;
  for (let i = 0; i < 6; i++) {
    const count = plan[i];
    if (count > 0) {
      goldCount += count;
      preserved.push({
        color: REPLACEABLE_GEM_KEYS[i],
        name: REPLACEABLE_GEM_NAMES[i],
        count,
      });
    }
  }
  return {
    planId,
    isDefault: false,
    goldCount,
    preserved,
  };
}

export const ABILITY_DISPLAY_NAMES = {
  ExtraTurn: '额外回合 🔁',
  TakePrivilege: '拿特权 📜',
  TakeSameColor: '拿同色 💎',
  StealToken: '偷标记 🤏',
  ColorCopy: '变色绑定 🎨',
  ColorCopyAndExtraTurn: '变色+回合 🔁',
};

export const ABILITY_DETAILS = {
  ExtraTurn: '购买此卡牌后，立即额外执行一个完整的行动轮。',
  TakePrivilege: '从特权池中拿取 1 个特权卷轴；若特权池已空，则强行从对手处夺取 1 个。',
  TakeSameColor: '立即从 5×5 棋盘上免费拿取 1 颗与本卡牌同色的宝石标记。',
  StealToken: '强行从对手手中夺取 1 颗非黄金标记（普通宝石或珍珠）。',
  ColorCopy: '变色复制：必须附着到自己已拥有的一张带颜色卡牌上，获得对应颜色的永久减免。',
  ColorCopyAndExtraTurn: '变色复制并立即获得一个额外行动轮！',
};

export const ROYAL_ABILITY_DISPLAY_NAMES = {
  ExtraTurn: '额外回合 🔁',
  TakePrivilege: '拿特权 📜',
  StealToken: '偷标记 🤏',
};

export const ROYAL_ABILITY_DETAILS = {
  ExtraTurn: '认领该王室卡后，立即额外执行一个完整的行动轮。',
  TakePrivilege: '从特权池拿取 1 个特权卷轴（若特权池为空则夺取对手的）。',
  StealToken: '强行从对手手中夺取 1 颗非黄金标记。',
};

/**
 * 计算卡牌在当前玩家资产下的购买差额与实付净花费
 */
export function calculateCardDeficit(card, player) {
  if (!card || !player) return null;
  const cost = card.cost || {};
  const bonuses = player.bonuses || [0, 0, 0, 0, 0, 0, 0];
  const tokens = player.tokens || [0, 0, 0, 0, 0, 0, 0];

  let totalDeficit = 0;
  const deficitDetails = [];
  const freeGold = tokens[6] || 0; // gold
  const netCost = {};

  COLOR_KEYS.forEach((col, idx) => {
    const raw = cost[col] || 0;
    if (raw === 0) return;
    const bonus = (idx < 5) ? (bonuses[idx] || 0) : 0;
    const need = Math.max(0, raw - bonus);
    netCost[col] = need;
    const have = tokens[idx] || 0;
    if (have < need) {
      const lack = need - have;
      totalDeficit += lack;
      deficitDetails.push({ color: col, name: COLOR_NAMES[idx], lack });
    }
  });

  const canAfford = freeGold >= totalDeficit;
  const goldUsed = Math.min(freeGold, totalDeficit);
  const remainingLack = totalDeficit - goldUsed;

  return {
    canAfford,
    totalDeficit,
    remainingLack,
    deficitDetails,
    netCost,
    goldUsed,
    freeGold
  };
}

/**
 * 全套 67 张珠宝卡对应 BGA 雪碧图的精确 1-based 帧索引
 * 0 号帧为各等级牌堆卡背 (Face-down card back)!
 * Tier 1: id 0..29 (cards1.jpg, 31 帧, 卡背在 index 0, 正面在 index 1..30)
 * Tier 2: id 30..53 (cards2.jpg, 25 帧, 卡背在 index 0, 正面在 index 1..24)
 * Tier 3: id 54..66 (cards3.jpg, 14 帧, 卡背在 index 0, 正面在 index 1..13)
 */
export const BGA_CARD_SPRITE_MAP = [
  // Level 1: id 0..29 (30 cards)
  16, 18, 19, 20, 17, 21, 23, 24, 25, 22, 11, 13, 14, 15, 12, 6, 8, 9, 10, 7, 1, 3, 4, 5, 2, 28, 26, 27, 30, 29,
  // Level 2: id 30..53 (24 cards)
  14, 16, 13, 15, 18, 20, 17, 19, 10, 12, 9, 11, 6, 8, 5, 7, 2, 4, 1, 3, 24, 21, 22, 23,
  // Level 3: id 54..66 (13 cards)
  7, 8, 9, 10, 5, 6, 3, 4, 1, 2, 13, 12, 11
];

/**
 * 为卡牌元素配置 BGA 原版雪碧图 (Sprite Sheet) 正面样式
 * @param {HTMLElement} el
 * @param {Object} card 卡牌 DTO
 */
export function applyBgaCardSprite(el, card) {
  if (!card) return;
  const tier = card.tier || 1;
  let sheetUrl, colIndex, totalCols;

  if (tier === 1) {
    sheetUrl = 'assets/images/cards1.jpg';
    totalCols = 31;
    colIndex = BGA_CARD_SPRITE_MAP[card.id] ?? (card.id + 1);
  } else if (tier === 2) {
    sheetUrl = 'assets/images/cards2.jpg';
    totalCols = 25;
    colIndex = BGA_CARD_SPRITE_MAP[card.id] ?? (card.id - 30 + 1);
  } else {
    sheetUrl = 'assets/images/cards3.jpg';
    totalCols = 14;
    colIndex = BGA_CARD_SPRITE_MAP[card.id] ?? (card.id - 54 + 1);
  }

  const posX = (colIndex / (totalCols - 1)) * 100;
  el.style.backgroundImage = `url('${sheetUrl}')`;
  el.style.backgroundSize = `${totalCols * 100}% 100%`;
  el.style.backgroundPosition = `${posX.toFixed(4)}% 0%`;
  el.style.backgroundRepeat = 'no-repeat';
}

/**
 * 为牌堆元素配置 BGA 卡背原画 (0 号帧)
 * @param {HTMLElement} el
 * @param {number} tier 1, 2, 3
 */
export function applyBgaDeckBackSprite(el, tier) {
  let sheetUrl, totalCols;
  if (tier === 1) {
    sheetUrl = 'assets/images/cards1.jpg';
    totalCols = 31;
  } else if (tier === 2) {
    sheetUrl = 'assets/images/cards2.jpg';
    totalCols = 25;
  } else {
    sheetUrl = 'assets/images/cards3.jpg';
    totalCols = 14;
  }
  // 0 号帧就是该等级对应的卡背 (Face-down card back)，位于最左侧 0%
  el.style.backgroundImage = `url('${sheetUrl}')`;
  el.style.backgroundSize = `${totalCols * 100}% 100%`;
  el.style.backgroundPosition = `0% 0%`;
  el.style.backgroundRepeat = 'no-repeat';
}

/**
 * 创建实体牌库卡背元素 (带剩余数量徽章与交互)
 * @param {number} tier 1, 2, 3
 * @param {number} count 牌堆剩余张数
 * @param {Object} options 配置项 (interactive, canReserve, onReserveDeck)
 */
export function createDeckPileElement(tier, count, options = {}) {
  const el = document.createElement('div');
  el.className = 'card-item deck-pile';
  applyBgaDeckBackSprite(el, tier);

  const hasCards = count > 0;
  if (!hasCards) {
    el.classList.add('deck-empty');
  }

  const isPendingDeck = Boolean(
    options.pendingReserveTarget &&
    options.pendingReserveTarget.isDeck &&
    options.pendingReserveTarget.tier === `Tier${tier}`
  );
  if (isPendingDeck) {
    el.classList.add('card-pending-reserve');
  }

  const tierNames = { 1: '初阶 (L1)', 2: '中阶 (L2)', 3: '高阶 (L3)' };
  el.title = `【${tierNames[tier] || '等级' + tier} 牌库】剩余 ${count} 张${options.canReserve ? ' (点击可盲抽预留)' : ''}`;

  const countBadge = document.createElement('div');
  countBadge.className = 'deck-count-badge';
  countBadge.innerText = `${count} 张`;
  el.appendChild(countBadge);

  if (isPendingDeck) {
    const pendingTag = document.createElement('span');
    pendingTag.className = 'pending-reserve-tag';
    pendingTag.innerText = '📌 待选黄金';
    el.appendChild(pendingTag);
  }

  if (options.interactive && hasCards && options.canReserve) {
    el.classList.add('deck-clickable');
    const overlay = document.createElement('div');
    overlay.className = 'card-action-overlay';
    const resBtn = document.createElement('button');
    resBtn.className = `btn-secondary card-action-btn ${isPendingDeck ? 'btn-pending-active' : ''}`;
    resBtn.innerText = isPendingDeck ? '✖ 取消预留' : '🎴 盲抽预留';
    resBtn.onclick = (e) => {
      e.stopPropagation();
      if (options.onReserveDeck) options.onReserveDeck(tier);
    };
    overlay.appendChild(resBtn);
    el.appendChild(overlay);

    el.onclick = () => {
      if (options.onReserveDeck) options.onReserveDeck(tier);
    };
  }

  return el;
}

/**
 * 为王室赞助卡配置 BGA 原版雪碧图样式 (royal-cards.jpg: 4 帧)
 * @param {HTMLElement} el
 * @param {Object} royal 王室卡 DTO
 */
export function applyBgaRoyalSprite(el, royal) {
  if (!royal) return;
  const colIndex = Math.max(0, Math.min(3, royal.id));
  const posX = (colIndex / 3) * 100;
  el.style.backgroundImage = `url('assets/images/royal-cards.jpg')`;
  el.style.backgroundSize = `400% 100%`;
  el.style.backgroundPosition = `${posX.toFixed(4)}% 0%`;
  el.style.backgroundRepeat = 'no-repeat';
}

/**
 * 全局即时卡牌悬停浮层单例 (Card Hovercard)
 */
class CardHovercardManager {
  constructor() {
    this.el = null;
    this.showTimer = null;
    this.hideTimer = null;
    this.currentCard = null;
    if (typeof document !== 'undefined' && document.body) {
      this.initDOM();
    }
  }

  initDOM() {
    if (typeof document === 'undefined') return;
    if (document.getElementById('globalCardHovercard')) {
      this.el = document.getElementById('globalCardHovercard');
      return;
    }
    if (!document.body) return;
    const el = document.createElement('div');
    el.id = 'globalCardHovercard';
    el.className = 'card-hovercard';
    el.style.display = 'none';
    document.body.appendChild(el);
    this.el = el;
  }

  show(targetEl, card, player) {
    if (!this.el) this.initDOM();
    if (!this.el) return;
    if (this.hideTimer) clearTimeout(this.hideTimer);
    if (this.showTimer) clearTimeout(this.showTimer);

    this.showTimer = setTimeout(() => {
      this.renderContent(card, player);
      this.position(targetEl);
      this.el.style.display = 'flex';
      this.el.classList.add('visible');
    }, 60);
  }

  hide() {
    if (this.showTimer) clearTimeout(this.showTimer);
    this.hideTimer = setTimeout(() => {
      if (this.el) {
        this.el.classList.remove('visible');
        this.el.style.display = 'none';
      }
    }, 80);
  }

  renderContent(card, player) {
    const isRoyal = !card.tier && card.points !== undefined && card.cost === undefined;
    const tierName = isRoyal ? '王室赞助卡' : (card.tier === 3 ? '高阶 (Tier 3)' : (card.tier === 2 ? '中阶 (Tier 2)' : '初阶 (Tier 1)'));
    const colorCn = translateColor(card.color || '');
    const abilityName = card.ability ? (ABILITY_DISPLAY_NAMES[card.ability] || ROYAL_ABILITY_DISPLAY_NAMES[card.ability] || card.ability) : '';
    const abilityDesc = card.ability ? (ABILITY_DETAILS[card.ability] || ROYAL_ABILITY_DETAILS[card.ability] || '特殊战术技能') : '';

    // 左侧：1.4x 放大雪碧图展示
    const previewEl = document.createElement('div');
    previewEl.className = 'hovercard-preview-card';
    if (isRoyal) {
      applyBgaRoyalSprite(previewEl, card);
    } else {
      applyBgaCardSprite(previewEl, card);
    }

    // 右侧：结构化信息
    let costHtml = '';
    let deficitHtml = '';

    if (!isRoyal && card.cost) {
      const deficit = calculateCardDeficit(card, player);
      const rawCosts = COLOR_KEYS
        .map(col => {
          const amt = card.cost[col] || card.cost[col.charAt(0).toUpperCase() + col.slice(1)] || 0;
          return { col, amt };
        })
        .filter(item => item.amt > 0)
        .map(({ col, amt }) => {
          const cCn = translateColor(col);
          const colKey = col.toLowerCase();
          return `<span class="hovercard-chip chip-${colKey}" title="${cCn} ×${amt}"><span class="chip-token-icon token-${colKey}"></span><span class="chip-qty">×${amt}</span></span>`;
        }).join('');

      costHtml = `
        <div class="hovercard-row">
          <span class="hovercard-label">卡牌原价:</span>
          <div class="hovercard-chips-wrap">${rawCosts || '<span style="color:#22c55e;">免费</span>'}</div>
        </div>
      `;

      if (deficit) {
        if (deficit.canAfford) {
          const goldNote = deficit.goldUsed > 0
            ? ` <span class="gold-used-note">(需消耗自由黄金 <span class="chip-token-icon token-gold inline-token"></span><span class="chip-qty">×${deficit.goldUsed}</span>)</span>`
            : ' (天然资源完全满足)';
          deficitHtml = `<div class="hovercard-status can-afford">✅ 当前资产可立即购买${goldNote}</div>`;
        } else {
          const lackChips = deficit.deficitDetails.map(d => {
            const colKey = d.color.toLowerCase();
            return `<span class="hovercard-chip deficit-chip chip-${colKey}" title="缺少 ${d.name} ×${d.lack}"><span class="chip-token-icon token-${colKey}"></span><span class="chip-qty">×${d.lack}</span></span>`;
          }).join('');

          const goldDeductHint = deficit.freeGold > 0
            ? ` (已折算黄金 ×${deficit.freeGold})`
            : '';

          deficitHtml = `
            <div class="hovercard-status cannot-afford">
              <div class="cannot-afford-header">
                <span class="cannot-afford-title">❌ 无法购买</span>
                <span class="deficit-summary">还差 <b>${deficit.remainingLack}</b> 标记${goldDeductHint}</span>
              </div>
              <div class="hovercard-row deficit-chips-row">
                <span class="hovercard-label deficit-label">缺少宝石:</span>
                <div class="hovercard-chips-wrap">${lackChips}</div>
              </div>
            </div>
          `;
        }
      }
    }

    const bonusHtml = (card.bonus > 0 && card.color !== 'joker' && card.color !== 'points')
      ? `<div class="hovercard-bonus"><span class="chip-token-icon token-${card.color}"></span> 永久减免: <b>${colorCn} +${card.bonus}</b></div>`
      : '';

    const abilityHtml = abilityName ? `
      <div class="hovercard-ability">
        <div class="ability-title">⚡ 技能: ${abilityName}</div>
        <div class="ability-desc">${abilityDesc}</div>
      </div>
    ` : '';

    this.el.innerHTML = `
      <div class="hovercard-left"></div>
      <div class="hovercard-right">
        <div class="hovercard-header">
          <span class="hovercard-title">【${tierName} #${card.id}】</span>
          <span class="hovercard-badges">
            <span class="badge-points">⭐ ${card.points || 0} 声望</span>
            ${card.crowns ? `<span class="badge-crowns">👑 ${card.crowns} 王冠</span>` : ''}
          </span>
        </div>
        ${bonusHtml}
        ${costHtml}
        ${deficitHtml}
        ${abilityHtml}
      </div>
    `;

    this.el.querySelector('.hovercard-left').appendChild(previewEl);
  }

  position(targetEl) {
    const rect = targetEl.getBoundingClientRect();
    const cardRect = this.el.getBoundingClientRect();
    const padding = 12;

    // 默认居于目标卡牌右侧，若溢出则居于左侧
    let left = rect.right + padding;
    if (left + 360 > window.innerWidth) {
      left = rect.left - 360 - padding;
    }
    if (left < padding) left = padding;

    let top = rect.top + (rect.height / 2) - 100;
    if (top < padding) top = padding;
    if (top + 280 > window.innerHeight) {
      top = window.innerHeight - 280 - padding;
    }

    this.el.style.left = `${Math.round(left)}px`;
    this.el.style.top = `${Math.round(top)}px`;
  }
}

export const globalHovercard = new CardHovercardManager();

function translateColor(color) {
  const map = {
    Blue: '蓝', Red: '红', Green: '绿', White: '白', Black: '黑', Pearl: '珍珠', Gold: '黄金',
    blue: '蓝', red: '红', green: '绿', white: '白', black: '黑', pearl: '珍珠', gold: '黄金'
  };
  return map[color] || color;
}

/**
 * 渲染 5x5 棋盘
 * @param {Array<Array<string|null>>} boardData 5x5 二维矩阵
 * @param {HTMLElement} containerEl 容器 DOM 元素
 * @param {Object} options 配置项
 */
export function renderBoard(boardData, containerEl, options = {}) {
  containerEl.innerHTML = '';
  const selectedSet = new Set((options.selectedPositions || []).map(([r, c]) => `${r},${c}`));
  const candidateSet = new Set((options.candidatePositions || []).map(([r, c]) => `${r},${c}`));
  const highlightSet = new Set((options.highlightPositions || []).map(([r, c]) => `${r},${c}`));
  const goldSelected = options.goldSelectedPos ? `${options.goldSelectedPos[0]},${options.goldSelectedPos[1]}` : null;
  const goldCandidatesSet = new Set((options.goldCandidatePositions || []).map(([r, c]) => `${r},${c}`));
  const ghostMap = new Map();

  if (options.ghostTokens && Array.isArray(options.ghostTokens)) {
    options.ghostTokens.forEach(g => {
      ghostMap.set(`${g.r},${g.c}`, g);
    });
  }

  let countTokens = 0;
  for (let r = 0; r < 5; r++) {
    for (let c = 0; c < 5; c++) {
      const cell = document.createElement('div');
      cell.className = 'board-cell';
      cell.setAttribute('data-pos', `${r},${c}`);
      const key = `${r},${c}`;

      if (selectedSet.has(key)) {
        cell.classList.add('cell-selected');
        const selIdx = (options.selectedPositions || []).findIndex(([sr, sc]) => sr === r && sc === c);
        if (selIdx >= 0) {
          const badge = document.createElement('span');
          badge.className = 'selected-order-badge';
          badge.innerText = selIdx + 1;
          cell.appendChild(badge);
        }
      }
      if (candidateSet.has(key)) cell.classList.add('cell-candidate');
      if (highlightSet.has(key)) cell.classList.add('cell-highlight');
      if (goldSelected === key) cell.classList.add('cell-gold-selected');
      if (goldCandidatesSet.has(key)) cell.classList.add('cell-gold-candidate');

      const gem = boardData[r][c];
      if (gem) {
        countTokens++;
        const tok = document.createElement('div');
        tok.className = `token token-${gem}`;
        const colIdx = COLOR_KEYS.indexOf(gem);
        const baseTitle = `${colIdx >= 0 ? COLOR_NAMES[colIdx] : gem}标记 (${r}, ${c})`;
        tok.title = (gem === 'gold' && goldCandidatesSet.has(key)) ? `💰 拿取此黄金 (${r}, ${c}) 并预留卡牌` : baseTitle;
        cell.appendChild(tok);

        if (options.clickable) {
          cell.classList.add('cell-clickable');
          cell.onclick = () => {
            if (options.onCellClick) options.onCellClick(r, c, gem);
          };
        }
      } else if (ghostMap.has(key)) {
        // 渲染幽灵残留发光标记 (Ghost Highlighting)
        const ghost = ghostMap.get(key);
        cell.classList.add('cell-ghost-active');
        const ghostEl = document.createElement('div');
        ghostEl.className = `ghost-token ghost-${ghost.gem || 'blue'}`;
        ghostEl.innerHTML = `<span class="ghost-badge">${ghost.label || '取走'}</span>`;
        ghostEl.title = `上一步从此处取走 ${translateColor(ghost.gem)} 标记`;
        cell.appendChild(ghostEl);
      }

      containerEl.appendChild(cell);
    }
  }

  return countTokens;
}

/**
 * 渲染卡牌列表（金字塔货架或预留卡槽）
 */
export function renderCardsList(cards, container, options = {}) {
  container.innerHTML = '';

  // 如果提供了牌堆信息，首位放置对应的等级牌库卡背元素
  if (options.deckInfo) {
    const deckEl = createDeckPileElement(options.deckInfo.tier, options.deckInfo.count, {
      interactive: options.interactive,
      canReserve: options.canReserve,
      onReserveDeck: options.onReserveDeck,
      pendingReserveTarget: options.pendingReserveTarget,
    });
    if (options.isReserveGuidance) {
      deckEl.classList.add('pulse-reservable');
    }
    container.appendChild(deckEl);
  }

  if (!cards || cards.length === 0) {
    if (options.emptyText && !options.deckInfo) {
      container.innerHTML = `<span style="font-size:0.7rem; color:var(--text-muted);">${options.emptyText}</span>`;
    }
    return;
  }

  const isCompact = options.isCompact || false;
  const affordableSet = new Set(options.affordableIds || []);
  const fromReserved = options.fromReserved || false;

  cards.forEach(c => {
    if (!c) return;
    const el = document.createElement('div');
    el.className = `card-item ${isCompact ? 'compact' : ''}`;

    // 若是对局中对手暗抽预留的卡牌（对手不可见）
    const isHiddenCard = options.isOpponent && (c.is_public === false);
    if (isHiddenCard) {
      applyBgaDeckBackSprite(el, c.tier || 1);
      el.title = `【预留卡】暗抽等级 ${c.tier} 卡牌（对手私有暗牌）`;
      container.appendChild(el);
      return;
    }

    applyBgaCardSprite(el, c);

    const canAfford = affordableSet.has(c.id);
    if (canAfford) {
      el.classList.add('affordable');
    }

    // 若该卡牌正处于等待选择棋盘黄金的预留中
    const isPendingReserveCard = Boolean(
      !fromReserved &&
      options.pendingReserveTarget &&
      !options.pendingReserveTarget.isDeck &&
      options.pendingReserveTarget.card?.id === c.id
    );

    if (isPendingReserveCard) {
      el.classList.add('card-pending-reserve');
    }

    // 若当前处于黄金已选定的预留引导阶段，所有金字塔明牌带有脉冲引导
    if (options.isReserveGuidance && !fromReserved) {
      el.classList.add('pulse-reservable');
    }

    // 绑定即时 Hovercard 事件（告别 500ms 原生延迟）
    el.onmouseenter = () => {
      globalHovercard.show(el, c, options.currentPlayerState);
    };
    el.onmouseleave = () => {
      globalHovercard.hide();
    };

    const costsDetail = Object.entries(c.cost)
      .filter(([_, amount]) => amount > 0)
      .map(([color, amount]) => `${color}:${amount}`)
      .join(', ');
    const abilityLabel = c.ability ? (ABILITY_DISPLAY_NAMES[c.ability] || c.ability) : '';
    const bonusText = c.bonus > 0 && c.color !== 'joker' && c.color !== 'points' ? `${c.color}+${c.bonus}` : '';
    const secretTag = (fromReserved && c.is_public === false) ? '\n【暗抽私有】(仅自己可见)' : '';

    el.title = `【卡牌 #${c.id}】L${c.tier} ${c.color}\n声望: ${c.points}⭐ | 王冠: ${c.crowns}👑\n永久加成: ${bonusText || '无'}\n能力: ${abilityLabel || '无'}\n花费: ${costsDetail || '免费'}${secretTag}`;
    el.innerHTML = '';

    // 若为自己暗抽私有卡牌，增加小角标提示
    if (fromReserved && c.is_public === false) {
      const lockBadge = document.createElement('span');
      lockBadge.style.position = 'absolute';
      lockBadge.style.bottom = '2px';
      lockBadge.style.right = '2px';
      lockBadge.style.fontSize = '0.65rem';
      lockBadge.style.backgroundColor = 'rgba(0,0,0,0.6)';
      lockBadge.style.borderRadius = '3px';
      lockBadge.style.padding = '1px 3px';
      lockBadge.innerText = '🔒暗';
      el.appendChild(lockBadge);
    } else if (isPendingReserveCard) {
      const pendingTag = document.createElement('span');
      pendingTag.className = 'pending-reserve-tag';
      pendingTag.innerText = '📌 待选黄金';
      el.appendChild(pendingTag);
    }

    // 交互手柄层（当传入操作回调时）
    if (options.interactive) {
      const overlay = document.createElement('div');
      overlay.className = 'card-action-overlay';

      if (canAfford && options.onPurchase) {
        const buyBtn = document.createElement('button');
        buyBtn.className = 'btn-success card-action-btn';
        buyBtn.innerText = '💎 购买卡牌';
        buyBtn.onclick = (e) => {
          e.stopPropagation();
          options.onPurchase(c, fromReserved);
        };
        overlay.appendChild(buyBtn);
      }

      if (!fromReserved && options.canReserve && options.onReserve) {
        const resBtn = document.createElement('button');
        resBtn.className = `btn-secondary card-action-btn ${isPendingReserveCard ? 'btn-pending-active' : ''}`;
        resBtn.innerText = isPendingReserveCard ? '✖ 取消待选' : '📌 预留卡牌';
        resBtn.onclick = (e) => {
          e.stopPropagation();
          options.onReserve(c);
        };
        overlay.appendChild(resBtn);
      }

      if (overlay.children.length > 0) {
        el.appendChild(overlay);
      }
    }

    container.appendChild(el);
  });
}

/**
 * 渲染玩家已购买卡牌微缩展示区 (Tableau)
 * 按照 5 种基础颜色分列，每列自下而上层叠只露出顶部卡头，并在列头整合永久减免 Bonus
 */
export function renderPurchasedCards(cards, container, bonuses = [0, 0, 0, 0, 0]) {
  container.innerHTML = '';
  const BASE_COLORS = ['white', 'blue', 'green', 'red', 'black'];
  const cardsByColor = {
    white: [],
    blue: [],
    green: [],
    red: [],
    black: [],
    points: [],
  };

  (cards || []).forEach(c => {
    if (!c) return;
    const col = (c.color || '').toLowerCase();
    if (cardsByColor[col]) {
      cardsByColor[col].push(c);
    } else {
      cardsByColor.points.push(c);
    }
  });

  BASE_COLORS.forEach((colKey) => {
    const colDiv = document.createElement('div');
    colDiv.className = `tableau-color-col col-${colKey}`;

    // 通过精确颜色键匹配后端对应的永久减免数值，防止由于遍历顺序不一致导致减免数字偏移错位
    const colIdx = COLOR_KEYS.indexOf(colKey);
    const bonusVal = (colIdx >= 0 && colIdx < 5) ? (bonuses[colIdx] || 0) : 0;
    const colName = (colIdx >= 0) ? COLOR_NAMES[colIdx] : colKey;
    const cardList = cardsByColor[colKey] || [];

    // 列头：宝石徽章 + Bonus 减免数
    const header = document.createElement('div');
    header.className = `tableau-col-header header-${colKey} ${bonusVal > 0 ? 'has-bonus' : ''}`;
    header.innerHTML = `
      <span class="badge-gem-icon token-${colKey}"></span>
      <span class="bonus-num">+${bonusVal}</span>
    `;
    header.title = `${colName}永久减免: +${bonusVal} (已拥有 ${cardList.length} 张卡牌)`;
    colDiv.appendChild(header);

    // 卡牌垂直堆叠槽 (Stack Body)
    const stackBody = document.createElement('div');
    stackBody.className = 'tableau-cards-stack';

    if (cardList.length === 0) {
      const emptySlot = document.createElement('div');
      emptySlot.className = `tableau-empty-slot slot-${colKey}`;
      stackBody.appendChild(emptySlot);
    } else {
      cardList.forEach((c, cIdx) => {
        const cardEl = document.createElement('div');
        cardEl.className = 'tableau-stacked-card';
        cardEl.style.zIndex = cIdx + 1;
        applyBgaCardSprite(cardEl, c);

        cardEl.onmouseenter = () => {
          globalHovercard.show(cardEl, c, null);
        };
        cardEl.onmouseleave = () => {
          globalHovercard.hide();
        };

        stackBody.appendChild(cardEl);
      });
    }

    colDiv.appendChild(stackBody);
    container.appendChild(colDiv);
  });

  // 若有纯分卡 (points)
  if (cardsByColor.points.length > 0) {
    const pointsCol = document.createElement('div');
    pointsCol.className = 'tableau-color-col col-points';
    const header = document.createElement('div');
    header.className = 'tableau-col-header header-points';
    header.innerHTML = `<span style="font-size:0.65rem;">⭐</span><span class="bonus-num">${cardsByColor.points.length}</span>`;
    pointsCol.appendChild(header);

    const stackBody = document.createElement('div');
    stackBody.className = 'tableau-cards-stack';
    cardsByColor.points.forEach((c, cIdx) => {
      const cardEl = document.createElement('div');
      cardEl.className = 'tableau-stacked-card';
      cardEl.style.zIndex = cIdx + 1;
      applyBgaCardSprite(cardEl, c);
      cardEl.onmouseenter = () => globalHovercard.show(cardEl, c, null);
      cardEl.onmouseleave = () => globalHovercard.hide();
      stackBody.appendChild(cardEl);
    });
    pointsCol.appendChild(stackBody);
    container.appendChild(pointsCol);
  }
}

/**
 * 渲染玩家已获得的王室卡
 */
export function renderPlayerRoyals(royals, container) {
  container.innerHTML = '';
  if (!royals || royals.length === 0) {
    container.innerHTML = '<span style="font-size:0.7rem; color:var(--text-muted); padding:2px;">暂无王室赞助</span>';
    return;
  }
  royals.forEach(r => {
    const el = document.createElement('div');
    el.className = 'royal-item mini';
    applyBgaRoyalSprite(el, r);
    const abilityText = r.ability ? (ROYAL_ABILITY_DISPLAY_NAMES[r.ability] || r.ability) : '荣誉赞助';
    el.title = `【王室赞助卡 #${r.id}】\n声望: ${r.points}⭐\n能力: ${abilityText}`;
    el.innerHTML = '';
    container.appendChild(el);
  });
}

/**
 * 渲染中央王室卡池
 */
export function renderRoyalsPool(royals, container, options = {}) {
  container.innerHTML = '';
  if (!royals || royals.length === 0) {
    container.innerHTML = '<span style="font-size:0.7rem; color:var(--text-muted); padding:2px;">王室卡已全部被认领</span>';
    return;
  }

  royals.forEach(r => {
    const el = document.createElement('div');
    el.className = 'royal-item';
    applyBgaRoyalSprite(el, r);
    const abilityText = r.ability ? (ROYAL_ABILITY_DISPLAY_NAMES[r.ability] || r.ability) : '荣誉赞助';
    el.title = `【王室赞助卡 #${r.id}】\n声望: ${r.points}⭐\n能力: ${abilityText}`;
    el.innerHTML = '';

    // 绑定即时 Hovercard
    el.onmouseenter = () => {
      globalHovercard.show(el, r, null);
    };
    el.onmouseleave = () => {
      globalHovercard.hide();
    };

    if (options.selectable) {
      el.style.cursor = 'pointer';
      el.style.boxShadow = '0 0 16px #fbbf24';
      el.classList.add('royal-selectable');
      el.onclick = () => {
        if (options.onSelect) options.onSelect(r);
      };
    }

    container.appendChild(el);
  });
}

/**
 * 渲染玩家仪表盘
 */
export function renderPlayerDashboard(p, cardEl, isActing, isNext, prefix, options = {}) {
  if (isActing) {
    cardEl.classList.add('acting');
  } else {
    cardEl.classList.remove('acting');
  }

  if (isNext) {
    cardEl.classList.add('next-turn');
  } else {
    cardEl.classList.remove('next-turn');
  }

  // 赛点危机检测 (声望 >= 16, 王冠 >= 8, 单色 >= 8)
  const maxColor = Math.max(...p.color_points);
  const isPointsThreat = p.total_points >= 16;
  const isCrownsThreat = p.total_crowns >= 8;
  const isColorThreat = maxColor >= 8;
  const inThreat = isPointsThreat || isCrownsThreat || isColorThreat;

  if (inThreat) {
    cardEl.classList.add('player-in-threat');
  } else {
    cardEl.classList.remove('player-in-threat');
  }

  // 动态渲染席位身份与先后手 (精简纯粹，移除全部文字 badge)
  const nameEl = cardEl.querySelector('.player-name');
  if (nameEl) {
    const pIndex = prefix === 'p0' ? 0 : 1;
    const pColor = pIndex === 0 ? '#38bdf8' : '#f472b6';
    const orderTag = pIndex === 0 ? '先手' : '后手';
    nameEl.innerHTML = `
      <span style="color:${pColor}; font-weight:700;">👤 Player ${pIndex}</span>
      <span style="font-size:0.75rem; color:var(--text-muted); font-weight:normal;">(${orderTag})</span>
    `;
  }

  const privEl = document.getElementById(`${prefix}Privileges`);
  if (privEl) privEl.innerHTML = `<span class="privilege-icon"></span> ${p.privileges}`;

  const ptsEl = document.getElementById(`${prefix}Points`);
  if (ptsEl) ptsEl.innerText = p.total_points;
  const ptsBar = document.getElementById(`${prefix}PointsBar`);
  if (ptsBar) {
    ptsBar.style.width = `${Math.min(100, (p.total_points / 20) * 100)}%`;
  }

  const crnEl = document.getElementById(`${prefix}Crowns`);
  if (crnEl) crnEl.innerText = p.total_crowns;
  const crnBar = document.getElementById(`${prefix}CrownsBar`);
  if (crnBar) {
    crnBar.style.width = `${Math.min(100, (p.total_crowns / 10) * 100)}%`;
  }

  const colEl = document.getElementById(`${prefix}MaxColor`);
  if (colEl) colEl.innerText = maxColor;
  const colBar = document.getElementById(`${prefix}ColorBar`);
  if (colBar) {
    colBar.style.width = `${Math.min(100, (maxColor / 10) * 100)}%`;
  }

  const tokTotal = document.getElementById(`${prefix}TokenTotal`);
  if (tokTotal) tokTotal.innerText = p.token_total;

  // 手牌标记筹码 7 列垂直堆叠 (Chip Stacks)
  const tokenContainer = document.getElementById(`${prefix}Tokens`);
  if (tokenContainer) {
    tokenContainer.innerHTML = '';
    COLOR_KEYS.forEach((colKey, idx) => {
      const count = (p.tokens && p.tokens[idx]) ? p.tokens[idx] : 0;
      const stackCol = document.createElement('div');
      stackCol.className = `token-stack-col col-${colKey} ${count > 0 ? 'has-tokens' : 'is-empty'}`;

      // 顶部数量标记
      const countBadge = document.createElement('span');
      countBadge.className = `stack-count-badge ${count > 0 ? 'active' : ''}`;
      countBadge.innerText = count > 0 ? count : '-';
      stackCol.appendChild(countBadge);

      // 垂直筹码堆叠槽
      const stackBody = document.createElement('div');
      stackBody.className = 'stack-body';

      if (count === 0) {
        const emptySlot = document.createElement('div');
        emptySlot.className = `token-slot-empty slot-${colKey}`;
        emptySlot.title = `${COLOR_NAMES[idx]}: 0`;
        stackBody.appendChild(emptySlot);
      } else {
        for (let i = 0; i < count; i++) {
          const chip = document.createElement('div');
          chip.className = `stack-chip token-${colKey}`;
          chip.style.zIndex = i + 1;
          chip.title = `${COLOR_NAMES[idx]}手牌筹码: ${count} 颗 (可支付)`;
          if (options.tokenClickable) {
            chip.style.cursor = 'pointer';
            chip.onclick = () => {
              if (options.onTokenClick) options.onTokenClick(colKey);
            };
          }
          stackBody.appendChild(chip);
        }
      }
      stackCol.appendChild(stackBody);
      tokenContainer.appendChild(stackCol);
    });
  }

  // 已购买卡牌 Tableau (按 5 基础颜色分列，每列自下而上层叠只露顶部卡头，列头整合 +Bonus 减免)
  const purchasedCards = p.purchased_cards || [];
  const purCount = document.getElementById(`${prefix}PurchasedCount`);
  if (purCount) purCount.innerText = purchasedCards.length;
  const purContainer = document.getElementById(`${prefix}PurchasedCards`);
  if (purContainer) renderPurchasedCards(purchasedCards, purContainer, p.bonuses);

  // 预留卡（明牌/暗抽展示，最多 3 张）
  const resCount = document.getElementById(`${prefix}ReservedCount`);
  if (resCount) resCount.innerText = p.reserved_cards.length;
  const resContainer = document.getElementById(`${prefix}ReservedCards`);
  if (resContainer) {
    if (p.reserved_cards.length === 0) {
      resContainer.innerHTML = '<span class="empty-cards-placeholder">无预留</span>';
    } else {
      renderCardsList(p.reserved_cards, resContainer, {
        isCompact: true,
        fromReserved: true,
        interactive: options.interactiveReserved,
        affordableIds: options.affordableReservedIds,
        onPurchase: options.onPurchaseReserved,
        isOpponent: options.isOpponent || false,
        currentPlayerState: p,
      });
    }
  }

  // 已获王室卡 (最多 2 张)
  const royalCards = p.royal_cards || [];
  const royCount = document.getElementById(`${prefix}RoyalCount`);
  if (royCount) royCount.innerText = royalCards.length;
  const royContainer = document.getElementById(`${prefix}RoyalCards`);
  if (royContainer) {
    if (royalCards.length === 0) {
      royContainer.innerHTML = '';
    } else {
      renderPlayerRoyals(royalCards, royContainer);
    }
  }
}
