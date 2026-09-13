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
  gold: 'token-gold'
};

export const COLOR_KEYS = ['white', 'blue', 'green', 'red', 'black', 'pearl', 'gold'];
export const COLOR_NAMES = ['白', '蓝', '绿', '红', '黑', '珍珠', '黄金'];

export const ABILITY_DISPLAY_NAMES = {
  ExtraTurn: '额外回合 🔁',
  TakePrivilege: '拿特权 📜',
  TakeSameColor: '拿同色 💎',
  StealToken: '偷标记 🤏',
  ColorCopy: '变色绑定 🎨',
  ColorCopyAndExtraTurn: '变色+回合 🔁',
};

export const ROYAL_ABILITY_DISPLAY_NAMES = {
  ExtraTurn: '额外回合 🔁',
  TakePrivilege: '拿特权 📜',
  StealToken: '偷标记 🤏',
};

/**
 * 为卡牌元素配置 BGA 原版雪碧图 (Sprite Sheet) 样式
 * cards1.jpg: 31 帧 (0..29 为 Tier1 卡牌, 30 为卡背)
 * cards2.jpg: 25 帧 (0..23 为 Tier2 卡牌, 24 为卡背)
 * cards3.jpg: 14 帧 (0..12 为 Tier3 卡牌, 13 为卡背)
 * @param {HTMLElement} el
 * @param {Object} card 卡牌 DTO
 */
export function applyBgaCardSprite(el, card) {
  if (!card) return;
  const tier = card.tier || 1;
  let sheetUrl, colIndex, totalCols;

  if (tier === 1) {
    sheetUrl = 'assets/images/cards1.jpg';
    colIndex = Math.max(0, Math.min(29, card.id));
    totalCols = 31;
  } else if (tier === 2) {
    sheetUrl = 'assets/images/cards2.jpg';
    colIndex = Math.max(0, Math.min(23, card.id - 30));
    totalCols = 25;
  } else {
    sheetUrl = 'assets/images/cards3.jpg';
    colIndex = Math.max(0, Math.min(12, card.id - 54));
    totalCols = 14;
  }

  const posX = (colIndex / (totalCols - 1)) * 100;
  el.style.backgroundImage = `url('${sheetUrl}')`;
  el.style.backgroundSize = `${totalCols * 100}% 100%`;
  el.style.backgroundPosition = `${posX.toFixed(4)}% 0%`;
  el.style.backgroundRepeat = 'no-repeat';
}

/**
 * 为牌堆元素配置 BGA 卡背原画
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
  el.style.backgroundImage = `url('${sheetUrl}')`;
  el.style.backgroundSize = `${totalCols * 100}% 100%`;
  el.style.backgroundPosition = `100% 0%`;
  el.style.backgroundRepeat = 'no-repeat';
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
 * 兼容性辅助函数
 */
export function getCardPlatePath(card) {
  if (!card) return 'assets/cards/plates/level-1-joker.webp';
  const tier = card.tier || 1;
  const rawColor = (card.color || '').toLowerCase();
  const validColors = ['white', 'blue', 'green', 'red', 'black'];
  const color = validColors.includes(rawColor) ? rawColor : 'joker';
  return `assets/cards/plates/level-${tier}-${color}.webp`;
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

  let countTokens = 0;
  for (let r = 0; r < 5; r++) {
    for (let c = 0; c < 5; c++) {
      const cell = document.createElement('div');
      cell.className = 'board-cell';
      const key = `${r},${c}`;

      if (selectedSet.has(key)) cell.classList.add('cell-selected');
      if (candidateSet.has(key)) cell.classList.add('cell-candidate');
      if (highlightSet.has(key)) cell.classList.add('cell-highlight');

      const gem = boardData[r][c];
      if (gem) {
        countTokens++;
        const tok = document.createElement('div');
        tok.className = `token token-${gem}`;
        const colIdx = COLOR_KEYS.indexOf(gem);
        tok.title = `${colIdx >= 0 ? COLOR_NAMES[colIdx] : gem}标记 (${r}, ${c})`;
        cell.appendChild(tok);

        if (options.clickable) {
          cell.classList.add('cell-clickable');
          cell.onclick = () => {
            if (options.onCellClick) options.onCellClick(r, c, gem);
          };
        }
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
  if (!cards || cards.length === 0) {
    if (options.emptyText) {
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
    applyBgaCardSprite(el, c);

    const canAfford = affordableSet.has(c.id);
    if (canAfford) {
      el.classList.add('affordable');
    }

    const costsDetail = Object.entries(c.cost)
      .filter(([_, amount]) => amount > 0)
      .map(([color, amount]) => `${color}:${amount}`)
      .join(', ');
    const abilityLabel = c.ability ? (ABILITY_DISPLAY_NAMES[c.ability] || c.ability) : '';
    const bonusText = c.bonus > 0 && c.color !== 'joker' && c.color !== 'points' ? `${c.color}+${c.bonus}` : '';

    el.title = `【卡牌 #${c.id}】L${c.tier} ${c.color}\n声望: ${c.points}⭐ | 王冠: ${c.crowns}👑\n永久加成: ${bonusText || '无'}\n能力: ${abilityLabel || '无'}\n花费: ${costsDetail || '免费'}`;
    el.innerHTML = '';

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
        resBtn.className = 'btn-secondary card-action-btn';
        resBtn.innerText = '📌 预留卡牌';
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
 */
export function renderPurchasedCards(cards, container) {
  container.innerHTML = '';
  if (!cards || cards.length === 0) {
    container.innerHTML = '<span style="font-size:0.7rem; color:var(--text-muted); padding:2px;">暂无已购卡牌</span>';
    return;
  }

  cards.forEach(c => {
    if (!c) return;
    const el = document.createElement('div');
    el.className = 'card-item mini';
    applyBgaCardSprite(el, c);

    const bonusText = c.bonus > 0 && c.color !== 'joker' && c.color !== 'points' ? `${c.color}+${c.bonus}` : '';
    const costsDetail = Object.entries(c.cost)
      .filter(([_, amount]) => amount > 0)
      .map(([color, amount]) => `${color}:${amount}`)
      .join(', ');
    const abilityLabel = c.ability ? (ABILITY_DISPLAY_NAMES[c.ability] || c.ability) : '';

    el.title = `【卡牌 #${c.id}】L${c.tier} ${c.color}\n声望: ${c.points}⭐ | 王冠: ${c.crowns}👑\n永久加成: ${bonusText || '无'}\n能力: ${abilityLabel || '无'}\n花费: ${costsDetail || '免费'}`;
    el.innerHTML = '';
    container.appendChild(el);
  });
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

    if (options.selectable) {
      el.style.cursor = 'pointer';
      el.style.boxShadow = '0 0 16px #fbbf24';
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

  const privEl = document.getElementById(`${prefix}Privileges`);
  if (privEl) privEl.innerHTML = `<span class="privilege-icon"></span> ${p.privileges}`;

  const ptsEl = document.getElementById(`${prefix}Points`);
  if (ptsEl) ptsEl.innerText = p.total_points;
  const ptsBar = document.getElementById(`${prefix}PointsBar`);
  if (ptsBar) ptsBar.style.width = `${Math.min(100, (p.total_points / 20) * 100)}%`;

  const crnEl = document.getElementById(`${prefix}Crowns`);
  if (crnEl) crnEl.innerText = p.total_crowns;
  const crnBar = document.getElementById(`${prefix}CrownsBar`);
  if (crnBar) crnBar.style.width = `${Math.min(100, (p.total_crowns / 10) * 100)}%`;

  const maxColor = Math.max(...p.color_points);
  const colEl = document.getElementById(`${prefix}MaxColor`);
  if (colEl) colEl.innerText = maxColor;
  const colBar = document.getElementById(`${prefix}ColorBar`);
  if (colBar) colBar.style.width = `${Math.min(100, (maxColor / 10) * 100)}%`;

  const tokTotal = document.getElementById(`${prefix}TokenTotal`);
  if (tokTotal) tokTotal.innerText = p.token_total;

  // 手牌标记 Chips
  const tokenContainer = document.getElementById(`${prefix}Tokens`);
  if (tokenContainer) {
    tokenContainer.innerHTML = '';
    let hasTokens = false;
    p.tokens.forEach((count, idx) => {
      if (count > 0) {
        hasTokens = true;
        const chip = document.createElement('span');
        const colKey = COLOR_KEYS[idx];
        chip.className = `chip ${COLOR_CLASSES[colKey]}`;
        chip.innerHTML = `<span class="chip-token-icon token-${colKey}"></span> <span>${COLOR_NAMES[idx]}: ${count}</span>`;

        // 交互：如果处于弃牌或偷牌模式
        if (options.tokenClickable) {
          chip.style.cursor = 'pointer';
          chip.style.boxShadow = '0 0 6px #fff';
          chip.onclick = () => {
            if (options.onTokenClick) options.onTokenClick(colKey);
          };
        }

        tokenContainer.appendChild(chip);
      }
    });
    if (!hasTokens) {
      tokenContainer.innerHTML = '<span style="font-size:0.7rem; color:var(--text-muted);">暂无标记</span>';
    }
  }

  // Bonuses 永久减免
  const bonusContainer = document.getElementById(`${prefix}Bonuses`);
  if (bonusContainer) {
    bonusContainer.innerHTML = '';
    let hasBonus = false;
    p.bonuses.forEach((count, idx) => {
      if (count > 0) {
        hasBonus = true;
        const chip = document.createElement('span');
        chip.className = `chip ${COLOR_CLASSES[COLOR_KEYS[idx]]}`;
        chip.innerText = `${COLOR_NAMES[idx]}: +${count}`;
        bonusContainer.appendChild(chip);
      }
    });
    if (!hasBonus) {
      bonusContainer.innerHTML = '<span style="font-size:0.7rem; color:var(--text-muted);">暂无减免</span>';
    }
  }

  // 预留卡（明牌展示）
  const resCount = document.getElementById(`${prefix}ReservedCount`);
  if (resCount) resCount.innerText = p.reserved_cards.length;
  const resContainer = document.getElementById(`${prefix}ReservedCards`);
  if (resContainer) {
    if (p.reserved_cards.length === 0) {
      resContainer.innerHTML = '<span style="font-size:0.7rem; color:var(--text-muted);">暂无预留卡</span>';
    } else {
      renderCardsList(p.reserved_cards, resContainer, {
        isCompact: true,
        fromReserved: true,
        interactive: options.interactiveReserved,
        affordableIds: options.affordableReservedIds,
        onPurchase: options.onPurchaseReserved
      });
    }
  }

  // 已获王室卡
  const royalCards = p.royal_cards || [];
  const royCount = document.getElementById(`${prefix}RoyalCount`);
  if (royCount) royCount.innerText = royalCards.length;
  const royContainer = document.getElementById(`${prefix}RoyalCards`);
  if (royContainer) renderPlayerRoyals(royalCards, royContainer);

  // 已购买卡牌 Tableau
  const purchasedCards = p.purchased_cards || [];
  const purCount = document.getElementById(`${prefix}PurchasedCount`);
  if (purCount) purCount.innerText = purchasedCards.length;
  const purContainer = document.getElementById(`${prefix}PurchasedCards`);
  if (purContainer) renderPurchasedCards(purchasedCards, purContainer);
}
