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

  const tierNames = { 1: '初阶 (L1)', 2: '中阶 (L2)', 3: '高阶 (L3)' };
  el.title = `【${tierNames[tier] || '等级' + tier} 牌库】剩余 ${count} 张${options.canReserve ? ' (点击可盲抽预留)' : ''}`;

  const countBadge = document.createElement('div');
  countBadge.className = 'deck-count-badge';
  countBadge.innerText = `${count} 张`;
  el.appendChild(countBadge);

  if (options.interactive && hasCards && options.canReserve) {
    el.classList.add('deck-clickable');
    const overlay = document.createElement('div');
    overlay.className = 'card-action-overlay';
    const resBtn = document.createElement('button');
    resBtn.className = 'btn-secondary card-action-btn';
    resBtn.innerText = '🎴 盲抽预留';
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

  // 如果提供了牌堆信息，首位放置对应的等级牌库卡背元素
  if (options.deckInfo) {
    const deckEl = createDeckPileElement(options.deckInfo.tier, options.deckInfo.count, {
      interactive: options.interactive,
      canReserve: options.canReserve,
      onReserveDeck: options.onReserveDeck,
    });
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
        const colKey = COLOR_KEYS[idx];
        chip.className = `chip ${COLOR_CLASSES[colKey]}`;
        chip.innerHTML = `<span class="chip-token-icon token-${colKey}"></span> <span><b>${COLOR_NAMES[idx]}</b>: +${count}</span>`;
        chip.title = `${COLOR_NAMES[idx]}宝石永久减免 +${count}`;
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
        onPurchase: options.onPurchaseReserved,
        isOpponent: options.isOpponent || false,
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
