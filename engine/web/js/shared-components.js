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
        tok.className = `token ${COLOR_CLASSES[gem]}`;
        tok.innerText = gem === 'pearl' ? '珠' : (gem === 'gold' ? '金' : gem[0].toUpperCase());
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
    const el = document.createElement('div');
    el.className = `card-item ${isCompact ? 'compact' : ''}`;
    el.style.backgroundImage = `url('plates/level-${c.tier}-${c.color}.webp')`;

    const canAfford = affordableSet.has(c.id);
    if (canAfford) {
      el.classList.add('affordable');
    }

    const costsHtml = Object.entries(c.cost)
      .filter(([_, amount]) => amount > 0)
      .map(([color, amount]) => `<span class="cost-item ${COLOR_CLASSES[color]}">${amount}</span>`)
      .join('');

    const pointsHtml = c.points > 0 ? `<span class="card-points">${c.points}⭐</span>` : '<span></span>';
    const crownsHtml = c.crowns > 0 ? `<span class="card-crowns">${'👑'.repeat(c.crowns)}</span>` : '';
    const abilityHtml = c.ability ? `<div class="card-ability-badge">${c.ability}</div>` : '';

    el.innerHTML = `
      <div class="card-top">
        ${pointsHtml}
        ${crownsHtml}
      </div>
      ${abilityHtml}
      <div class="card-costs">${costsHtml}</div>
    `;

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
    const el = document.createElement('div');
    el.className = 'card-item mini';
    el.style.backgroundImage = `url('plates/level-${c.tier}-${c.color}.webp')`;

    const bonusText = c.bonus ? `${c.bonus.color}+${c.bonus.amount}` : '';
    const costsDetail = Object.entries(c.cost)
      .filter(([_, amount]) => amount > 0)
      .map(([color, amount]) => `${color}:${amount}`)
      .join(', ');

    el.title = `【卡牌 #${c.id}】L${c.tier} ${c.color}\n声望: ${c.points}⭐ | 王冠: ${c.crowns}👑\n永久加成: ${bonusText || '无'}\n能力: ${c.ability || '无'}\n花费: ${costsDetail || '免费'}`;

    const pointsHtml = c.points > 0 ? `<span class="card-points">${c.points}⭐</span>` : '<span></span>';
    const crownsHtml = c.crowns > 0 ? `<span class="card-crowns">${'👑'.repeat(c.crowns)}</span>` : '';
    const abilityHtml = c.ability ? `<div class="card-ability-badge">${c.ability}</div>` : '';

    el.innerHTML = `
      <div class="card-top">
        ${pointsHtml}
        ${crownsHtml}
      </div>
      ${abilityHtml}
      <div class="card-mini-bottom">
        <span class="mini-tier">L${c.tier}</span>
        ${bonusText ? `<span class="mini-bonus">${bonusText}</span>` : ''}
      </div>
    `;
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
    el.style.backgroundImage = `url('assets/cards/royals/royal-${r.id}.webp')`;
    el.title = `【王室赞助卡 #${r.id}】\n声望: ${r.points}⭐\n能力: ${r.ability || '荣誉赞助'}`;
    el.innerHTML = `
      <div style="font-size:0.92rem; font-weight:800; color:#fff; text-shadow:0 1px 3px #000;">${r.points}⭐</div>
      <div style="font-size:0.58rem; background:rgba(0,0,0,0.75); padding:1px 3px; border-radius:3px; color:#fbbf24; text-align:center;">${r.ability || '赞助'}</div>
    `;
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
    el.style.backgroundImage = `url('assets/cards/royals/royal-${r.id}.webp')`;
    el.innerHTML = `
      <div style="font-size:1.05rem; font-weight:800; color:#fff; text-shadow:0 1px 3px #000;">${r.points}⭐</div>
      <div style="font-size:0.65rem; background:rgba(0,0,0,0.75); padding:2px 4px; border-radius:3px; color:#fbbf24; text-align:center;">${r.ability || '荣誉赞助'}</div>
    `;

    if (options.selectable) {
      el.style.cursor = 'pointer';
      el.style.boxShadow = '0 0 12px #fbbf24';
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
  if (privEl) privEl.innerText = `📜 ${p.privileges}`;

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
        chip.innerText = `${COLOR_NAMES[idx]}: ${count}`;

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
