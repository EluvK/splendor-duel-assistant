(function initCardArt(root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.CardArt = factory();
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function createCardArt() {
  const LEVEL_NAME_TO_NUMBER = {
    One: 1,
    Two: 2,
    Three: 3,
  };

  const GEM_COLORS = ['white', 'blue', 'green', 'red', 'black'];

  function cardLevelNumber(level) {
    if (typeof level === 'number' && [1, 2, 3].includes(level)) return level;
    if (typeof level === 'string' && LEVEL_NAME_TO_NUMBER[level]) {
      return LEVEL_NAME_TO_NUMBER[level];
    }
    throw new Error(`Unsupported card level: ${level}`);
  }

  function normalizedBonusColor(card) {
    const bonus = card && Array.isArray(card.bonuses) ? card.bonuses[0] : null;
    if (!bonus) return 'joker';
    return String(bonus).toLowerCase();
  }

  function developmentPlatePath(card) {
    const level = cardLevelNumber(card.level);
    const color = normalizedBonusColor(card);
    return `assets/cards/plates/level-${level}-${color}.webp`;
  }

  function royalPlatePath(royal) {
    return `assets/cards/royals/royal-${royal.id}.webp`;
  }

  function requiredDevelopmentPlatePaths() {
    const paths = [];
    for (const level of [1, 2, 3]) {
      for (const color of [...GEM_COLORS, 'joker']) {
        paths.push(`assets/cards/plates/level-${level}-${color}.webp`);
      }
    }
    return paths;
  }

  function requiredRoyalPlatePaths() {
    return [0, 1, 2, 3].map((id) => `assets/cards/royals/royal-${id}.webp`);
  }

  return {
    cardLevelNumber,
    developmentPlatePath,
    royalPlatePath,
    requiredDevelopmentPlatePaths,
    requiredRoyalPlatePaths,
  };
});
