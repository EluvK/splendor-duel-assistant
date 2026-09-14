use _engine::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

#[test]
fn test_encode_state_bounds_and_nan() {
    // 运行 20 局随机对弈，每一步均检验编码张量的正确性
    for seed in 0..20 {
        let mut game = GameState::new_game(seed);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        while !matches!(game.phase, TurnPhase::GameOver(_)) {
            let obs = encode_state(&game);
            assert_eq!(obs.len(), OBS_SIZE);

            for (idx, &val) in obs.iter().enumerate() {
                assert!(!val.is_nan(), "Seed {seed} 特征索引 {idx} 出现 NaN!");
                assert!(!val.is_infinite(), "Seed {seed} 特征索引 {idx} 出现 Inf!");
                assert!(
                    (0.0..=1.0001).contains(&val),
                    "Seed {seed} 特征索引 {idx} 越界: {val}"
                );
            }

            if let Some(action) = RandomAI::select_action(&game, &mut rng) {
                let _ = GameEngine::step(&mut game, &action);
            } else {
                break;
            }
        }
    }
}

#[test]
fn test_action_mask_consistency() {
    let mut game = GameState::new_game(42);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    for _ in 0..100 {
        if matches!(game.phase, TurnPhase::GameOver(_)) {
            break;
        }

        let legals = RuleEngine::legal_actions(&game);
        let mask = action_mask(&game);

        // 验证每个合法动作对应的 ID 在 mask 中必为 true
        for act in legals.iter() {
            let id = action_to_id(act);
            assert!(id < ACTION_SIZE, "动作 ID {id} 超过动作空间上限 {ACTION_SIZE}!");
            assert!(mask[id], "合法动作 {:?} 对应的 ID {id} 在 mask 中为 false!", act);
        }

        // 验证 mask 中为 true 的总数至少等于去重后的合法动作 ID 数量
        let mask_true_count = mask.iter().filter(|&&b| b).count();
        assert!(mask_true_count >= 1, "非终局状态下 action mask 不能全为 false!");

        if let Some(act) = RandomAI::select_action(&game, &mut rng) {
            let _ = GameEngine::step(&mut game, &act);
        } else {
            break;
        }
    }
}

#[test]
fn test_reserved_cards_anti_leakage_and_encoding() {
    let mut game = GameState::new_game(123);
    game.phase = TurnPhase::MandatoryAction;

    // 清空两名玩家的预留牌
    game.players[0].reserved_cards.clear();
    game.players[1].reserved_cards.clear();

    // 构造测试卡牌
    let secret_card = JewelCard {
        id: 10,
        tier: CardTier::Tier1,
        color: CardColor::Red,
        points: 2,
        bonus: 1,
        ability: None,
        crowns: 1,
        cost: CardCost::new(0, 0, 0, 0, 0, 0), // 免费，买得起
    };

    let public_card = JewelCard {
        id: 20,
        tier: CardTier::Tier2,
        color: CardColor::Blue,
        points: 3,
        bonus: 1,
        ability: None,
        crowns: 2,
        cost: CardCost::new(5, 5, 5, 5, 5, 2), // 昂贵，买不起
    };

    // P0 (当前行动方) 拥有一张盲抽暗牌 (is_public: false)
    game.players[0].reserved_cards.push(ReservedCard::new(secret_card, false));

    // P1 (对手) 拥有一张盲抽暗牌 (is_public: false) 与一张公开预留牌 (is_public: true)
    game.players[1].reserved_cards.push(ReservedCard::new(secret_card, false));
    game.players[1].reserved_cards.push(ReservedCard::new(public_card, true));

    game.current_player = 0;
    let obs = encode_state(&game);

    // P0 (自己) 槽位 0: [649..655]
    // 应该编码: present=1.0, is_public=0.0 (因为是暗抽), points=2/6, crowns=1/3, color=Red(3)/5, can_afford=1.0
    let p0_slot0 = &obs[649..655];
    assert_eq!(p0_slot0[0], 1.0, "P0 槽位 0 present 应为 1.0");
    assert_eq!(p0_slot0[1], 0.0, "P0 槽位 0 暗抽 is_public 应为 0.0");
    assert!((p0_slot0[2] - 2.0 / 6.0).abs() < 1e-5, "P0 槽位 0 点数自己可见");
    assert!((p0_slot0[3] - 1.0 / 3.0).abs() < 1e-5, "P0 槽位 0 皇冠自己可见");
    assert!((p0_slot0[4] - 3.0 / 5.0).abs() < 1e-5, "P0 槽位 0 颜色自己可见");
    assert_eq!(p0_slot0[5], 1.0, "P0 槽位 0 can_afford 自己可见且能买得起");

    // P1 (对手) 槽位 0 (盲抽暗牌): [691..697]
    // 必须掩蔽所有私有信息！
    // present=1.0, is_public=0.0, 其余字段全部为 0.0
    let p1_slot0 = &obs[691..697];
    assert_eq!(p1_slot0[0], 1.0, "P1 槽位 0 present 应为 1.0");
    assert_eq!(p1_slot0[1], 0.0, "P1 槽位 0 暗抽 is_public 应为 0.0");
    assert_eq!(p1_slot0[2], 0.0, "P1 暗抽卡点数对 P0 必须完全保密 (0.0)");
    assert_eq!(p1_slot0[3], 0.0, "P1 暗抽卡皇冠对 P0 必须完全保密 (0.0)");
    assert_eq!(p1_slot0[4], 0.0, "P1 暗抽卡颜色对 P0 必须完全保密 (0.0)");
    assert_eq!(p1_slot0[5], 0.0, "P1 暗抽卡支付能力对 P0 必须完全保密 (0.0)");

    // P1 (对手) 槽位 1 (金字塔明牌公开预留): [697..703]
    // present=1.0, is_public=1.0, points=3/6, crowns=2/3, color=Blue(1)/5, can_afford=0.0 (买不起)
    let p1_slot1 = &obs[697..703];
    assert_eq!(p1_slot1[0], 1.0, "P1 槽位 1 present 应为 1.0");
    assert_eq!(p1_slot1[1], 1.0, "P1 槽位 1 明牌预留 is_public 应为 1.0");
    assert!((p1_slot1[2] - 3.0 / 6.0).abs() < 1e-5, "P1 公开牌点数双方可见");
    assert!((p1_slot1[3] - 2.0 / 3.0).abs() < 1e-5, "P1 公开牌皇冠双方可见");
    assert!((p1_slot1[4] - 1.0 / 5.0).abs() < 1e-5, "P1 公开牌颜色双方可见");
    assert_eq!(p1_slot1[5], 0.0, "P1 买不起该昂贵牌，can_afford 应为 0.0 (且反映的是对手支付能力)");

    // 特征 32: op_has_winning_purchase [741]
    // P1 拥有一张免费直接斩杀卡 (secret_card 若为 20 分)，但由于是暗抽，P0 绝不能知晓！
    let mut lethal_secret_card = secret_card;
    lethal_secret_card.points = 20; // 满足 20 分胜负条件
    game.players[1].reserved_cards[0] = ReservedCard::new(lethal_secret_card, false);
    // 清空金字塔卡牌以排除场面干扰
    for row in game.pyramid.iter_mut() {
        row.clear();
    }
    let obs_secret = encode_state(&game);
    assert_eq!(
        obs_secret[709 + 32],
        0.0,
        "对手暗抽即使持有斩杀牌，特征 32 也必须为 0.0 (严禁 POMDP 私密信息泄露)"
    );

    // 反之，若该斩杀牌公开 (is_public: true)，特征 32 应感知到对手斩杀威胁 (1.0)
    game.players[1].reserved_cards[0] = ReservedCard::new(lethal_secret_card, true);
    let obs_public = encode_state(&game);
    assert_eq!(
        obs_public[709 + 32],
        1.0,
        "对手若持有公开可买的斩杀牌，特征 32 应为 1.0"
    );
}

