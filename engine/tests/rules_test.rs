use _engine::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

#[test]
fn test_joker_purchase_rules() {
    let mut game = GameState::new_game(100);
    game.phase = TurnPhase::MandatoryAction;

    // 找到一张 Joker 卡
    let joker_card = ALL_JEWEL_CARDS
        .iter()
        .find(|c| c.color == CardColor::Joker)
        .copied()
        .unwrap();

    // 1. 当玩家没有任何 bonus 时，禁止购买 Joker
    game.players[0].bonuses = [0; 5];
    game.players[0].tokens.add(GemType::Gold, 10); // 赋予大量黄金确保能付得起
    game.pyramid[joker_card.tier.index()][0] = joker_card;

    let legal = RuleEngine::legal_actions(&game);
    let can_buy_joker = legal.iter().any(|a| match a {
        Action::PurchaseCard {
            from_reserved: false,
            tier,
            slot: 0,
        } => *tier == joker_card.tier,
        _ => false,
    });
    assert!(!can_buy_joker, "玩家无 bonus 时不应生成购买 Joker 的合法动作");

    // 2. 赋予玩家 1 个蓝色 bonus，现在应该允许购买
    game.players[0].bonuses[GemType::Blue.index()] = 1;
    let legal = RuleEngine::legal_actions(&game);
    let can_buy_joker = legal.iter().any(|a| match a {
        Action::PurchaseCard {
            from_reserved: false,
            tier,
            slot: 0,
        } => *tier == joker_card.tier,
        _ => false,
    });
    assert!(can_buy_joker, "玩家拥有 bonus 时应允许购买 Joker");

    // 3. 购买 Joker，进入 CardAbilityJoker 阶段
    let buy_action = Action::PurchaseCard {
        from_reserved: false,
        tier: joker_card.tier,
        slot: 0,
    };
    assert!(GameEngine::step(&mut game, &buy_action).is_ok());
    assert!(matches!(game.phase, TurnPhase::CardAbilityJoker { .. }));

    // 4. 只能附着到已有 bonus 的颜色（即 Blue），不能附着到其他颜色
    let legal_colors = RuleEngine::legal_actions(&game);
    assert_eq!(legal_colors.len(), 1);
    assert_eq!(
        legal_colors[0],
        Action::AssignJokerColor {
            color: GemType::Blue
        }
    );

    // 5. 执行附着，蓝色 bonus 变为 1 + joker_bonus
    let blue_bonus_before = game.players[0].bonuses[GemType::Blue.index()];
    assert!(GameEngine::step(&mut game, &legal_colors[0]).is_ok());
    assert_eq!(
        game.players[0].bonuses[GemType::Blue.index()],
        blue_bonus_before + joker_card.bonus
    );
}

#[test]
fn test_privilege_scroll_exhaustion_steal() {
    let mut game = GameState::new_game(200);

    // 设置公用特权池为 0，对手拥有 2 个特权
    game.privilege_pool = 0;
    game.players[1].privileges = 2;
    game.players[0].privileges = 0;

    // 当 Player 0 触发获得特权时，应直接从对手处偷取 1 个
    game.grant_privilege_to(0);

    assert_eq!(game.players[0].privileges, 1);
    assert_eq!(game.players[1].privileges, 1);
    assert_eq!(game.privilege_pool, 0);
}

#[test]
fn test_reserve_card_without_gold_on_board() {
    let mut game = GameState::new_game(300);
    game.phase = TurnPhase::MandatoryAction;

    // 清空棋盘上的所有黄金
    for r in 0..5 {
        for c in 0..5 {
            if game.board.get(r, c) == Some(GemType::Gold) {
                game.board.take(r, c);
            }
        }
    }
    assert!(!game.board.has_gold());

    let p0_gold_before = game.players[0].tokens.get(GemType::Gold);
    let p0_reserved_before = game.players[0].reserved_cards.len();

    // 即使棋盘无黄金，只要预留卡未满 3 张，仍然可以合法预留
    let reserve_action = Action::ReserveCard {
        tier: CardTier::Tier1,
        slot: Some(0),
    };
    assert!(GameEngine::step(&mut game, &reserve_action).is_ok());

    assert_eq!(
        game.players[0].tokens.get(GemType::Gold),
        p0_gold_before,
        "无黄金时预留不应增加黄金"
    );
    assert_eq!(
        game.players[0].reserved_cards.len(),
        p0_reserved_before + 1,
        "卡牌应成功进入预留手牌"
    );
}

#[test]
fn test_hand_limit_discard() {
    let mut game = GameState::new_game(400);
    game.phase = TurnPhase::MandatoryAction;

    // 赋予当前玩家 11 枚标记
    game.players[0].tokens.set(GemType::White, 6);
    game.players[0].tokens.set(GemType::Blue, 5);
    assert_eq!(game.players[0].tokens.total(), 11);

    // 构造一个合法的 1 标记拿取动作
    game.board.set(0, 0, Some(GemType::Red));
    let take_action = Action::TakeTokens {
        count: 1,
        positions: [(0, 0), (0, 0), (0, 0)],
    };
    assert!(GameEngine::step(&mut game, &take_action).is_ok());

    // 手牌达到 12 枚，应进入 DiscardTokens 阶段，且未换人
    assert_eq!(game.players[0].tokens.total(), 12);
    assert_eq!(game.phase, TurnPhase::DiscardTokens);
    assert_eq!(game.current_player, 0);

    // 弃掉 1 枚白标记
    let discard1 = Action::DiscardToken {
        gem: GemType::White,
    };
    assert!(GameEngine::step(&mut game, &discard1).is_ok());
    // 依然有 11 枚，仍应在 DiscardTokens
    assert_eq!(game.phase, TurnPhase::DiscardTokens);

    // 再弃掉 1 枚白标记，达到 10 枚
    let discard2 = Action::DiscardToken {
        gem: GemType::White,
    };
    assert!(GameEngine::step(&mut game, &discard2).is_ok());

    // 弃牌完成，应结束回合轮到对手
    assert_eq!(game.players[0].tokens.total(), 10);
    assert_eq!(game.current_player, 1);
    assert_eq!(game.phase, TurnPhase::OptionalActions);
}

#[test]
fn test_single_color_ten_points_victory() {
    let mut game = GameState::new_game(500);
    // 人工赋予 10 点单色声望
    game.players[0].color_points[GemType::Red.index()] = 10;
    assert_eq!(
        check_victory(&game.players[0]),
        Some(VictoryReason::TenPointsSameColor(GemType::Red))
    );
}

#[test]
fn test_parallel_selfplay_stress() {
    // 并行运行 1000 局自博弈压力测试
    let total_games = 1000;
    let results: Vec<_> = (0..total_games)
        .into_par_iter()
        .map(|seed| {
            let mut game = GameState::new_game(seed);
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut step_count = 0;

            while !matches!(game.phase, TurnPhase::GameOver(_)) {
                step_count += 1;
                if step_count > 2000 {
                    panic!("Seed {seed} 超过 2000 步未结束！");
                }

                let action = RandomAI::select_action(&game, &mut rng)
                    .unwrap_or_else(|| panic!("Seed {seed} 在阶段 {:?} 无合法动作！", game.phase));
                GameEngine::step(&mut game, &action)
                    .unwrap_or_else(|e| panic!("Seed {seed} 执行动作失败: {e}"));
            }

            let (winner, reason) = game.winner.unwrap();
            (winner, reason, step_count)
        })
        .collect();

    assert_eq!(results.len(), total_games as usize);
    let avg_steps: f64 = results.iter().map(|(_, _, s)| *s as f64).sum::<f64>() / total_games as f64;
    let mut reason_20_pts = 0;
    let mut reason_10_crowns = 0;
    let mut reason_single_color = 0;

    for (_, reason, _) in results.iter() {
        match reason {
            VictoryReason::TwentyPrestigePoints => reason_20_pts += 1,
            VictoryReason::TenCrowns => reason_10_crowns += 1,
            VictoryReason::TenPointsSameColor(_) => reason_single_color += 1,
        }
    }

    println!("\n=== 1000 局自博弈压力测试全部通过！===");
    println!("平均步数: {avg_steps:.1}");
    println!("20声望获胜局数: {reason_20_pts} (占比 {:.1}%)", reason_20_pts as f64 / 10.0);
    println!("10王冠获胜局数: {reason_10_crowns} (占比 {:.1}%)", reason_10_crowns as f64 / 10.0);
    println!("单色10分获胜局数: {reason_single_color} (占比 {:.1}%)", reason_single_color as f64 / 10.0);
}
