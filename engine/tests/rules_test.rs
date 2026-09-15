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
fn test_cannot_reserve_card_without_gold_on_board() {
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

    // 规则限制：当棋盘无黄金时，不应生成任何预留卡牌行动
    let legal_actions = RuleEngine::legal_actions(&game);
    let has_reserve = legal_actions.iter().any(|a| matches!(a, Action::ReserveCard { .. }));
    assert!(!has_reserve, "棋盘无黄金时不能生成预留卡牌动作");

    // 若强行执行预留动作，引擎应拒绝报错
    let reserve_action = Action::ReserveCard {
        tier: CardTier::Tier1,
        slot: Some(0),
    };
    let res = GameEngine::step(&mut game, &reserve_action);
    assert!(res.is_err(), "棋盘无黄金时强行预留应返回错误");

    assert_eq!(
        game.players[0].tokens.get(GemType::Gold),
        p0_gold_before,
        "失败的预留不应增加黄金"
    );
    assert_eq!(
        game.players[0].reserved_cards.len(),
        p0_reserved_before,
        "失败的预留不应进入预留手牌"
    );
}

#[test]
fn test_reserve_card_with_gold_selection() {
    let mut game = GameState::new_game(42);
    game.phase = TurnPhase::MandatoryAction;

    // 确保盘上有黄金
    assert!(game.board.has_gold());
    let mut gold_coords = Vec::new();
    for r in 0..5 {
        for c in 0..5 {
            if game.board.get(r, c) == Some(GemType::Gold) {
                gold_coords.push((r, c));
            }
        }
    }
    assert!(!gold_coords.is_empty(), "标准开局应存在黄金");

    let p0_gold_before = game.players[0].tokens.get(GemType::Gold);
    let p0_reserved_before = game.players[0].reserved_cards.len();

    // 执行预留
    let reserve_action = Action::ReserveCard {
        tier: CardTier::Tier1,
        slot: Some(0),
    };
    assert!(GameEngine::step(&mut game, &reserve_action).is_ok());

    // 状态应转移到 SelectReserveGold，且尚未增加黄金
    assert_eq!(game.phase, TurnPhase::SelectReserveGold);
    assert_eq!(game.players[0].reserved_cards.len(), p0_reserved_before + 1);
    assert_eq!(game.players[0].tokens.get(GemType::Gold), p0_gold_before);

    // 验证合法动作：必须全部为 TakeGoldToken 且与盘上黄金坐标一一对应
    let legals = RuleEngine::legal_actions(&game);
    assert_eq!(legals.len(), gold_coords.len());
    for act in &legals {
        match act {
            Action::TakeGoldToken { r, c } => {
                assert!(gold_coords.contains(&(*r, *c)));
            }
            _ => panic!("SelectReserveGold 阶段不应出现非 TakeGoldToken 动作: {:?}", act),
        }
    }

    // 玩家自主挑选第一个黄金执行
    let chosen_coord = gold_coords[0];
    let take_gold_action = Action::TakeGoldToken {
        r: chosen_coord.0,
        c: chosen_coord.1,
    };
    assert!(GameEngine::step(&mut game, &take_gold_action).is_ok());

    // 验证所选黄金被取走，玩家黄金 +1，且回合完成推进
    assert_eq!(game.board.get(chosen_coord.0, chosen_coord.1), None);
    assert_eq!(
        game.players[0].tokens.get(GemType::Gold),
        p0_gold_before + 1
    );
    assert!(
        game.players[0].reserved_cards.last().unwrap().is_public,
        "从金字塔明牌预留应标记为公开"
    );
}

#[test]
fn test_blind_reserve_card_is_private() {
    let mut game = GameState::new_game(99);
    game.phase = TurnPhase::MandatoryAction;

    // 确保棋盘有黄金以满足预留前提
    assert!(game.board.has_gold());

    let deck_len_before = game.decks[0].len();
    assert!(deck_len_before > 0);

    // 从 Tier1 牌堆顶盲抽预留 (slot: None)
    let blind_reserve = Action::ReserveCard {
        tier: CardTier::Tier1,
        slot: None,
    };
    assert!(GameEngine::step(&mut game, &blind_reserve).is_ok());
    assert_eq!(game.phase, TurnPhase::SelectReserveGold);

    assert_eq!(game.decks[0].len(), deck_len_before - 1);
    let reserved = game.players[0].reserved_cards.last().unwrap();
    assert!(
        !reserved.is_public,
        "从牌堆顶盲抽预留卡牌必须标记为非公开私有暗牌 (is_public == false)"
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

#[test]
fn test_heuristic_selfplay_sanity() {
    // 验证启发式 AI 在 100 局自博弈中均能在合理步数内终结且绝不死锁
    for seed in 0..100 {
        let mut game = GameState::new_game(seed);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut steps = 0;
        while !matches!(game.phase, TurnPhase::GameOver(_)) {
            steps += 1;
            if steps > 300 {
                panic!("Seed {seed} 异常超过 300 步！可能发生死循环");
            }
            let action = HeuristicAI::select_action(&game, &mut rng)
                .unwrap_or_else(|| panic!("Seed {seed} 无合法动作"));
            let _ = GameEngine::step(&mut game, &action);
        }

        assert!(
            game.winner.is_some(),
            "Seed {seed} 终局时必须产生明确获胜者!"
        );
        let (winner, reason) = game.winner.unwrap();
        assert!(
            winner == 0 || winner == 1,
            "Seed {seed} 获胜者玩家索引必须为 0 或 1!"
        );
        assert!(
            matches!(
                game.phase,
                TurnPhase::GameOver(r) if r == reason
            ),
            "Seed {seed} phase 中的胜负原因与 winner 记录必须一致!"
        );
    }
}

#[test]
fn test_replenish_board_rng_entropy_and_determinism() {
    // 1. 同一初始种子推演两次 ReplenishBoard 产生严格相同结果 (确定性与可复现性)
    let mut game1 = GameState::new_game(42);
    let mut game2 = GameState::new_game(42);

    // 拿走盘上一部分标记以便能够执行 ReplenishBoard
    for r in 0..2 {
        for c in 0..5 {
            if let Some(gem) = game1.board.take(r, c) {
                game1.bag.push(gem);
            }
            if let Some(gem) = game2.board.take(r, c) {
                game2.bag.push(gem);
            }
        }
    }

    GameEngine::step(&mut game1, &Action::ReplenishBoard).unwrap();
    GameEngine::step(&mut game2, &Action::ReplenishBoard).unwrap();

    assert_eq!(
        game1.board, game2.board,
        "相同 seed 的游戏执行 ReplenishBoard 必须得到完全相同的盘面!"
    );
    assert_eq!(game1.rng_counter, 1);
    assert_eq!(game2.rng_counter, 1);

    // 2. 具有相同 turn_number 和 bag 长度、但初始种子不同的两个游戏，ReplenishBoard 应产生不同盘面 (高熵，消除伪随机碰撞)
    let mut game_a = GameState::new_game(100);
    let mut game_b = GameState::new_game(999);
    for r in 0..2 {
        for c in 0..5 {
            if let Some(gem) = game_a.board.take(r, c) {
                game_a.bag.push(gem);
            }
            if let Some(gem) = game_b.board.take(r, c) {
                game_b.bag.push(gem);
            }
        }
    }
    assert_eq!(game_a.turn_number, game_b.turn_number);
    assert_eq!(game_a.bag.len(), game_b.bag.len());

    GameEngine::step(&mut game_a, &Action::ReplenishBoard).unwrap();
    GameEngine::step(&mut game_b, &Action::ReplenishBoard).unwrap();

    assert_ne!(
        game_a.board, game_b.board,
        "不同 seed 的游戏在相同回合与袋子余量下执行 ReplenishBoard 不应产生相同的排列!"
    );
}

#[test]
fn test_determinize_for_player_conservation_and_privacy() {
    let mut game = GameState::new_game(12345);
    let mut rng = ChaCha8Rng::seed_from_u64(999);

    // 构造对局局面：
    // Player 0 (观察者): 拥有 1 张公开预留卡和 1 张盲抽暗牌
    // Player 1 (对手): 拥有 1 张公开预留卡和 2 张盲抽暗牌 (分别来自 Tier1 和 Tier2)
    let opp_blind_t1 = game.decks[0].pop().unwrap();
    let opp_blind_t2 = game.decks[1].pop().unwrap();
    let opp_public_t2 = game.decks[1].pop().unwrap();

    let obs_blind_t1 = game.decks[0].pop().unwrap();
    let obs_public_t1 = game.decks[0].pop().unwrap();

    game.players[1].reserved_cards.push(ReservedCard::new(opp_blind_t1, false));
    game.players[1].reserved_cards.push(ReservedCard::new(opp_blind_t2, false));
    game.players[1].reserved_cards.push(ReservedCard::new(opp_public_t2, true));

    game.players[0].reserved_cards.push(ReservedCard::new(obs_blind_t1, false));
    game.players[0].reserved_cards.push(ReservedCard::new(obs_public_t1, true));

    // 执行对观察者 Player 0 的确定化重抽样
    let det_state = game.determinize_for_player(0, &mut rng);

    // 1. 观察者自身的卡牌严格保持不变
    assert_eq!(det_state.players[0].reserved_cards, game.players[0].reserved_cards);
    assert_eq!(det_state.players[0].cards, game.players[0].cards);

    // 2. 金字塔公开卡牌严格保持不变
    assert_eq!(det_state.pyramid, game.pyramid);

    // 3. 对手的公开预留卡严格保持不变
    assert_eq!(
        det_state.players[1].reserved_cards[2],
        game.players[1].reserved_cards[2]
    );

    // 4. 对手的盲抽暗牌：等级保持不变
    assert_eq!(
        det_state.players[1].reserved_cards[0].card.tier,
        CardTier::Tier1
    );
    assert!(!det_state.players[1].reserved_cards[0].is_public);
    assert_eq!(
        det_state.players[1].reserved_cards[1].card.tier,
        CardTier::Tier2
    );
    assert!(!det_state.players[1].reserved_cards[1].is_public);

    // 5. 牌堆剩余张数严格保持不变
    for tier in CardTier::ALL {
        let t = tier.index();
        assert_eq!(det_state.decks[t].len(), game.decks[t].len());
    }

    // 6. 全局 67 张珠宝卡守恒验证 (任何卡牌不重复、不遗漏)
    let mut card_counts = [0u32; 68];
    for row in &det_state.pyramid {
        for c in row { card_counts[c.id as usize] += 1; }
    }
    for p in &det_state.players {
        for c in &p.cards { card_counts[c.id as usize] += 1; }
        for rc in &p.reserved_cards { card_counts[rc.card.id as usize] += 1; }
    }
    for deck in &det_state.decks {
        for c in deck { card_counts[c.id as usize] += 1; }
    }

    for id in 0..67 {
        assert_eq!(card_counts[id], 1, "卡牌 ID {id} 在确定化后必须且只能出现恰好 1 次！");
    }
}

#[test]
fn test_mcts_determinization_search() {
    let mut game = GameState::new_game(777);
    let mut rng = ChaCha8Rng::seed_from_u64(888);

    // 对手持有一张盲抽暗牌
    let opp_blind = game.decks[0].pop().unwrap();
    game.players[1].reserved_cards.push(ReservedCard::new(opp_blind, false));

    let mcts = RustMCTS::default();
    let action = mcts.search_with_exploration(&game, 30, false, 0.3, 0.25, 0.0, &mut rng);

    assert!(action.is_some(), "带有对手盲抽暗牌的局面下，MCTS 搜索必须顺利生成合法决策！");
    let legal = RuleEngine::legal_actions(&game);
    assert!(
        legal.contains(&action.unwrap()),
        "MCTS 产生的决策必须属于当前真实局面的合法动作！"
    );
}

#[test]
fn test_optional_actions_order_and_privilege_restriction_after_replenish() {
    let mut game = GameState::new_game(777);
    assert_eq!(game.phase, TurnPhase::OptionalActions);

    // 给予当前玩家 2 个特权卷轴
    game.players[0].privileges = 2;
    game.privilege_pool = 1;

    // 1. 制造盘面空格以允许补盘
    let taken_gem = game.board.take(0, 0).unwrap();
    game.bag.push(taken_gem);

    // 此时合法可选行动应包含：SkipOptional、UsePrivilege 以及 ReplenishBoard
    let legals = RuleEngine::legal_actions(&game);
    assert!(legals.contains(&Action::SkipOptional));
    assert!(legals.contains(&Action::ReplenishBoard));
    let has_use_priv = legals.iter().any(|a| matches!(a, Action::UsePrivilege { .. }));
    assert!(has_use_priv, "持有特权且未补盘时，应生成 UsePrivilege 行动");

    // 2. 先使用特权拿取 (0, 1) 的宝石
    let privilege_target = (0, 1);
    assert!(game.board.get(privilege_target.0, privilege_target.1).is_some());
    let use_priv_action = Action::UsePrivilege {
        r: privilege_target.0,
        c: privilege_target.1,
    };
    assert!(GameEngine::step(&mut game, &use_priv_action).is_ok());

    // 使用 1 个特权后，玩家特权减为 1，公用池加 1，phase 仍为 OptionalActions
    assert_eq!(game.players[0].privileges, 1);
    assert_eq!(game.privilege_pool, 2);
    assert_eq!(game.phase, TurnPhase::OptionalActions);
    assert_eq!(game.privileges_used_this_turn, 1);
    assert!(!game.replenished_this_turn);

    // 此时仍可继续使用特权或执行补盘
    let legals_after_priv = RuleEngine::legal_actions(&game);
    assert!(legals_after_priv.iter().any(|a| matches!(a, Action::UsePrivilege { .. })));
    assert!(legals_after_priv.contains(&Action::ReplenishBoard));

    // 3. 执行补充棋盘
    let p1_priv_before = game.players[1].privileges;
    assert!(GameEngine::step(&mut game, &Action::ReplenishBoard).is_ok());

    // 补盘后：
    // - 对手获得 1 个特权
    // - replenished_this_turn 标记为 true
    // - 阶段立即自动流转到 MandatoryAction（可选行动阶段彻底结束）
    assert_eq!(game.players[1].privileges, p1_priv_before + 1);
    assert!(game.replenished_this_turn);
    assert_eq!(game.phase, TurnPhase::MandatoryAction);

    // 4. 验证补盘后合法行动中绝无 UsePrivilege 或 ReplenishBoard
    let legals_after_replenish = RuleEngine::legal_actions(&game);
    assert!(!legals_after_replenish.iter().any(|a| matches!(a, Action::UsePrivilege { .. })));
    assert!(!legals_after_replenish.contains(&Action::ReplenishBoard));

    // 5. 验证若强行调用 step 执行 UsePrivilege，引擎必须拒绝
    let illegal_use_priv = Action::UsePrivilege { r: 2, c: 2 };
    let res = GameEngine::step(&mut game, &illegal_use_priv);
    assert!(res.is_err(), "补盘后或非 OptionalActions 阶段强行使用特权必须报错");

    // 6. 验证同一回合不可再次补充棋盘
    let illegal_replenish = Action::ReplenishBoard;
    let res_rep = GameEngine::step(&mut game, &illegal_replenish);
    assert!(res_rep.is_err(), "同一回合重复补充棋盘必须报错");
}

#[test]
fn test_must_replenish_when_no_mandatory_actions_available() {
    let mut game = GameState::new_game(999);
    assert_eq!(game.phase, TurnPhase::OptionalActions);

    // 构造极端场景：棋盘上清空所有非黄金标记与黄金，玩家手中无任何标记，无预留牌，买不起任何金字塔卡牌
    for r in 0..5 {
        for c in 0..5 {
            if let Some(gem) = game.board.take(r, c) {
                game.bag.push(gem);
            }
        }
    }
    // 此时棋盘全空，袋子有全部标记
    assert!(!game.board.has_non_gold());
    assert!(!game.board.has_gold());
    assert_eq!(game.players[0].tokens.total(), 0);

    // 校验极速判定：当前玩家无任何合法强制行动
    assert!(!RuleEngine::has_any_mandatory_action(&game));

    // 校验可选行动列表：
    // 规则书规定：无法执行任何强制行动时，必须在可选阶段补充棋盘，绝对不能跳过！
    let legals = RuleEngine::legal_actions(&game);
    assert!(
        !legals.contains(&Action::SkipOptional),
        "无必选操作可做时，可选阶段绝不能允许 SkipOptional 跳过！"
    );
    assert!(
        legals.contains(&Action::ReplenishBoard),
        "无必选操作可做时，可选阶段必须允许 ReplenishBoard！"
    );

    // 执行补充棋盘
    assert!(GameEngine::step(&mut game, &Action::ReplenishBoard).is_ok());

    // 补盘后棋盘被填满，自动转入 MandatoryAction，此时已有宝石可拿
    assert_eq!(game.phase, TurnPhase::MandatoryAction);
    assert!(game.board.has_non_gold());
    let mandatory_legals = RuleEngine::legal_actions(&game);
    assert!(!mandatory_legals.is_empty(), "补盘后必须有合法的拿取连线等强制行动！");
}



