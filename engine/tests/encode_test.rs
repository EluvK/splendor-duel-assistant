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
