use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::ai::heuristic_ai::HeuristicAI;
use crate::ai::mcts::RustMCTS;
use crate::ai::neural_evaluator::TractNeuralEvaluator;
use crate::bridge::encode::{action_mask, action_to_id, encode_state, ACTION_SIZE, OBS_SIZE};
use crate::game_state::phase::{TurnPhase, VictoryReason};
use crate::game_state::state::GameState;
use crate::gameplay::engine::GameEngine;
use crate::gameplay::rules::RuleEngine;

/// 单局紧凑对弈轨迹
struct SingleGameTrajectory {
    obs: Vec<f32>,     // steps * OBS_SIZE
    masks: Vec<u8>,    // steps * ACTION_SIZE (0 或 1)
    actions: Vec<i32>, // steps (0..ACTION_SIZE-1)
    values: Vec<f32>,  // steps (-1.0 或 1.0)
    steps: usize,
}

/// 批量多线程紧凑样本包
pub struct CompactBatchSamples {
    pub total_steps: usize,
    pub obs: Vec<f32>,
    pub masks: Vec<u8>,
    pub actions: Vec<i32>,
    pub values: Vec<f32>,
}

fn simulate_single_heuristic_game(seed: u64) -> Option<SingleGameTrajectory> {
    let mut game = GameState::new_game(seed);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut raw_obs = Vec::with_capacity(200 * OBS_SIZE);
    let mut raw_masks = Vec::with_capacity(200 * ACTION_SIZE);
    let mut raw_actions = Vec::with_capacity(200);
    let mut raw_players = Vec::with_capacity(200);
    let mut steps = 0;

    while !matches!(game.phase, TurnPhase::GameOver(_)) {
        steps += 1;
        if steps > 1500 {
            return None;
        }

        let acting_player = game.current_player;
        let obs = encode_state(&game);
        let mask = action_mask(&game);

        let action = HeuristicAI::select_action(&game, &mut rng)?;
        let action_id = action_to_id(&action);
        if action_id >= ACTION_SIZE {
            return None;
        }

        raw_obs.extend_from_slice(&obs);
        for &b in mask.iter() {
            raw_masks.push(if b { 1 } else { 0 });
        }
        raw_actions.push(action_id as i32);
        raw_players.push(acting_player);

        if GameEngine::step(&mut game, &action).is_err() {
            return None;
        }
    }

    let winner = game.winner.map(|(w, _)| w)?;

    let mut values = Vec::with_capacity(steps);
    for &p in raw_players.iter() {
        values.push(if p == winner { 1.0 } else { -1.0 });
    }

    Some(SingleGameTrajectory {
        obs: raw_obs,
        masks: raw_masks,
        actions: raw_actions,
        values,
        steps,
    })
}

/// 并行采样 N 局启发式对决数据 (8 线程全速)
pub fn sample_heuristic_games_parallel(num_games: usize, start_seed: u64) -> CompactBatchSamples {
    let trajectories: Vec<SingleGameTrajectory> = (0..num_games)
        .into_par_iter()
        .filter_map(|idx| simulate_single_heuristic_game(start_seed + idx as u64))
        .collect();

    let total_steps: usize = trajectories.iter().map(|t| t.steps).sum();

    let mut all_obs = Vec::with_capacity(total_steps * OBS_SIZE);
    let mut all_masks = Vec::with_capacity(total_steps * ACTION_SIZE);
    let mut all_actions = Vec::with_capacity(total_steps);
    let mut all_values = Vec::with_capacity(total_steps);

    for t in trajectories {
        all_obs.extend(t.obs);
        all_masks.extend(t.masks);
        all_actions.extend(t.actions);
        all_values.extend(t.values);
    }

    CompactBatchSamples {
        total_steps,
        obs: all_obs,
        masks: all_masks,
        actions: all_actions,
        values: all_values,
    }
}

fn simulate_single_mcts_game(
    mcts: &RustMCTS,
    num_sims: usize,
    seed: u64,
    temp_steps: usize,
    dirichlet_alpha: f32,
    dirichlet_eps: f32,
) -> Option<SingleGameTrajectory> {
    let mut game = GameState::new_game(seed);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut raw_obs = Vec::with_capacity(200 * OBS_SIZE);
    let mut raw_masks = Vec::with_capacity(200 * ACTION_SIZE);
    let mut raw_actions = Vec::with_capacity(200);
    let mut raw_players = Vec::with_capacity(200);
    let mut steps = 0;

    while !matches!(game.phase, TurnPhase::GameOver(_)) {
        steps += 1;
        if steps > 1500 {
            return None;
        }

        let acting_player = game.current_player;
        let obs = encode_state(&game);
        let mask = action_mask(&game);

        // 前 temp_steps 步 (如前 12 步) 启用温度 1.0 轮盘赌与根节点 Dirichlet 噪声，破除开局盲区
        let (add_noise, temp) = if steps <= temp_steps {
            (true, 1.0)
        } else {
            (false, 0.0)
        };

        let action = mcts.search_with_exploration(
            &game,
            num_sims,
            add_noise,
            dirichlet_alpha,
            dirichlet_eps,
            temp,
            &mut rng,
        )?;
        let action_id = action_to_id(&action);
        if action_id >= ACTION_SIZE {
            return None;
        }

        raw_obs.extend_from_slice(&obs);
        for &b in mask.iter() {
            raw_masks.push(if b { 1 } else { 0 });
        }
        raw_actions.push(action_id as i32);
        raw_players.push(acting_player);

        if GameEngine::step(&mut game, &action).is_err() {
            return None;
        }
    }

    let winner = game.winner.map(|(w, _)| w)?;

    let mut values = Vec::with_capacity(steps);
    for &p in raw_players.iter() {
        values.push(if p == winner { 1.0 } else { -1.0 });
    }

    Some(SingleGameTrajectory {
        obs: raw_obs,
        masks: raw_masks,
        actions: raw_actions,
        values,
        steps,
    })
}

/// 并行采样 N 局带 MCTS 深度推演与 AlphaZero 探索机制的自博弈对局 (8 线程全速并发)
pub fn sample_mcts_games_parallel_with_config(
    num_games: usize,
    num_sims: usize,
    start_seed: u64,
    temp_steps: usize,
    dirichlet_alpha: f32,
    dirichlet_eps: f32,
) -> CompactBatchSamples {
    let mcts = RustMCTS::default();
    let trajectories: Vec<SingleGameTrajectory> = (0..num_games)
        .into_par_iter()
        .filter_map(|idx| {
            simulate_single_mcts_game(
                &mcts,
                num_sims,
                start_seed + idx as u64,
                temp_steps,
                dirichlet_alpha,
                dirichlet_eps,
            )
        })
        .collect();

    let total_steps: usize = trajectories.iter().map(|t| t.steps).sum();

    let mut all_obs = Vec::with_capacity(total_steps * OBS_SIZE);
    let mut all_masks = Vec::with_capacity(total_steps * ACTION_SIZE);
    let mut all_actions = Vec::with_capacity(total_steps);
    let mut all_values = Vec::with_capacity(total_steps);

    for t in trajectories {
        all_obs.extend(t.obs);
        all_masks.extend(t.masks);
        all_actions.extend(t.actions);
        all_values.extend(t.values);
    }

    CompactBatchSamples {
        total_steps,
        obs: all_obs,
        masks: all_masks,
        actions: all_actions,
        values: all_values,
    }
}

/// 兼容老旧签名的并行 MCTS 采样
pub fn sample_mcts_games_parallel(
    num_games: usize,
    num_sims: usize,
    start_seed: u64,
) -> CompactBatchSamples {
    sample_mcts_games_parallel_with_config(num_games, num_sims, start_seed, 12, 0.3, 0.25)
}

fn simulate_single_neural_mcts_game(
    mcts: &RustMCTS,
    evaluator: &TractNeuralEvaluator,
    num_sims: usize,
    seed: u64,
    temp_steps: usize,
    dirichlet_alpha: f32,
    dirichlet_eps: f32,
) -> Option<SingleGameTrajectory> {
    let mut game = GameState::new_game(seed);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut raw_obs = Vec::with_capacity(200 * OBS_SIZE);
    let mut raw_masks = Vec::with_capacity(200 * ACTION_SIZE);
    let mut raw_actions = Vec::with_capacity(200);
    let mut raw_players = Vec::with_capacity(200);
    let mut steps = 0;

    while !matches!(game.phase, TurnPhase::GameOver(_)) {
        steps += 1;
        if steps > 1500 {
            return None;
        }

        let acting_player = game.current_player;
        let obs = encode_state(&game);
        let mask = action_mask(&game);

        let (add_noise, temp) = if steps <= temp_steps {
            (true, 1.0)
        } else {
            (false, 0.0)
        };

        let action = mcts.search_neural_with_exploration(
            &game,
            evaluator,
            num_sims,
            add_noise,
            dirichlet_alpha,
            dirichlet_eps,
            temp,
            &mut rng,
        )?;
        let action_id = action_to_id(&action);
        if action_id >= ACTION_SIZE {
            return None;
        }

        raw_obs.extend_from_slice(&obs);
        for &b in mask.iter() {
            raw_masks.push(if b { 1 } else { 0 });
        }
        raw_actions.push(action_id as i32);
        raw_players.push(acting_player);

        if GameEngine::step(&mut game, &action).is_err() {
            return None;
        }
    }

    let winner = game.winner.map(|(w, _)| w)?;

    let mut values = Vec::with_capacity(steps);
    for &p in raw_players.iter() {
        values.push(if p == winner { 1.0 } else { -1.0 });
    }

    Some(SingleGameTrajectory {
        obs: raw_obs,
        masks: raw_masks,
        actions: raw_actions,
        values,
        steps,
    })
}

/// 并行采样 N 局由 ONNX 神经网络指导的纯 AlphaZero MCTS 自博弈对局 (多线程全速并发)
pub fn sample_neural_mcts_games_parallel(
    onnx_bytes: &[u8],
    num_games: usize,
    num_sims: usize,
    start_seed: u64,
    temp_steps: usize,
    dirichlet_alpha: f32,
    dirichlet_eps: f32,
) -> Result<CompactBatchSamples, String> {
    let evaluator = TractNeuralEvaluator::from_bytes(onnx_bytes)?;
    let mcts = RustMCTS::default();

    let trajectories: Vec<SingleGameTrajectory> = (0..num_games)
        .into_par_iter()
        .filter_map(|idx| {
            simulate_single_neural_mcts_game(
                &mcts,
                &evaluator,
                num_sims,
                start_seed + idx as u64,
                temp_steps,
                dirichlet_alpha,
                dirichlet_eps,
            )
        })
        .collect();

    let total_steps: usize = trajectories.iter().map(|t| t.steps).sum();

    let mut all_obs = Vec::with_capacity(total_steps * OBS_SIZE);
    let mut all_masks = Vec::with_capacity(total_steps * ACTION_SIZE);
    let mut all_actions = Vec::with_capacity(total_steps);
    let mut all_values = Vec::with_capacity(total_steps);

    for t in trajectories {
        all_obs.extend(t.obs);
        all_masks.extend(t.masks);
        all_actions.extend(t.actions);
        all_values.extend(t.values);
    }

    Ok(CompactBatchSamples {
        total_steps,
        obs: all_obs,
        masks: all_masks,
        actions: all_actions,
        values: all_values,
    })
}

#[derive(Debug, Default)]
pub struct ParallelMatchResult {
    pub total_games: usize,
    pub agent0_wins: usize,
    pub agent1_wins: usize,
    pub draws: usize,
    pub reasons_20_pts: usize,
    pub reasons_10_crowns: usize,
    pub reasons_10_color: usize,
}

/// 纯 Rust 多线程 8 核并发进行严格换座的 PolicyNet 门禁对抗评测 (0.1 秒秒杀 20 局对决)
pub fn evaluate_neural_match_parallel(
    bytes0: &[u8],
    bytes1: &[u8],
    num_pairs: usize,
    base_seed: u64,
) -> Result<ParallelMatchResult, String> {
    let eval0 = TractNeuralEvaluator::from_bytes(bytes0)?;
    let eval1 = TractNeuralEvaluator::from_bytes(bytes1)?;

    let total_games = num_pairs * 2;
    let mut tasks = Vec::with_capacity(total_games);
    for i in 0..num_pairs {
        let seed = base_seed + (i as u64) * 997;
        tasks.push((seed, false)); // 局 1: agent0 是 P0, agent1 是 P1
        tasks.push((seed, true));  // 局 2: agent1 是 P0, agent0 是 P1
    }

    let results: Vec<(Option<usize>, bool, Option<VictoryReason>)> = tasks
        .into_par_iter()
        .map(|(seed, is_swap)| {
            let mut game = GameState::new_game(seed);
            let mut steps = 0;
            while !matches!(game.phase, TurnPhase::GameOver(_)) && steps < 400 {
                steps += 1;
                let acting_player = game.current_player;
                let is_agent0 = if !is_swap {
                    acting_player == 0
                } else {
                    acting_player == 1
                };

                let legals = RuleEngine::legal_actions(&game);
                if legals.is_empty() {
                    break;
                }
                if legals.len() == 1 {
                    let _ = GameEngine::step(&mut game, &legals[0]);
                    continue;
                }

                let obs = encode_state(&game);
                let eval = if is_agent0 { &eval0 } else { &eval1 };
                let (logits, _) = match eval.evaluate(&obs) {
                    Ok(r) => r,
                    Err(_) => break,
                };

                let mut best_score = f32::NEG_INFINITY;
                let mut best_act = legals[0].clone();
                for act in legals {
                    let id = action_to_id(&act);
                    let logit = if id < ACTION_SIZE { logits[id] } else { 0.0 };
                    if logit > best_score {
                        best_score = logit;
                        best_act = act;
                    }
                }

                if GameEngine::step(&mut game, &best_act).is_err() {
                    break;
                }
            }

            let winner = game.winner.map(|(w, r)| (w, r));
            let w_id = winner.map(|(w, _)| w);
            let reason = winner.map(|(_, r)| r);
            (w_id, is_swap, reason)
        })
        .collect();

    let mut res = ParallelMatchResult {
        total_games,
        ..Default::default()
    };

    for (winner, is_swap, reason) in results {
        match winner {
            Some(w) => {
                let agent0_won = if !is_swap { w == 0 } else { w == 1 };
                if agent0_won {
                    res.agent0_wins += 1;
                } else {
                    res.agent1_wins += 1;
                }
                if let Some(r) = reason {
                    match r {
                        VictoryReason::TwentyPrestigePoints => res.reasons_20_pts += 1,
                        VictoryReason::TenCrowns => res.reasons_10_crowns += 1,
                        VictoryReason::TenPointsSameColor(_) => res.reasons_10_color += 1,
                    }
                }
            }
            None => res.draws += 1,
        }
    }

    Ok(res)
}
