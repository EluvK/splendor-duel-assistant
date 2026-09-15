use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::ai::heuristic_ai::HeuristicAI;
use crate::ai::mcts::RustMCTS;
use crate::ai::neural_evaluator::TractNeuralEvaluator;
use crate::bridge::encode::{
    action_mask_from_legals, action_to_id, encode_state, ACTION_SIZE, OBS_SIZE,
};
use crate::game_state::phase::{TurnPhase, VictoryReason};
use crate::game_state::state::GameState;
use crate::gameplay::engine::GameEngine;
use crate::gameplay::rules::RuleEngine;

/// 单局对弈步数安全上限：防止早期未成熟策略陷入死循环，超过此步数判定为超时平局结算
pub const MAX_GAME_STEPS: usize = 400;

/// 单局紧凑对弈轨迹
struct SingleGameTrajectory {
    obs: Vec<f32>,     // steps * OBS_SIZE
    masks: Vec<u8>,    // steps * ACTION_SIZE (0 或 1)
    actions: Vec<i32>, // steps (0..ACTION_SIZE-1)
    values: Vec<f32>,  // steps * 2: [win_value, turns_value]
    reasons: Vec<i32>, // steps: [0=20_pts, 1=10_crowns, 2=10_color, 3=draw]
    steps: usize,
}

/// 批量多线程紧凑样本包
pub struct CompactBatchSamples {
    pub total_steps: usize,
    pub obs: Vec<f32>,
    pub masks: Vec<u8>,
    pub actions: Vec<i32>,
    pub values: Vec<f32>,
    pub reasons: Vec<i32>,
}

#[inline]
fn victory_reason_to_id(reason_opt: Option<VictoryReason>) -> i32 {
    match reason_opt {
        Some(VictoryReason::TwentyPrestigePoints) => 0,
        Some(VictoryReason::TenCrowns) => 1,
        Some(VictoryReason::TenPointsSameColor(_)) => 2,
        None => 3, // 超时平局 / 无胜因
    }
}

/// 计算多任务目标标签 (纯胜率期望、归一化剩余轮数、终局胜因类别)
#[inline]
fn compute_multi_target_labels(
    raw_players: &[usize],
    raw_turns: &[u32],
    winner_opt: Option<usize>,
    reason_opt: Option<VictoryReason>,
    final_turn: u32,
) -> (Vec<f32>, Vec<i32>) {
    let steps = raw_players.len();
    let mut values = Vec::with_capacity(steps * 2);
    let mut reasons = Vec::with_capacity(steps);
    let reason_id = victory_reason_to_id(reason_opt);

    for i in 0..steps {
        let p = raw_players[i];
        let turn = raw_turns[i];
        // 1. 纯胜率期望 (不带任何时间折现，消除苟活负折扣漏洞)
        let win_target = match winner_opt {
            Some(winner) => {
                if p == winner {
                    1.0
                } else {
                    -1.0
                }
            }
            None => 0.0, // 超时平局
        };
        // 2. 剩余对局轮数归一化 (当前步距离终局的回合数 / 80.0，范围 [0.0, 1.0])
        let rem_turns = final_turn.saturating_sub(turn) as f32;
        let turns_target = (rem_turns / 80.0).clamp(0.0, 1.0);

        values.push(win_target);
        values.push(turns_target);
        reasons.push(reason_id);
    }
    (values, reasons)
}

fn simulate_single_heuristic_game(seed: u64) -> Option<SingleGameTrajectory> {
    let mut game = GameState::new_game(seed);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut raw_obs = Vec::with_capacity(200 * OBS_SIZE);
    let mut raw_masks = Vec::with_capacity(200 * ACTION_SIZE);
    let mut raw_actions = Vec::with_capacity(200);
    let mut raw_players = Vec::with_capacity(200);
    let mut raw_turns = Vec::with_capacity(200);
    let mut steps = 0;

    while !matches!(game.phase, TurnPhase::GameOver(_)) {
        steps += 1;
        if steps > MAX_GAME_STEPS {
            break;
        }

        let legals = RuleEngine::legal_actions(&game);
        if legals.is_empty() {
            break;
        }

        let acting_player = game.current_player;
        let obs = encode_state(&game);
        let mask = action_mask_from_legals(&legals);

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
        raw_turns.push(game.turn_number);

        if GameEngine::step(&mut game, &action).is_err() {
            return None;
        }
    }

    let actual_steps = raw_actions.len();
    if actual_steps == 0 {
        return None;
    }

    let winner_opt = game.winner.map(|(w, _)| w);
    let reason_opt = game.winner.map(|(_, r)| r);
    let final_turn = game.turn_number;
    let (values, reasons) = compute_multi_target_labels(
        &raw_players,
        &raw_turns,
        winner_opt,
        reason_opt,
        final_turn,
    );

    Some(SingleGameTrajectory {
        obs: raw_obs,
        masks: raw_masks,
        actions: raw_actions,
        values,
        reasons,
        steps: actual_steps,
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
    let mut all_values = Vec::with_capacity(total_steps * 2);
    let mut all_reasons = Vec::with_capacity(total_steps);

    for t in trajectories {
        all_obs.extend(t.obs);
        all_masks.extend(t.masks);
        all_actions.extend(t.actions);
        all_values.extend(t.values);
        all_reasons.extend(t.reasons);
    }

    CompactBatchSamples {
        total_steps,
        obs: all_obs,
        masks: all_masks,
        actions: all_actions,
        values: all_values,
        reasons: all_reasons,
    }
}

fn simulate_single_mcts_game(
    mcts: &RustMCTS,
    num_sims: usize,
    seed: u64,
    temp_steps: usize,
    temp_final: f32,
    dirichlet_alpha: f32,
    dirichlet_eps: f32,
) -> Option<SingleGameTrajectory> {
    let mut game = GameState::new_game(seed);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut raw_obs = Vec::with_capacity(200 * OBS_SIZE);
    let mut raw_masks = Vec::with_capacity(200 * ACTION_SIZE);
    let mut raw_actions = Vec::with_capacity(200);
    let mut raw_players = Vec::with_capacity(200);
    let mut raw_turns = Vec::with_capacity(200);
    let mut steps = 0;

    while !matches!(game.phase, TurnPhase::GameOver(_)) {
        steps += 1;
        if steps > MAX_GAME_STEPS {
            break;
        }

        let legals = RuleEngine::legal_actions(&game);
        if legals.is_empty() {
            break;
        }

        let acting_player = game.current_player;
        let obs = encode_state(&game);
        let mask = action_mask_from_legals(&legals);

        // 前 temp_steps 步 (如前 12 步) 启用温度 1.0 轮盘赌与根节点 Dirichlet 噪声，破除开局盲区
        let (add_noise, temp) = if steps <= temp_steps {
            (true, 1.0)
        } else {
            (false, temp_final)
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
        raw_turns.push(game.turn_number);

        if GameEngine::step(&mut game, &action).is_err() {
            return None;
        }
    }

    let actual_steps = raw_actions.len();
    if actual_steps == 0 {
        return None;
    }

    let winner_opt = game.winner.map(|(w, _)| w);
    let reason_opt = game.winner.map(|(_, r)| r);
    let final_turn = game.turn_number;
    let (values, reasons) = compute_multi_target_labels(
        &raw_players,
        &raw_turns,
        winner_opt,
        reason_opt,
        final_turn,
    );

    Some(SingleGameTrajectory {
        obs: raw_obs,
        masks: raw_masks,
        actions: raw_actions,
        values,
        reasons,
        steps: actual_steps,
    })
}

/// 并行采样 N 局带 MCTS 深度推演与 AlphaZero 探索机制的自博弈对局 (8 线程全速并发)
pub fn sample_mcts_games_parallel_with_config(
    num_games: usize,
    num_sims: usize,
    start_seed: u64,
    temp_steps: usize,
    temp_final: f32,
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
                temp_final,
                dirichlet_alpha,
                dirichlet_eps,
            )
        })
        .collect();

    let total_steps: usize = trajectories.iter().map(|t| t.steps).sum();

    let mut all_obs = Vec::with_capacity(total_steps * OBS_SIZE);
    let mut all_masks = Vec::with_capacity(total_steps * ACTION_SIZE);
    let mut all_actions = Vec::with_capacity(total_steps);
    let mut all_values = Vec::with_capacity(total_steps * 2);
    let mut all_reasons = Vec::with_capacity(total_steps);

    for t in trajectories {
        all_obs.extend(t.obs);
        all_masks.extend(t.masks);
        all_actions.extend(t.actions);
        all_values.extend(t.values);
        all_reasons.extend(t.reasons);
    }

    CompactBatchSamples {
        total_steps,
        obs: all_obs,
        masks: all_masks,
        actions: all_actions,
        values: all_values,
        reasons: all_reasons,
    }
}

fn simulate_single_neural_mcts_game(
    mcts: &RustMCTS,
    evaluator: &TractNeuralEvaluator,
    num_sims: usize,
    seed: u64,
    temp_steps: usize,
    temp_final: f32,
    dirichlet_alpha: f32,
    dirichlet_eps: f32,
) -> Option<SingleGameTrajectory> {
    let mut game = GameState::new_game(seed);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut raw_obs = Vec::with_capacity(200 * OBS_SIZE);
    let mut raw_masks = Vec::with_capacity(200 * ACTION_SIZE);
    let mut raw_actions = Vec::with_capacity(200);
    let mut raw_players = Vec::with_capacity(200);
    let mut raw_turns = Vec::with_capacity(200);
    let mut steps = 0;

    while !matches!(game.phase, TurnPhase::GameOver(_)) {
        steps += 1;
        if steps > MAX_GAME_STEPS {
            break;
        }

        let legals = RuleEngine::legal_actions(&game);
        if legals.is_empty() {
            break;
        }

        let acting_player = game.current_player;
        let obs = encode_state(&game);
        let mask = action_mask_from_legals(&legals);

        let (add_noise, temp) = if steps <= temp_steps {
            (true, 1.0)
        } else {
            (false, temp_final)
        };

        let action = mcts.search_neural_with_exploration_and_legals(
            &game,
            legals,
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
        raw_turns.push(game.turn_number);

        if GameEngine::step(&mut game, &action).is_err() {
            return None;
        }
    }

    let actual_steps = raw_actions.len();
    if actual_steps == 0 {
        return None;
    }

    let winner_opt = game.winner.map(|(w, _)| w);
    let reason_opt = game.winner.map(|(_, r)| r);
    let final_turn = game.turn_number;
    let (values, reasons) = compute_multi_target_labels(
        &raw_players,
        &raw_turns,
        winner_opt,
        reason_opt,
        final_turn,
    );

    Some(SingleGameTrajectory {
        obs: raw_obs,
        masks: raw_masks,
        actions: raw_actions,
        values,
        reasons,
        steps: actual_steps,
    })
}

/// 并行采样 N 局由 ONNX 神经网络指导的纯 AlphaZero MCTS 自博弈对局 (多线程全速并发)
pub fn sample_neural_mcts_games_parallel(
    onnx_bytes: &[u8],
    num_games: usize,
    num_sims: usize,
    start_seed: u64,
    temp_steps: usize,
    temp_final: f32,
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
                temp_final,
                dirichlet_alpha,
                dirichlet_eps,
            )
        })
        .collect();

    let total_steps: usize = trajectories.iter().map(|t| t.steps).sum();

    let mut all_obs = Vec::with_capacity(total_steps * OBS_SIZE);
    let mut all_masks = Vec::with_capacity(total_steps * ACTION_SIZE);
    let mut all_actions = Vec::with_capacity(total_steps);
    let mut all_values = Vec::with_capacity(total_steps * 2);
    let mut all_reasons = Vec::with_capacity(total_steps);

    for t in trajectories {
        all_obs.extend(t.obs);
        all_masks.extend(t.masks);
        all_actions.extend(t.actions);
        all_values.extend(t.values);
        all_reasons.extend(t.reasons);
    }

    Ok(CompactBatchSamples {
        total_steps,
        obs: all_obs,
        masks: all_masks,
        actions: all_actions,
        values: all_values,
        reasons: all_reasons,
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
    pub p0_seat_wins: usize,
    pub p1_seat_wins: usize,
    pub total_steps: usize,
    pub total_rounds: usize,
    pub agent0_win_steps: usize,
    pub agent0_win_rounds: usize,
    pub agent0_lose_steps: usize,
    pub agent0_lose_rounds: usize,
}

/// 纯 Rust 多线程 8 核并发进行严格换座的对抗评测 (支持纯 PolicyNet 或 Neural MCTS，支持模型间对战或模型对战启发式)
pub fn evaluate_neural_match_parallel(
    bytes0: &[u8],
    bytes1: Option<&[u8]>,
    num_pairs: usize,
    base_seed: u64,
    num_sims: usize,
) -> Result<ParallelMatchResult, String> {
    let eval0 = TractNeuralEvaluator::from_bytes(bytes0)?;
    let eval1 = match bytes1 {
        Some(b) if !b.is_empty() => Some(TractNeuralEvaluator::from_bytes(b)?),
        _ => None,
    };

    let total_games = num_pairs * 2;
    let mut tasks = Vec::with_capacity(total_games);
    for i in 0..num_pairs {
        let seed = base_seed + (i as u64) * 997;
        tasks.push((seed, false)); // 局 1: agent0 是 P0, agent1 是 P1
        tasks.push((seed, true));  // 局 2: agent1 是 P0, agent0 是 P1
    }

    let results: Vec<(Option<usize>, bool, Option<VictoryReason>, usize, usize)> = tasks
        .into_par_iter()
        .map(|(seed, is_swap)| {
            let mut game = GameState::new_game(seed);
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mcts = RustMCTS::default();
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

                let chosen_act = if is_agent0 {
                    if num_sims == 0 {
                        let obs = encode_state(&game);
                        let (logits, _) = match eval0.evaluate(&obs) {
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
                        best_act
                    } else {
                        match mcts.search_neural_with_exploration_and_legals(
                            &game,
                            legals,
                            &eval0,
                            num_sims,
                            false,
                            0.0,
                            0.0,
                            0.0,
                            &mut rng,
                        ) {
                            Some(act) => act,
                            None => break,
                        }
                    }
                } else if let Some(ref eval1) = eval1 {
                    if num_sims == 0 {
                        let obs = encode_state(&game);
                        let (logits, _) = match eval1.evaluate(&obs) {
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
                        best_act
                    } else {
                        match mcts.search_neural_with_exploration_and_legals(
                            &game,
                            legals,
                            eval1,
                            num_sims,
                            false,
                            0.0,
                            0.0,
                            0.0,
                            &mut rng,
                        ) {
                            Some(act) => act,
                            None => break,
                        }
                    }
                } else {
                    // 对战内置 HeuristicAI
                    match HeuristicAI::select_action(&game, &mut rng) {
                        Some(act) => act,
                        None => legals[0].clone(),
                    }
                };

                if GameEngine::step(&mut game, &chosen_act).is_err() {
                    break;
                }
            }

            let winner = game.winner.map(|(w, r)| (w, r));
            let w_id = winner.map(|(w, _)| w);
            let reason = winner.map(|(_, r)| r);
            let rounds = game.round_number() as usize;
            (w_id, is_swap, reason, steps, rounds)
        })
        .collect();

    let mut res = ParallelMatchResult {
        total_games,
        ..Default::default()
    };

    for (winner, is_swap, reason, steps, rounds) in results {
        res.total_steps += steps;
        res.total_rounds += rounds;
        match winner {
            Some(w) => {
                let agent0_won = if !is_swap { w == 0 } else { w == 1 };
                if agent0_won {
                    res.agent0_wins += 1;
                    res.agent0_win_steps += steps;
                    res.agent0_win_rounds += rounds;
                } else {
                    res.agent1_wins += 1;
                    res.agent0_lose_steps += steps;
                    res.agent0_lose_rounds += rounds;
                }
                if w == 0 {
                    res.p0_seat_wins += 1;
                } else {
                    res.p1_seat_wins += 1;
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
