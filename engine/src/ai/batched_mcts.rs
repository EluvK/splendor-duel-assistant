//! 向量化 128 局大规模并发 GPU 批处理 MCTS 推演引擎
//!
//! 核心设计理念：
//! 1. 彻底摒弃 Thread-per-game 模式，采用轻量级事件循环（Vectorized Step Loop）；
//! 2. 单核驱动 64~128 局活跃游戏，在每个模拟波次（Wave）中统一收集所有待评估叶子盘面；
//! 3. 集中组装成连续扁平切片 [B, 1005]，一次性调用 GPU 执行批推理（Batch=32~128，耗时仅 ~1.5ms）；
//! 4. 广播回填结果，批量执行 Expansion & Backup，最大化 GPU Tensor Core 吞吐与利用率。

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::multi::{Dirichlet, MultiDistribution};

use crate::ai::heuristic_ai::HeuristicAI;
use crate::ai::mcts::{
    collapse_deterministic_micro_steps, compute_adaptive_sims, create_edges_from_logits,
    fast_state_hash, NeuralEvalCache, Node, LAMBDA_TURNS, MIS_BLOCK_SIZE,
};
use crate::ai::neural_evaluator::NeuralPrediction;
use crate::ai::sampling::{
    compute_multi_target_labels, CompactBatchSamples, ParallelMatchResult, SingleGameTrajectory,
    MAX_GAME_STEPS,
};
use crate::bridge::{action_to_id, encode_state, ACTION_SIZE, OBS_SIZE};
use crate::game_state::phase::{TurnPhase, VictoryReason};
use crate::game_state::state::GameState;
use crate::gameplay::engine::GameEngine;
use crate::gameplay::rules::RuleEngine;
use crate::model::action::Action;

/// GPU 批推理单样本融合输出维度: ACTION_SIZE (Logits) + 1 (Win) + 1 (Turns) + 3 (Reason) = 1861
pub const FUSED_OUTPUT_SIZE: usize = ACTION_SIZE + 5;

/// 单局工作单元当前状态
pub enum WorkerState {
    /// 空闲（未初始化或所有对局已完成）
    Idle,
    /// 处于轮次决策起点（检查终局、微步或启动 MCTS）
    AtDecision,
    /// 等待根节点神经网络评估
    WaitingRootEval {
        legals: Vec<Action>,
        add_dirichlet: bool,
    },
    /// 模拟迭代进行中
    Simulating,
    /// 等待叶子节点神经网络评估
    WaitingLeafEval {
        sim_state: GameState,
        path: Vec<(usize, usize)>,
        legals: Vec<Action>,
    },
}

/// 单个并发对局工作器
pub struct GameWorker {
    pub id: usize,
    pub rng: ChaCha8Rng,
    pub game: GameState,
    pub game_idx: usize,
    pub is_swap: bool,
    pub step_count: usize,
    pub state: WorkerState,

    // 轨迹数据收集
    pub raw_obs: Vec<f32>,
    pub raw_masks: Vec<u8>,
    pub raw_policies: Vec<f32>,
    pub raw_actions: Vec<i32>,
    pub raw_players: Vec<usize>,
    pub raw_turns: Vec<u32>,

    // MCTS 搜索数据
    pub eval_cache: NeuralEvalCache,
    pub nodes: Vec<Node>,
    pub sim_idx: usize,
    pub effective_sims: usize,
    pub base_sim_state: GameState,

    // 联赛模式标记
    pub is_heuristic_opponent: bool,
}

impl GameWorker {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            rng: ChaCha8Rng::seed_from_u64(1000 + id as u64),
            game: GameState::new_game(1000 + id as u64),
            game_idx: 0,
            is_swap: false,
            step_count: 0,
            state: WorkerState::Idle,
            raw_obs: Vec::with_capacity(MAX_GAME_STEPS * OBS_SIZE),
            raw_masks: Vec::with_capacity(MAX_GAME_STEPS * ACTION_SIZE),
            raw_policies: Vec::with_capacity(MAX_GAME_STEPS * ACTION_SIZE),
            raw_actions: Vec::with_capacity(MAX_GAME_STEPS),
            raw_players: Vec::with_capacity(MAX_GAME_STEPS),
            raw_turns: Vec::with_capacity(MAX_GAME_STEPS),
            eval_cache: NeuralEvalCache::new(2048),
            nodes: Vec::with_capacity(128),
            sim_idx: 0,
            effective_sims: 0,
            base_sim_state: GameState::new_game(1000 + id as u64),
            is_heuristic_opponent: false,
        }
    }

    pub fn start_game(
        &mut self,
        game_idx: usize,
        seed: u64,
        is_heuristic_opponent: bool,
    ) {
        self.game_idx = game_idx;
        self.is_swap = (game_idx % 2) == 1;
        self.rng = ChaCha8Rng::seed_from_u64(seed);
        self.game = GameState::new_game(seed);
        self.step_count = 0;
        self.is_heuristic_opponent = is_heuristic_opponent;

        self.raw_obs.clear();
        self.raw_masks.clear();
        self.raw_policies.clear();
        self.raw_actions.clear();
        self.raw_players.clear();
        self.raw_turns.clear();
        self.eval_cache = NeuralEvalCache::new(2048);
        self.nodes.clear();
        self.sim_idx = 0;
        self.effective_sims = 0;

        self.state = WorkerState::AtDecision;
    }

    /// 当前行动玩家是否为主要评估智能体 (Agent 0)
    #[inline]
    pub fn is_acting_agent0(&self) -> bool {
        let p = self.game.current_player;
        if !self.is_swap {
            p == 0
        } else {
            p == 1
        }
    }

    /// 将当前局对弈轨迹归档并计算多任务目标标签
    pub(crate) fn finalize_trajectory(&mut self) -> Option<SingleGameTrajectory> {
        let actual_steps = self.raw_actions.len();
        if actual_steps == 0 {
            return None;
        }

        let winner_opt = self.game.winner.map(|(w, _)| w);
        let final_turn = self.game.turn_number;
        let (values, reasons) = compute_multi_target_labels(
            &self.game,
            &self.raw_players,
            &self.raw_turns,
            winner_opt,
            final_turn,
        );

        Some(SingleGameTrajectory {
            obs: std::mem::take(&mut self.raw_obs),
            masks: std::mem::take(&mut self.raw_masks),
            policies: std::mem::take(&mut self.raw_policies),
            actions: std::mem::take(&mut self.raw_actions),
            values,
            reasons,
            steps: actual_steps,
        })
    }
}

/// 批处理推演调度引擎
pub struct BatchedMctsRunner {
    pub workers: Vec<GameWorker>,
    pub total_games: usize,
    pub num_sims: usize,
    pub start_seed: u64,
    pub temp_steps: usize,
    pub temp_final: f32,
    pub dirichlet_alpha: f32,
    pub dirichlet_eps: f32,
    pub c_puct: f32,
    pub heuristic_ratio: f32,
    pub record_opponent: bool,

    pub games_started: usize,
    pub games_completed: usize,
    pub(crate) completed_trajectories: Vec<SingleGameTrajectory>,

    // 扁平批处理缓冲区，预先分配容量避免频繁堆开销
    pub batch_obs: Vec<f32>,
    pub batch_indices: Vec<usize>, // 对应哪个 worker
}

impl BatchedMctsRunner {
    pub fn new(
        num_games: usize,
        num_sims: usize,
        max_concurrent: usize,
        start_seed: u64,
        temp_steps: usize,
        temp_final: f32,
        dirichlet_alpha: f32,
        dirichlet_eps: f32,
        heuristic_ratio: f32,
        record_opponent: bool,
    ) -> Self {
        let concurrency = max_concurrent.min(num_games).max(1);
        let mut workers = Vec::with_capacity(concurrency);
        for i in 0..concurrency {
            workers.push(GameWorker::new(i));
        }

        Self {
            workers,
            total_games: num_games,
            num_sims,
            start_seed,
            temp_steps,
            temp_final,
            dirichlet_alpha,
            dirichlet_eps,
            c_puct: 1.5,
            heuristic_ratio,
            record_opponent,
            games_started: 0,
            games_completed: 0,
            completed_trajectories: Vec::with_capacity(num_games),
            batch_obs: Vec::with_capacity(concurrency * OBS_SIZE),
            batch_indices: Vec::with_capacity(concurrency),
        }
    }

    /// 初始化第一批并发对局
    pub fn init_games(&mut self) {
        let num_heuristic = ((self.total_games as f32) * self.heuristic_ratio).round() as usize;
        let concurrency = self.workers.len();
        for i in 0..concurrency {
            if self.games_started < self.total_games {
                let g_idx = self.games_started;
                self.games_started += 1;
                let seed = self.start_seed + (g_idx as u64) * 997;
                let is_heu = g_idx < num_heuristic;
                self.workers[i].start_game(g_idx, seed, is_heu);
            }
        }
    }

    /// 推进各 worker 的 CPU 局部逻辑，直到所有活跃 worker 均进入等待 GPU 状态或已全部终局
    /// 返回待评估盘面总数 (即 batch 大小)
    pub fn collect_batch_requests(&mut self) -> usize {
        self.batch_obs.clear();
        self.batch_indices.clear();

        let num_heuristic = ((self.total_games as f32) * self.heuristic_ratio).round() as usize;

        // 循环推进各 worker，直到所有 worker 都处于 WaitingRootEval、WaitingLeafEval 或 Idle
        loop {
            let mut progressed = false;

            for w_idx in 0..self.workers.len() {
                let worker = &mut self.workers[w_idx];

                match &mut worker.state {
                    WorkerState::Idle => {
                        if self.games_started < self.total_games {
                            let g_idx = self.games_started;
                            self.games_started += 1;
                            let seed = self.start_seed + (g_idx as u64) * 997;
                            let is_heu = g_idx < num_heuristic;
                            worker.start_game(g_idx, seed, is_heu);
                            progressed = true;
                        }
                    }

                    WorkerState::AtDecision => {
                        worker.step_count += 1;
                        let is_game_over = matches!(worker.game.phase, TurnPhase::GameOver(_))
                            || worker.step_count > MAX_GAME_STEPS;

                        if is_game_over {
                            if let Some(traj) = worker.finalize_trajectory() {
                                self.completed_trajectories.push(traj);
                            }
                            self.games_completed += 1;
                            worker.state = WorkerState::Idle;
                            progressed = true;
                            continue;
                        }

                        let legals = RuleEngine::legal_actions(&worker.game);
                        if legals.is_empty() {
                            if let Some(traj) = worker.finalize_trajectory() {
                                self.completed_trajectories.push(traj);
                            }
                            self.games_completed += 1;
                            worker.state = WorkerState::Idle;
                            progressed = true;
                            continue;
                        }

                        // 单一确定性动作快速步进，跳过神经网络
                        if legals.len() == 1 {
                            let act = &legals[0];
                            let is_agent0 = worker.is_acting_agent0();
                            let should_record = is_agent0 || self.record_opponent;

                            if should_record {
                                let obs = encode_state(&worker.game);
                                let act_id = action_to_id(act);
                                if act_id < ACTION_SIZE {
                                    worker.raw_obs.extend_from_slice(&obs);
                                    let mut mask = [0u8; ACTION_SIZE];
                                    mask[act_id] = 1;
                                    worker.raw_masks.extend_from_slice(&mask);
                                    let mut pol = [0.0f32; ACTION_SIZE];
                                    pol[act_id] = 1.0;
                                    worker.raw_policies.extend_from_slice(&pol);
                                    worker.raw_actions.push(act_id as i32);
                                    worker.raw_players.push(worker.game.current_player);
                                    worker.raw_turns.push(worker.game.turn_number);
                                }
                            }
                            let _ = GameEngine::step(&mut worker.game, act);
                            progressed = true;
                            continue;
                        }

                        // 对战 Heuristic AI 模式
                        if worker.is_heuristic_opponent && !worker.is_acting_agent0() {
                            if let Some(act) =
                                HeuristicAI::select_action(&worker.game, &mut worker.rng)
                            {
                                if self.record_opponent {
                                    let obs = encode_state(&worker.game);
                                    let act_id = action_to_id(&act);
                                    if act_id < ACTION_SIZE {
                                        worker.raw_obs.extend_from_slice(&obs);
                                        let mut mask = [0u8; ACTION_SIZE];
                                        for a in &legals {
                                            let id = action_to_id(a);
                                            if id < ACTION_SIZE {
                                                mask[id] = 1;
                                            }
                                        }
                                        worker.raw_masks.extend_from_slice(&mask);
                                        let mut pol = [0.0f32; ACTION_SIZE];
                                        pol[act_id] = 1.0;
                                        worker.raw_policies.extend_from_slice(&pol);
                                        worker.raw_actions.push(act_id as i32);
                                        worker.raw_players.push(worker.game.current_player);
                                        worker.raw_turns.push(worker.game.turn_number);
                                    }
                                }
                                let _ = GameEngine::step(&mut worker.game, &act);
                                progressed = true;
                                continue;
                            }
                        }

                        // 准备进入 MCTS 搜索
                        let add_dirichlet = worker.step_count <= self.temp_steps;
                        let effective_sims = compute_adaptive_sims(
                            &worker.game.phase,
                            legals.len(),
                            self.num_sims,
                        );
                        worker.effective_sims = effective_sims;
                        worker.sim_idx = 0;

                        // 检查根节点转置表缓存
                        let key = fast_state_hash(&worker.game);
                        if let Some(&(logits, _pred)) = worker.eval_cache.get(&key) {
                            let mut root_edges =
                                create_edges_from_logits(&legals, &logits);
                            if add_dirichlet && root_edges.len() >= 2 {
                                let alphas = vec![self.dirichlet_alpha; root_edges.len()];
                                if let Ok(dir) = Dirichlet::new(&alphas) {
                                    let mut noise = vec![0.0f32; root_edges.len()];
                                    dir.sample_to_slice(&mut worker.rng, &mut noise);
                                    let mut sum = 0.0f32;
                                    for (i, edge) in root_edges.iter_mut().enumerate() {
                                        edge.prior = (1.0 - self.dirichlet_eps) * edge.prior
                                            + self.dirichlet_eps * noise[i];
                                        sum += edge.prior;
                                    }
                                    if sum > 1e-6 {
                                        for edge in root_edges.iter_mut() {
                                            edge.prior /= sum;
                                        }
                                    }
                                }
                            }
                            worker.nodes.clear();
                            worker.nodes.push(Node {
                                player: worker.game.current_player,
                                visits: 1,
                                edges: root_edges,
                                is_terminal: false,
                            });
                            worker.base_sim_state = worker
                                .game
                                .determinize_for_player(worker.game.current_player, &mut worker.rng);
                            worker.state = WorkerState::Simulating;
                            progressed = true;
                        } else {
                            worker.state = WorkerState::WaitingRootEval {
                                legals,
                                add_dirichlet,
                            };
                            progressed = true;
                        }
                    }

                    WorkerState::Simulating => {
                        // 推进单步模拟
                        // 1. 检查自适应早停：
                        //    - 确定性贪婪模式 (temp <= 0.01): 第一分支优势不可逆反超则纯数学无损截断
                        //    - 探索模式: 必须脱离初始探索噪声 (!add_dirichlet) 且完成 >= 60% 模拟后才允许提前收敛，
                        //      保护自博弈探索多样性并防止 MCTS 软策略分布熵塌缩
                        let add_dirichlet = worker.step_count <= self.temp_steps;
                        let temp = if add_dirichlet {
                            1.0f32
                        } else {
                            self.temp_final
                        };

                        let early_stopped = if worker.nodes[0].edges.len() >= 2 {
                            let can_check_early_stop = if temp <= 0.01 {
                                true
                            } else {
                                !add_dirichlet && worker.sim_idx >= (worker.effective_sims * 3) / 5
                            };

                            if can_check_early_stop {
                                let mut top1 = 0u32;
                                let mut top2 = 0u32;
                                for e in &worker.nodes[0].edges {
                                    if e.visits > top1 {
                                        top2 = top1;
                                        top1 = e.visits;
                                    } else if e.visits > top2 {
                                        top2 = e.visits;
                                    }
                                }
                                let remaining = worker.effective_sims.saturating_sub(worker.sim_idx) as u32;
                                top1.saturating_sub(top2) > remaining
                            } else {
                                false
                            }
                        } else {
                            false
                        };

                        if early_stopped || worker.sim_idx >= worker.effective_sims {
                            // MCTS 决策收敛，采样动作并步进游戏

                            let selected_edge_idx = if temp <= 0.05 {
                                let mut best_idx = 0;
                                let mut best_v = 0u32;
                                for (i, e) in worker.nodes[0].edges.iter().enumerate() {
                                    if e.visits > best_v {
                                        best_v = e.visits;
                                        best_idx = i;
                                    }
                                }
                                best_idx
                            } else {
                                let mut weights = Vec::with_capacity(worker.nodes[0].edges.len());
                                let inv_t = 1.0 / temp;
                                for e in &worker.nodes[0].edges {
                                    weights.push((e.visits as f32).powf(inv_t));
                                }
                                let sum_w: f32 = weights.iter().sum();
                                if sum_w > 1e-6 {
                                    let mut r = worker.rng.random::<f32>() * sum_w;
                                    let mut pick = 0;
                                    for (i, &w) in weights.iter().enumerate() {
                                        if r <= w {
                                            pick = i;
                                            break;
                                        }
                                        r -= w;
                                        pick = i;
                                    }
                                    pick
                                } else {
                                    0
                                }
                            };

                            let selected_action =
                                worker.nodes[0].edges[selected_edge_idx].action.clone();
                            let action_id = action_to_id(&selected_action);

                            let is_agent0 = worker.is_acting_agent0();
                            let should_record = is_agent0 || self.record_opponent;

                            if should_record && action_id < ACTION_SIZE {
                                let obs = encode_state(&worker.game);
                                let mut policy_vec = [0.0f32; ACTION_SIZE];
                                let total_visits: u32 =
                                    worker.nodes[0].edges.iter().map(|e| e.visits).sum();
                                if total_visits > 0 {
                                    for e in &worker.nodes[0].edges {
                                        let id = action_to_id(&e.action);
                                        if id < ACTION_SIZE {
                                            policy_vec[id] = (e.visits as f32) / (total_visits as f32);
                                        }
                                    }
                                } else {
                                    policy_vec[action_id] = 1.0;
                                }

                                let mut mask = [0u8; ACTION_SIZE];
                                for e in &worker.nodes[0].edges {
                                    let id = action_to_id(&e.action);
                                    if id < ACTION_SIZE {
                                        mask[id] = 1;
                                    }
                                }

                                worker.raw_obs.extend_from_slice(&obs);
                                worker.raw_masks.extend_from_slice(&mask);
                                worker.raw_policies.extend_from_slice(&policy_vec);
                                worker.raw_actions.push(action_id as i32);
                                worker.raw_players.push(worker.game.current_player);
                                worker.raw_turns.push(worker.game.turn_number);
                            }

                            let _ = GameEngine::step(&mut worker.game, &selected_action);
                            worker.state = WorkerState::AtDecision;
                            progressed = true;
                            continue;
                        }

                        // 执行 PUCT Selection 探寻叶子
                        if worker.sim_idx > 0 && worker.sim_idx % MIS_BLOCK_SIZE == 0 {
                            worker.base_sim_state = worker
                                .game
                                .determinize_for_player(worker.game.current_player, &mut worker.rng);
                        }
                        let mut sim_state = worker.base_sim_state;
                        let mut curr_node_idx = 0;
                        let mut path = Vec::with_capacity(16);

                        while !worker.nodes[curr_node_idx].is_terminal
                            && !worker.nodes[curr_node_idx].edges.is_empty()
                        {
                            let best_edge_idx =
                                worker.nodes[curr_node_idx].select_best_edge(self.c_puct);
                            let action =
                                worker.nodes[curr_node_idx].edges[best_edge_idx].action.clone();
                            if GameEngine::step(&mut sim_state, &action).is_err() {
                                break;
                            }
                            collapse_deterministic_micro_steps(&mut sim_state);
                            path.push((curr_node_idx, best_edge_idx));

                            if let Some(child_idx) =
                                worker.nodes[curr_node_idx].edges[best_edge_idx].child_idx
                            {
                                curr_node_idx = child_idx;
                            } else {
                                break;
                            }
                        }

                        if path.is_empty() {
                            worker.sim_idx += 1;
                            progressed = true;
                            continue;
                        }

                        let &(last_n, last_e) = path.last().unwrap();
                        let existing_child = worker.nodes[last_n].edges[last_e].child_idx;

                        // 检查叶子是否为终局
                        if matches!(sim_state.phase, TurnPhase::GameOver(_)) {
                            let win_v = match sim_state.winner.map(|(w, _)| w) {
                                Some(0) => 1.0,
                                Some(1) => -1.0,
                                _ => 0.0,
                            };
                            if existing_child.is_none() {
                                let new_node_idx = worker.nodes.len();
                                worker.nodes.push(Node {
                                    player: sim_state.current_player,
                                    visits: 1,
                                    edges: Vec::new(),
                                    is_terminal: true,
                                });
                                worker.nodes[last_n].edges[last_e].child_idx = Some(new_node_idx);
                            }
                            for &(n_idx, e_idx) in &path {
                                worker.nodes[n_idx].visits += 1;
                                worker.nodes[n_idx].edges[e_idx].visits += 1;
                                worker.nodes[n_idx].edges[e_idx].w_p0 += win_v;
                            }
                            worker.sim_idx += 1;
                            progressed = true;
                            continue;
                        }

                        let next_legals = RuleEngine::legal_actions(&sim_state);
                        if next_legals.is_empty() {
                            if existing_child.is_none() {
                                let new_node_idx = worker.nodes.len();
                                worker.nodes.push(Node {
                                    player: sim_state.current_player,
                                    visits: 1,
                                    edges: Vec::new(),
                                    is_terminal: true,
                                });
                                worker.nodes[last_n].edges[last_e].child_idx = Some(new_node_idx);
                            }
                            for &(n_idx, e_idx) in &path {
                                worker.nodes[n_idx].visits += 1;
                                worker.nodes[n_idx].edges[e_idx].visits += 1;
                            }
                            worker.sim_idx += 1;
                            progressed = true;
                            continue;
                        }

                        // 检查叶子节点缓存命中
                        let key = fast_state_hash(&sim_state);
                        if let Some(&(logits, pred)) = worker.eval_cache.get(&key) {
                            let edges =
                                create_edges_from_logits(&next_legals, &logits);
                            let v_mover = pred.combined_value(LAMBDA_TURNS);
                            let vp0 = if sim_state.current_player == 0 {
                                v_mover
                            } else {
                                -v_mover
                            };
                            if existing_child.is_none() {
                                let new_node_idx = worker.nodes.len();
                                worker.nodes.push(Node {
                                    player: sim_state.current_player,
                                    visits: 1,
                                    edges,
                                    is_terminal: false,
                                });
                                worker.nodes[last_n].edges[last_e].child_idx = Some(new_node_idx);
                            }
                            for &(n_idx, e_idx) in &path {
                                worker.nodes[n_idx].visits += 1;
                                worker.nodes[n_idx].edges[e_idx].visits += 1;
                                worker.nodes[n_idx].edges[e_idx].w_p0 += vp0;
                            }
                            worker.sim_idx += 1;
                            progressed = true;
                            continue;
                        }

                        // 未命中缓存，需要 GPU 批评估
                        worker.state = WorkerState::WaitingLeafEval {
                            sim_state,
                            path,
                            legals: next_legals,
                        };
                        progressed = true;
                    }

                    WorkerState::WaitingRootEval { .. } | WorkerState::WaitingLeafEval { .. } => {
                        // 已经处于就绪待评估状态，不重复推进
                    }
                }
            }

            if !progressed {
                break;
            }
        }

        // 收集所有处于等待状态的 worker 盘面
        for (w_idx, worker) in self.workers.iter().enumerate() {
            match &worker.state {
                WorkerState::WaitingRootEval { .. } => {
                    let obs = encode_state(&worker.game);
                    self.batch_obs.extend_from_slice(&obs);
                    self.batch_indices.push(w_idx);
                }
                WorkerState::WaitingLeafEval { sim_state, .. } => {
                    let obs = encode_state(sim_state);
                    self.batch_obs.extend_from_slice(&obs);
                    self.batch_indices.push(w_idx);
                }
                _ => {}
            }
        }

        self.batch_indices.len()
    }

    /// 将 GPU 批前向预测融合结果分发回流至各等待 worker
    pub fn apply_batch_results(&mut self, fused_slice: &[f32]) {
        let count = self.batch_indices.len();
        assert_eq!(
            fused_slice.len(),
            count * FUSED_OUTPUT_SIZE,
            "fused_slice length mismatch: expected {}, got {}",
            count * FUSED_OUTPUT_SIZE,
            fused_slice.len()
        );

        for (i, &w_idx) in self.batch_indices.iter().enumerate() {
            let offset = i * FUSED_OUTPUT_SIZE;
            let mut logits = [0.0f32; ACTION_SIZE];
            logits.copy_from_slice(&fused_slice[offset..offset + ACTION_SIZE]);
            let win_val = fused_slice[offset + ACTION_SIZE];
            let turns_val = fused_slice[offset + ACTION_SIZE + 1];
            let mut reason_probs = [0.0f32; 3];
            reason_probs.copy_from_slice(
                &fused_slice[offset + ACTION_SIZE + 2..offset + ACTION_SIZE + 5],
            );

            let pred = NeuralPrediction {
                win_value: win_val,
                turns_value: turns_val,
                reason_probs,
            };

            let worker = &mut self.workers[w_idx];

            let cur_state = std::mem::replace(&mut worker.state, WorkerState::Idle);
            match cur_state {
                WorkerState::WaitingRootEval {
                    legals,
                    add_dirichlet,
                } => {
                    let key = fast_state_hash(&worker.game);
                    worker.eval_cache.insert(key, (logits, pred));

                    let mut root_edges = create_edges_from_logits(&legals, &logits);
                    if add_dirichlet && root_edges.len() >= 2 {
                        let alphas = vec![self.dirichlet_alpha; root_edges.len()];
                        if let Ok(dir) = Dirichlet::new(&alphas) {
                            let mut noise = vec![0.0f32; root_edges.len()];
                            dir.sample_to_slice(&mut worker.rng, &mut noise);
                            let mut sum = 0.0f32;
                            for (idx, edge) in root_edges.iter_mut().enumerate() {
                                edge.prior = (1.0 - self.dirichlet_eps) * edge.prior
                                    + self.dirichlet_eps * noise[idx];
                                sum += edge.prior;
                            }
                            if sum > 1e-6 {
                                for edge in root_edges.iter_mut() {
                                    edge.prior /= sum;
                                }
                            }
                        }
                    }
                    worker.nodes.clear();
                    worker.nodes.push(Node {
                        player: worker.game.current_player,
                        visits: 1,
                        edges: root_edges,
                        is_terminal: false,
                    });
                    worker.base_sim_state = worker
                        .game
                        .determinize_for_player(worker.game.current_player, &mut worker.rng);
                    worker.state = WorkerState::Simulating;
                }

                WorkerState::WaitingLeafEval {
                    sim_state,
                    path,
                    legals,
                } => {
                    let key = fast_state_hash(&sim_state);
                    worker.eval_cache.insert(key, (logits, pred));

                    let edges = create_edges_from_logits(&legals, &logits);
                    let v_mover = pred.combined_value(LAMBDA_TURNS);
                    let vp0 = if sim_state.current_player == 0 {
                        v_mover
                    } else {
                        -v_mover
                    };

                    let (last_n, last_e) = *path.last().unwrap();
                    if worker.nodes[last_n].edges[last_e].child_idx.is_none() {
                        let new_node_idx = worker.nodes.len();
                        worker.nodes.push(Node {
                            player: sim_state.current_player,
                            visits: 1,
                            edges,
                            is_terminal: false,
                        });
                        worker.nodes[last_n].edges[last_e].child_idx = Some(new_node_idx);
                    }

                    for &(n_idx, e_idx) in &path {
                        worker.nodes[n_idx].visits += 1;
                        worker.nodes[n_idx].edges[e_idx].visits += 1;
                        worker.nodes[n_idx].edges[e_idx].w_p0 += vp0;
                    }
                    worker.sim_idx += 1;
                    worker.state = WorkerState::Simulating;
                }

                _ => {}
            }
        }
    }

    /// 判断全部对局是否均已推演完毕
    #[inline]
    pub fn is_finished(&self) -> bool {
        self.games_completed >= self.total_games
    }

    /// 聚合全部已完成对局的轨迹样本为单一标准样本包
    pub fn finalize_samples(self) -> CompactBatchSamples {
        let total_steps: usize = self.completed_trajectories.iter().map(|t| t.steps).sum();

        let mut all_obs = Vec::with_capacity(total_steps * OBS_SIZE);
        let mut all_masks = Vec::with_capacity(total_steps * ACTION_SIZE);
        let mut all_policies = Vec::with_capacity(total_steps * ACTION_SIZE);
        let mut all_actions = Vec::with_capacity(total_steps);
        let mut all_values = Vec::with_capacity(total_steps * 2);
        let mut all_reasons = Vec::with_capacity(total_steps * 3);

        for t in self.completed_trajectories {
            all_obs.extend(t.obs);
            all_masks.extend(t.masks);
            all_policies.extend(t.policies);
            all_actions.extend(t.actions);
            all_values.extend(t.values);
            all_reasons.extend(t.reasons);
        }

        CompactBatchSamples {
            total_steps,
            obs: all_obs,
            masks: all_masks,
            policies: all_policies,
            actions: all_actions,
            values: all_values,
            reasons: all_reasons,
        }
    }
}

/// 门禁对抗评测单个工作单元
pub struct MatchWorker {
    pub id: usize,
    pub rng: ChaCha8Rng,
    pub game: GameState,
    pub pair_idx: usize,
    pub is_swap: bool,
    pub step_count: usize,
    pub state: WorkerState,

    // MCTS 搜索数据
    pub eval_cache_0: NeuralEvalCache,
    pub eval_cache_1: NeuralEvalCache,
    pub nodes: Vec<Node>,
    pub sim_idx: usize,
    pub effective_sims: usize,
    pub base_sim_state: GameState,

    pub is_finished: bool,
    pub winner: Option<usize>,
    pub reason: Option<VictoryReason>,
    pub rounds: usize,
}

impl MatchWorker {
    pub fn new(id: usize, pair_idx: usize, seed: u64, is_swap: bool) -> Self {
        Self {
            id,
            rng: ChaCha8Rng::seed_from_u64(seed),
            game: GameState::new_game(seed),
            pair_idx,
            is_swap,
            step_count: 0,
            state: WorkerState::AtDecision,
            eval_cache_0: NeuralEvalCache::new(2048),
            eval_cache_1: NeuralEvalCache::new(2048),
            nodes: Vec::with_capacity(128),
            sim_idx: 0,
            effective_sims: 0,
            base_sim_state: GameState::new_game(seed),
            is_finished: false,
            winner: None,
            reason: None,
            rounds: 0,
        }
    }

    #[inline]
    pub fn is_acting_agent0(&self) -> bool {
        let p = self.game.current_player;
        if !self.is_swap {
            p == 0
        } else {
            p == 1
        }
    }
}

/// GPU 批推演成对对决门禁评测器
pub struct BatchedMatchRunner {
    pub workers: Vec<MatchWorker>,
    pub num_pairs: usize,
    pub total_games: usize,
    pub num_sims: usize,
    pub base_seed: u64,
    pub c_puct: f32,
    pub is_agent1_heuristic: bool,

    pub games_completed: usize,

    pub batch_obs_0: Vec<f32>,
    pub batch_indices_0: Vec<usize>,

    pub batch_obs_1: Vec<f32>,
    pub batch_indices_1: Vec<usize>,
}

impl BatchedMatchRunner {
    pub fn new(
        num_pairs: usize,
        base_seed: u64,
        num_sims: usize,
        is_agent1_heuristic: bool,
    ) -> Self {
        let total_games = num_pairs * 2;
        let mut workers = Vec::with_capacity(total_games);

        for i in 0..num_pairs {
            let seed = base_seed + (i as u64) * 997;
            // 局 1: agent0 是 P0, agent1 是 P1
            workers.push(MatchWorker::new(i * 2, i, seed, false));
            // 局 2: agent1 是 P0, agent0 是 P1 (严格成对消除发牌偏置)
            workers.push(MatchWorker::new(i * 2 + 1, i, seed, true));
        }

        Self {
            workers,
            num_pairs,
            total_games,
            num_sims,
            base_seed,
            c_puct: 1.5,
            is_agent1_heuristic,
            games_completed: 0,
            batch_obs_0: Vec::with_capacity(total_games * OBS_SIZE),
            batch_indices_0: Vec::with_capacity(total_games),
            batch_obs_1: Vec::with_capacity(total_games * OBS_SIZE),
            batch_indices_1: Vec::with_capacity(total_games),
        }
    }

    /// 集中轮询推进所有对局，直到所有对局均需等待 GPU 评估或全部终局
    /// 返回 `(agent0_count, agent1_count)`
    pub fn collect_batch_requests(&mut self) -> (usize, usize) {
        self.batch_obs_0.clear();
        self.batch_indices_0.clear();
        self.batch_obs_1.clear();
        self.batch_indices_1.clear();

        loop {
            let mut progressed = false;

            for w_idx in 0..self.workers.len() {
                let worker = &mut self.workers[w_idx];
                if worker.is_finished {
                    continue;
                }

                match &mut worker.state {
                    WorkerState::Idle => {}

                    WorkerState::AtDecision => {
                        worker.step_count += 1;
                        let is_game_over = matches!(worker.game.phase, TurnPhase::GameOver(_))
                            || worker.step_count > MAX_GAME_STEPS;

                        if is_game_over {
                            worker.is_finished = true;
                            worker.winner = worker.game.winner.map(|(w, _)| w);
                            worker.reason = worker.game.winner.map(|(_, r)| r);
                            worker.rounds = worker.game.round_number() as usize;
                            worker.state = WorkerState::Idle;
                            self.games_completed += 1;
                            progressed = true;
                            continue;
                        }

                        let legals = RuleEngine::legal_actions(&worker.game);
                        if legals.is_empty() {
                            worker.is_finished = true;
                            worker.winner = worker.game.winner.map(|(w, _)| w);
                            worker.reason = worker.game.winner.map(|(_, r)| r);
                            worker.rounds = worker.game.round_number() as usize;
                            worker.state = WorkerState::Idle;
                            self.games_completed += 1;
                            progressed = true;
                            continue;
                        }

                        // 单一动作微步确定性折叠
                        if legals.len() == 1 {
                            let _ = GameEngine::step(&mut worker.game, &legals[0]);
                            progressed = true;
                            continue;
                        }

                        let is_agent0 = worker.is_acting_agent0();

                        // 若为 Agent 1 且使用内置启发式 AI，直接在 CPU 执行
                        if !is_agent0 && self.is_agent1_heuristic {
                            let chosen_act =
                                match HeuristicAI::select_action(&worker.game, &mut worker.rng) {
                                    Some(act) => act,
                                    None => legals[0].clone(),
                                };
                            let _ = GameEngine::step(&mut worker.game, &chosen_act);
                            progressed = true;
                            continue;
                        }

                        // 纯 PolicyNet 直觉贪婪模式 (num_sims == 0)
                        if self.num_sims == 0 {
                            let key = fast_state_hash(&worker.game);
                            let cache = if is_agent0 {
                                &mut worker.eval_cache_0
                            } else {
                                &mut worker.eval_cache_1
                            };

                            if let Some(&(logits, _)) = cache.get(&key) {
                                let mut best_score = f32::NEG_INFINITY;
                                let mut best_act = legals[0].clone();
                                for act in &legals {
                                    let id = action_to_id(act);
                                    let logit = if id < ACTION_SIZE { logits[id] } else { 0.0 };
                                    if logit > best_score {
                                        best_score = logit;
                                        best_act = act.clone();
                                    }
                                }
                                let _ = GameEngine::step(&mut worker.game, &best_act);
                                progressed = true;
                                continue;
                            } else {
                                worker.state = WorkerState::WaitingRootEval {
                                    legals,
                                    add_dirichlet: false,
                                };
                                progressed = true;
                                continue;
                            }
                        }

                        // MCTS 模式 (num_sims > 0)
                        let effective_sims = compute_adaptive_sims(
                            &worker.game.phase,
                            legals.len(),
                            self.num_sims,
                        );
                        worker.effective_sims = effective_sims;
                        worker.sim_idx = 0;

                        let key = fast_state_hash(&worker.game);
                        let cache = if is_agent0 {
                            &mut worker.eval_cache_0
                        } else {
                            &mut worker.eval_cache_1
                        };

                        if let Some(&(logits, _)) = cache.get(&key) {
                            let root_edges = create_edges_from_logits(&legals, &logits);
                            worker.nodes.clear();
                            worker.nodes.push(Node {
                                player: worker.game.current_player,
                                visits: 1,
                                edges: root_edges,
                                is_terminal: false,
                            });
                            worker.base_sim_state = worker
                                .game
                                .determinize_for_player(worker.game.current_player, &mut worker.rng);
                            worker.state = WorkerState::Simulating;
                            progressed = true;
                        } else {
                            worker.state = WorkerState::WaitingRootEval {
                                legals,
                                add_dirichlet: false,
                            };
                            progressed = true;
                        }
                    }

                    WorkerState::Simulating => {
                        let early_stopped = if worker.nodes[0].edges.len() >= 2 {
                            let mut top1 = 0u32;
                            let mut top2 = 0u32;
                            for e in &worker.nodes[0].edges {
                                if e.visits > top1 {
                                    top2 = top1;
                                    top1 = e.visits;
                                } else if e.visits > top2 {
                                    top2 = e.visits;
                                }
                            }
                            let remaining = worker.effective_sims.saturating_sub(worker.sim_idx) as u32;
                            top1 > top2 + remaining
                        } else {
                            false
                        };

                        if early_stopped || worker.sim_idx >= worker.effective_sims {
                            let best_idx = worker
                                .nodes[0]
                                .edges
                                .iter()
                                .enumerate()
                                .max_by_key(|(_, e)| e.visits)
                                .map(|(i, _)| i)
                                .unwrap_or(0);
                            let selected_action = worker.nodes[0].edges[best_idx].action.clone();
                            let _ = GameEngine::step(&mut worker.game, &selected_action);
                            worker.state = WorkerState::AtDecision;
                            progressed = true;
                            continue;
                        }

                        if worker.sim_idx > 0 && worker.sim_idx % MIS_BLOCK_SIZE == 0 {
                            worker.base_sim_state = worker
                                .game
                                .determinize_for_player(worker.game.current_player, &mut worker.rng);
                        }
                        let mut sim_state = worker.base_sim_state;
                        let mut curr_node_idx = 0;
                        let mut path = Vec::with_capacity(16);

                        while !worker.nodes[curr_node_idx].is_terminal
                            && !worker.nodes[curr_node_idx].edges.is_empty()
                        {
                            let best_edge_idx =
                                worker.nodes[curr_node_idx].select_best_edge(self.c_puct);
                            let action =
                                worker.nodes[curr_node_idx].edges[best_edge_idx].action.clone();
                            if GameEngine::step(&mut sim_state, &action).is_err() {
                                break;
                            }
                            collapse_deterministic_micro_steps(&mut sim_state);
                            path.push((curr_node_idx, best_edge_idx));

                            if let Some(child_idx) =
                                worker.nodes[curr_node_idx].edges[best_edge_idx].child_idx
                            {
                                curr_node_idx = child_idx;
                            } else {
                                break;
                            }
                        }

                        if path.is_empty() {
                            worker.sim_idx += 1;
                            progressed = true;
                            continue;
                        }

                        let &(last_n, last_e) = path.last().unwrap();
                        let existing_child = worker.nodes[last_n].edges[last_e].child_idx;

                        if matches!(sim_state.phase, TurnPhase::GameOver(_)) {
                            let win_v = match sim_state.winner.map(|(w, _)| w) {
                                Some(0) => 1.0,
                                Some(1) => -1.0,
                                _ => 0.0,
                            };
                            if existing_child.is_none() {
                                let new_node_idx = worker.nodes.len();
                                worker.nodes.push(Node {
                                    player: sim_state.current_player,
                                    visits: 1,
                                    edges: Vec::new(),
                                    is_terminal: true,
                                });
                                worker.nodes[last_n].edges[last_e].child_idx = Some(new_node_idx);
                            }
                            for &(n_idx, e_idx) in &path {
                                worker.nodes[n_idx].visits += 1;
                                worker.nodes[n_idx].edges[e_idx].visits += 1;
                                worker.nodes[n_idx].edges[e_idx].w_p0 += win_v;
                            }
                            worker.sim_idx += 1;
                            progressed = true;
                            continue;
                        }

                        let next_legals = RuleEngine::legal_actions(&sim_state);
                        if next_legals.is_empty() {
                            if existing_child.is_none() {
                                let new_node_idx = worker.nodes.len();
                                worker.nodes.push(Node {
                                    player: sim_state.current_player,
                                    visits: 1,
                                    edges: Vec::new(),
                                    is_terminal: true,
                                });
                                worker.nodes[last_n].edges[last_e].child_idx = Some(new_node_idx);
                            }
                            for &(n_idx, e_idx) in &path {
                                worker.nodes[n_idx].visits += 1;
                                worker.nodes[n_idx].edges[e_idx].visits += 1;
                            }
                            worker.sim_idx += 1;
                            progressed = true;
                            continue;
                        }

                        let is_agent0 = worker.is_acting_agent0();
                        let key = fast_state_hash(&sim_state);
                        let cache = if is_agent0 {
                            &mut worker.eval_cache_0
                        } else {
                            &mut worker.eval_cache_1
                        };

                        if let Some(&(logits, pred)) = cache.get(&key) {
                            let edges = create_edges_from_logits(&next_legals, &logits);
                            let v_mover = pred.combined_value(LAMBDA_TURNS);
                            let vp0 = if sim_state.current_player == 0 {
                                v_mover
                            } else {
                                -v_mover
                            };
                            if existing_child.is_none() {
                                let new_node_idx = worker.nodes.len();
                                worker.nodes.push(Node {
                                    player: sim_state.current_player,
                                    visits: 1,
                                    edges,
                                    is_terminal: false,
                                });
                                worker.nodes[last_n].edges[last_e].child_idx = Some(new_node_idx);
                            }
                            for &(n_idx, e_idx) in &path {
                                worker.nodes[n_idx].visits += 1;
                                worker.nodes[n_idx].edges[e_idx].visits += 1;
                                worker.nodes[n_idx].edges[e_idx].w_p0 += vp0;
                            }
                            worker.sim_idx += 1;
                            progressed = true;
                            continue;
                        }

                        worker.state = WorkerState::WaitingLeafEval {
                            sim_state,
                            path,
                            legals: next_legals,
                        };
                        progressed = true;
                    }

                    WorkerState::WaitingRootEval { .. } | WorkerState::WaitingLeafEval { .. } => {}
                }
            }

            if !progressed {
                break;
            }
        }

        // 分流打包 Agent 0 与 Agent 1 的盘面
        for (w_idx, worker) in self.workers.iter().enumerate() {
            if worker.is_finished {
                continue;
            }
            let is_agent0 = worker.is_acting_agent0();
            match &worker.state {
                WorkerState::WaitingRootEval { .. } => {
                    let obs = encode_state(&worker.game);
                    if is_agent0 {
                        self.batch_obs_0.extend_from_slice(&obs);
                        self.batch_indices_0.push(w_idx);
                    } else {
                        self.batch_obs_1.extend_from_slice(&obs);
                        self.batch_indices_1.push(w_idx);
                    }
                }
                WorkerState::WaitingLeafEval { sim_state, .. } => {
                    let obs = encode_state(sim_state);
                    if is_agent0 {
                        self.batch_obs_0.extend_from_slice(&obs);
                        self.batch_indices_0.push(w_idx);
                    } else {
                        self.batch_obs_1.extend_from_slice(&obs);
                        self.batch_indices_1.push(w_idx);
                    }
                }
                _ => {}
            }
        }

        (self.batch_indices_0.len(), self.batch_indices_1.len())
    }

    /// 将 GPU 对 Agent 0 (候选模型) 的预测结果分发回流
    pub fn apply_batch_results_0(&mut self, fused_slice: &[f32]) {
        let count = self.batch_indices_0.len();
        assert_eq!(
            fused_slice.len(),
            count * FUSED_OUTPUT_SIZE,
            "fused_slice_0 length mismatch"
        );

        for (i, &w_idx) in self.batch_indices_0.iter().enumerate() {
            let offset = i * FUSED_OUTPUT_SIZE;
            let mut logits = [0.0f32; ACTION_SIZE];
            logits.copy_from_slice(&fused_slice[offset..offset + ACTION_SIZE]);
            let win_val = fused_slice[offset + ACTION_SIZE];
            let turns_val = fused_slice[offset + ACTION_SIZE + 1];
            let mut reason_probs = [0.0f32; 3];
            reason_probs.copy_from_slice(
                &fused_slice[offset + ACTION_SIZE + 2..offset + ACTION_SIZE + 5],
            );

            let pred = NeuralPrediction {
                win_value: win_val,
                turns_value: turns_val,
                reason_probs,
            };

            let worker = &mut self.workers[w_idx];
            let cur_state = std::mem::replace(&mut worker.state, WorkerState::Idle);

            match cur_state {
                WorkerState::WaitingRootEval { legals, .. } => {
                    let key = fast_state_hash(&worker.game);
                    worker.eval_cache_0.insert(key, (logits, pred));

                    if self.num_sims == 0 {
                        let mut best_score = f32::NEG_INFINITY;
                        let mut best_act = legals[0].clone();
                        for act in &legals {
                            let id = action_to_id(act);
                            let logit = if id < ACTION_SIZE { logits[id] } else { 0.0 };
                            if logit > best_score {
                                best_score = logit;
                                best_act = act.clone();
                            }
                        }
                        let _ = GameEngine::step(&mut worker.game, &best_act);
                        worker.state = WorkerState::AtDecision;
                    } else {
                        let root_edges = create_edges_from_logits(&legals, &logits);
                        worker.nodes.clear();
                        worker.nodes.push(Node {
                            player: worker.game.current_player,
                            visits: 1,
                            edges: root_edges,
                            is_terminal: false,
                        });
                        worker.base_sim_state = worker
                            .game
                            .determinize_for_player(worker.game.current_player, &mut worker.rng);
                        worker.state = WorkerState::Simulating;
                    }
                }

                WorkerState::WaitingLeafEval {
                    sim_state,
                    path,
                    legals,
                } => {
                    let key = fast_state_hash(&sim_state);
                    worker.eval_cache_0.insert(key, (logits, pred));

                    let edges = create_edges_from_logits(&legals, &logits);
                    let v_mover = pred.combined_value(LAMBDA_TURNS);
                    let vp0 = if sim_state.current_player == 0 {
                        v_mover
                    } else {
                        -v_mover
                    };

                    let (last_n, last_e) = *path.last().unwrap();
                    if worker.nodes[last_n].edges[last_e].child_idx.is_none() {
                        let new_node_idx = worker.nodes.len();
                        worker.nodes.push(Node {
                            player: sim_state.current_player,
                            visits: 1,
                            edges,
                            is_terminal: false,
                        });
                        worker.nodes[last_n].edges[last_e].child_idx = Some(new_node_idx);
                    }

                    for &(n_idx, e_idx) in &path {
                        worker.nodes[n_idx].visits += 1;
                        worker.nodes[n_idx].edges[e_idx].visits += 1;
                        worker.nodes[n_idx].edges[e_idx].w_p0 += vp0;
                    }
                    worker.sim_idx += 1;
                    worker.state = WorkerState::Simulating;
                }

                _ => {}
            }
        }
    }

    /// 将 GPU 对 Agent 1 (基准模型) 的预测结果分发回流
    pub fn apply_batch_results_1(&mut self, fused_slice: &[f32]) {
        let count = self.batch_indices_1.len();
        assert_eq!(
            fused_slice.len(),
            count * FUSED_OUTPUT_SIZE,
            "fused_slice_1 length mismatch"
        );

        for (i, &w_idx) in self.batch_indices_1.iter().enumerate() {
            let offset = i * FUSED_OUTPUT_SIZE;
            let mut logits = [0.0f32; ACTION_SIZE];
            logits.copy_from_slice(&fused_slice[offset..offset + ACTION_SIZE]);
            let win_val = fused_slice[offset + ACTION_SIZE];
            let turns_val = fused_slice[offset + ACTION_SIZE + 1];
            let mut reason_probs = [0.0f32; 3];
            reason_probs.copy_from_slice(
                &fused_slice[offset + ACTION_SIZE + 2..offset + ACTION_SIZE + 5],
            );

            let pred = NeuralPrediction {
                win_value: win_val,
                turns_value: turns_val,
                reason_probs,
            };

            let worker = &mut self.workers[w_idx];
            let cur_state = std::mem::replace(&mut worker.state, WorkerState::Idle);

            match cur_state {
                WorkerState::WaitingRootEval { legals, .. } => {
                    let key = fast_state_hash(&worker.game);
                    worker.eval_cache_1.insert(key, (logits, pred));

                    if self.num_sims == 0 {
                        let mut best_score = f32::NEG_INFINITY;
                        let mut best_act = legals[0].clone();
                        for act in &legals {
                            let id = action_to_id(act);
                            let logit = if id < ACTION_SIZE { logits[id] } else { 0.0 };
                            if logit > best_score {
                                best_score = logit;
                                best_act = act.clone();
                            }
                        }
                        let _ = GameEngine::step(&mut worker.game, &best_act);
                        worker.state = WorkerState::AtDecision;
                    } else {
                        let root_edges = create_edges_from_logits(&legals, &logits);
                        worker.nodes.clear();
                        worker.nodes.push(Node {
                            player: worker.game.current_player,
                            visits: 1,
                            edges: root_edges,
                            is_terminal: false,
                        });
                        worker.base_sim_state = worker
                            .game
                            .determinize_for_player(worker.game.current_player, &mut worker.rng);
                        worker.state = WorkerState::Simulating;
                    }
                }

                WorkerState::WaitingLeafEval {
                    sim_state,
                    path,
                    legals,
                } => {
                    let key = fast_state_hash(&sim_state);
                    worker.eval_cache_1.insert(key, (logits, pred));

                    let edges = create_edges_from_logits(&legals, &logits);
                    let v_mover = pred.combined_value(LAMBDA_TURNS);
                    let vp0 = if sim_state.current_player == 0 {
                        v_mover
                    } else {
                        -v_mover
                    };

                    let (last_n, last_e) = *path.last().unwrap();
                    if worker.nodes[last_n].edges[last_e].child_idx.is_none() {
                        let new_node_idx = worker.nodes.len();
                        worker.nodes.push(Node {
                            player: sim_state.current_player,
                            visits: 1,
                            edges,
                            is_terminal: false,
                        });
                        worker.nodes[last_n].edges[last_e].child_idx = Some(new_node_idx);
                    }

                    for &(n_idx, e_idx) in &path {
                        worker.nodes[n_idx].visits += 1;
                        worker.nodes[n_idx].edges[e_idx].visits += 1;
                        worker.nodes[n_idx].edges[e_idx].w_p0 += vp0;
                    }
                    worker.sim_idx += 1;
                    worker.state = WorkerState::Simulating;
                }

                _ => {}
            }
        }
    }

    #[inline]
    pub fn is_finished(&self) -> bool {
        self.games_completed >= self.total_games
    }

    /// 聚合评测统计结果
    pub fn finalize_match_result(self) -> ParallelMatchResult {
        let mut res = ParallelMatchResult {
            total_games: self.total_games,
            ..Default::default()
        };

        for w in self.workers {
            res.total_steps += w.step_count;
            res.total_rounds += w.rounds;

            match w.winner {
                Some(winner) => {
                    let agent0_won = if !w.is_swap {
                        winner == 0
                    } else {
                        winner == 1
                    };

                    if agent0_won {
                        res.agent0_wins += 1;
                        res.agent0_win_steps += w.step_count;
                        res.agent0_win_rounds += w.rounds;
                    } else {
                        res.agent1_wins += 1;
                        res.agent0_lose_steps += w.step_count;
                        res.agent0_lose_rounds += w.rounds;
                    }

                    if winner == 0 {
                        res.p0_seat_wins += 1;
                    } else {
                        res.p1_seat_wins += 1;
                    }

                    match w.reason {
                        Some(VictoryReason::TwentyPrestigePoints) => res.reasons_20_pts += 1,
                        Some(VictoryReason::TenCrowns) => res.reasons_10_crowns += 1,
                        Some(VictoryReason::TenPointsSameColor(_)) => res.reasons_10_color += 1,
                        None => {}
                    }
                }
                None => {
                    res.draws += 1;
                }
            }
        }

        res
    }
}
