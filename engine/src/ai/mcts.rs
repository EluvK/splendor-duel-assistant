use rand::prelude::*;
use rand_distr::multi::{Dirichlet, MultiDistribution};
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hasher};

use crate::ai::heuristic_ai::HeuristicAI;
use crate::ai::neural_evaluator::{NeuralPrediction, TractNeuralEvaluator};
use crate::bridge::{action_to_id, encode_state, ACTION_SIZE, OBS_SIZE};
use crate::game_state::phase::TurnPhase;
use crate::game_state::state::GameState;
use crate::gameplay::engine::GameEngine;
use crate::gameplay::rules::RuleEngine;
use crate::model::action::Action;

#[inline]
fn hash_obs(obs: &[f32; OBS_SIZE]) -> u64 {
    let mut hasher = DefaultHasher::new();
    let bytes = unsafe {
        std::slice::from_raw_parts(obs.as_ptr() as *const u8, std::mem::size_of_val(obs))
    };
    hasher.write(bytes);
    hasher.finish()
}

/// 子节点边
struct Edge {
    action: Action,
    prior: f32,       // 动作先验概率 P(s, a)
    visits: u32,      // 访问次数 N
    w_p0: f32,        // Player-0 绝对累积价值 W
    child_idx: Option<usize>, // 若已展开则指向子节点索引
}

impl Edge {
    #[inline]
    fn q_p0(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.w_p0 / (self.visits as f32)
        }
    }
}

/// 树节点
struct Node {
    player: usize,
    visits: u32,
    edges: Vec<Edge>,
    is_terminal: bool,
}

/// MCTS 搜索综合价值中的时间敏感度惩罚系数 (鼓励快速斩杀，惩罚拖延苟活)
pub const LAMBDA_TURNS: f32 = 0.20;

/// 多重确定化信息集洗牌块大小 (MIS-MCTS: 每隔 K 次模拟重抽暗牌，兼顾无偏估计与局部备份一致性)
pub const MIS_BLOCK_SIZE: usize = 8;

/// 自动折叠确定性单选项微步 (预留唯一黄金、单一合法弃牌)，压缩搜索树无谓深度
#[inline]
fn collapse_deterministic_micro_steps(sim_state: &mut GameState) {
    loop {
        match sim_state.phase {
            TurnPhase::SelectReserveGold => {
                let legals = RuleEngine::legal_actions(sim_state);
                if legals.len() == 1 {
                    let _ = GameEngine::step(sim_state, &legals[0]);
                } else {
                    break;
                }
            }
            TurnPhase::DiscardTokens => {
                let legals = RuleEngine::legal_actions(sim_state);
                if legals.len() == 1 {
                    let _ = GameEngine::step(sim_state, &legals[0]);
                } else {
                    break;
                }
            }
            _ => break,
        }
    }
}

/// 基于先验剪枝的高性能 AlphaZero 风格 MCTS
pub struct RustMCTS {
    c_puct: f32,
    max_rollout_steps: usize,
}

impl Default for RustMCTS {
    fn default() -> Self {
        Self {
            c_puct: 1.5,
            max_rollout_steps: 30,
        }
    }
}

impl RustMCTS {
    pub fn new(c_puct: f32, max_rollout_steps: usize) -> Self {
        Self {
            c_puct,
            max_rollout_steps,
        }
    }

    /// 执行 MCTS 搜索并返回推荐的最佳动作 (确定性贪婪模式)
    pub fn search<R: Rng + ?Sized>(
        &self,
        state: &GameState,
        num_simulations: usize,
        rng: &mut R,
    ) -> Option<Action> {
        self.search_with_exploration(state, num_simulations, false, 0.3, 0.25, 0.0, rng)
    }

    /// 执行带 AlphaZero 探索机制的 MCTS 搜索 (支持根节点 Dirichlet 噪声与温度轮盘赌采样)
    pub fn search_with_exploration<R: Rng + ?Sized>(
        &self,
        state: &GameState,
        num_simulations: usize,
        add_dirichlet: bool,
        dirichlet_alpha: f32,
        dirichlet_eps: f32,
        temperature: f32,
        rng: &mut R,
    ) -> Option<Action> {
        self.search_with_exploration_policy(
            state,
            num_simulations,
            add_dirichlet,
            dirichlet_alpha,
            dirichlet_eps,
            temperature,
            rng,
        )
        .map(|(a, _)| a)
    }

    /// 执行带启发式评估的 MCTS 搜索并返回选择的动作以及完整的 288 维访问频次软策略分布
    pub fn search_with_exploration_policy<R: Rng + ?Sized>(
        &self,
        state: &GameState,
        num_simulations: usize,
        add_dirichlet: bool,
        dirichlet_alpha: f32,
        dirichlet_eps: f32,
        temperature: f32,
        rng: &mut R,
    ) -> Option<(Action, [f32; ACTION_SIZE])> {
        let legals = RuleEngine::legal_actions(state);
        if legals.is_empty() {
            return None;
        }
        // 零开销快捷路径：单一动作免搜索直接返回
        if legals.len() == 1 {
            let mut policy = [0.0f32; ACTION_SIZE];
            let id = action_to_id(&legals[0]);
            if id < ACTION_SIZE {
                policy[id] = 1.0;
            }
            return Some((legals[0].clone(), policy));
        }

        let mut nodes: Vec<Node> = Vec::with_capacity(num_simulations * 2);
        let root_idx = 0;
        let is_term = matches!(state.phase, TurnPhase::GameOver(_));

        // 根节点利用启发式先验展开所有合法分支
        let mut root_edges = Self::create_edges_with_priors(state, legals);

        // 注入 Dirichlet 探索噪声 (强行给次优分支分配搜索预算，打破开局盲区)
        if add_dirichlet && root_edges.len() >= 2 {
            let alphas = vec![dirichlet_alpha; root_edges.len()];
            if let Ok(dir) = Dirichlet::new(&alphas) {
                let mut noise = vec![0.0f32; root_edges.len()];
                dir.sample_to_slice(rng, &mut noise);
                let mut sum = 0.0f32;
                for (i, edge) in root_edges.iter_mut().enumerate() {
                    edge.prior = (1.0 - dirichlet_eps) * edge.prior + dirichlet_eps * noise[i];
                    sum += edge.prior;
                }
                if sum > 1e-6 {
                    for edge in root_edges.iter_mut() {
                        edge.prior /= sum;
                    }
                }
            }
        }

        nodes.push(Node {
            player: state.current_player,
            visits: 1,
            edges: root_edges,
            is_terminal: is_term,
        });

        let mut base_sim_state = state.determinize_for_player(state.current_player, rng);
        for sim_idx in 0..num_simulations {
            if sim_idx > 0 && sim_idx % MIS_BLOCK_SIZE == 0 {
                base_sim_state = state.determinize_for_player(state.current_player, rng);
            }
            let mut sim_state = base_sim_state.clone();
            let mut curr_node_idx = root_idx;
            // 记录沿途 (node_idx, edge_idx)
            let mut path: Vec<(usize, usize)> = Vec::with_capacity(16);

            // 1. Selection: 沿着树向下选择 PUCT 最大分支
            while !nodes[curr_node_idx].is_terminal && !nodes[curr_node_idx].edges.is_empty() {
                let best_edge_idx = self.select_best_edge(&nodes[curr_node_idx]);
                let action = nodes[curr_node_idx].edges[best_edge_idx].action.clone();
                if GameEngine::step(&mut sim_state, &action).is_err() {
                    break;
                }
                collapse_deterministic_micro_steps(&mut sim_state);
                path.push((curr_node_idx, best_edge_idx));

                // 若该边已有子节点，继续向下探索；否则在当前叶子停止展开
                if let Some(child_idx) = nodes[curr_node_idx].edges[best_edge_idx].child_idx {
                    curr_node_idx = child_idx;
                } else {
                    break;
                }
            }

            if path.is_empty() {
                continue;
            }

            // 2. Expansion: 为当前选中的末梢边展开新子节点
            let &(last_node_idx, last_edge_idx) = path.last().unwrap();
            let next_is_term = matches!(sim_state.phase, TurnPhase::GameOver(_));
            let new_node_idx = nodes.len();

            let next_legals = RuleEngine::legal_actions(&sim_state);
            let next_edges = if next_is_term {
                Vec::new()
            } else {
                Self::create_edges_with_priors(&sim_state, next_legals)
            };

            nodes.push(Node {
                player: sim_state.current_player,
                visits: 0,
                edges: next_edges,
                is_terminal: next_is_term,
            });
            nodes[last_node_idx].edges[last_edge_idx].child_idx = Some(new_node_idx);

            // 3. Evaluation / Rollout
            let v_p0 = self.evaluate_or_rollout(&mut sim_state, rng);

            // 4. Backup: 沿路径反向回传 Player-0 绝对价值
            for &(n_idx, e_idx) in path.iter() {
                nodes[n_idx].visits += 1;
                nodes[n_idx].edges[e_idx].visits += 1;
                nodes[n_idx].edges[e_idx].w_p0 += v_p0;
            }

            // 自适应早停判断：仅在确定性贪婪模式 (temperature <= 0.01) 下生效
            // 若根节点第一分支访问量 N1 与第二分支访问量 N2 的差值大于剩余推演次数，
            // 则即使剩余推演全部给 N2，N2 也绝对无法反超，提前安全截断
            if temperature <= 0.01 && nodes[root_idx].edges.len() >= 2 {
                let remaining = (num_simulations - 1 - sim_idx) as u32;
                let mut max_visits = 0;
                let mut second_max_visits = 0;
                for edge in &nodes[root_idx].edges {
                    if edge.visits > max_visits {
                        second_max_visits = max_visits;
                        max_visits = edge.visits;
                    } else if edge.visits > second_max_visits {
                        second_max_visits = edge.visits;
                    }
                }
                if max_visits.saturating_sub(second_max_visits) > remaining {
                    break;
                }
            }
        }

        Self::extract_policy_distribution(&nodes[root_idx].edges, temperature, rng)
    }

    /// 从根节点分支访问量中提取完整的 288 维软策略分布并进行采样
    #[inline]
    fn extract_policy_distribution<R: Rng + ?Sized>(
        edges: &[Edge],
        temperature: f32,
        rng: &mut R,
    ) -> Option<(Action, [f32; ACTION_SIZE])> {
        if edges.is_empty() {
            return None;
        }
        let mut policy = [0.0f32; ACTION_SIZE];
        if temperature <= 0.01 {
            let best_edge = edges.iter().max_by_key(|e| e.visits)?;
            let best_id = action_to_id(&best_edge.action);
            if best_id < ACTION_SIZE {
                policy[best_id] = 1.0;
            }
            Some((best_edge.action.clone(), policy))
        } else {
            let inv_temp = 1.0 / temperature;
            let mut exp_visits = Vec::with_capacity(edges.len());
            let mut sum_v = 0.0f32;
            for edge in edges.iter() {
                let v = (edge.visits as f32).powf(inv_temp);
                exp_visits.push(v);
                sum_v += v;
            }
            if sum_v <= 1e-6 {
                let first_edge = edges.first()?;
                let first_id = action_to_id(&first_edge.action);
                if first_id < ACTION_SIZE {
                    policy[first_id] = 1.0;
                }
                return Some((first_edge.action.clone(), policy));
            }

            for (i, edge) in edges.iter().enumerate() {
                let id = action_to_id(&edge.action);
                if id < ACTION_SIZE {
                    policy[id] = exp_visits[i] / sum_v;
                }
            }

            let mut pick = rng.random_range(0.0..sum_v);
            let mut chosen_action = edges.last().unwrap().action.clone();
            for (i, &v) in exp_visits.iter().enumerate() {
                if pick <= v {
                    chosen_action = edges[i].action.clone();
                    break;
                }
                pick -= v;
            }
            Some((chosen_action, policy))
        }
    }

    /// 执行带纯神经网络指导与 AlphaZero 探索机制的 MCTS 搜索 (完全脱离启发式打分与模拟)
    pub fn search_neural_with_exploration<R: Rng + ?Sized>(
        &self,
        state: &GameState,
        evaluator: &TractNeuralEvaluator,
        num_simulations: usize,
        add_dirichlet: bool,
        dirichlet_alpha: f32,
        dirichlet_eps: f32,
        temperature: f32,
        rng: &mut R,
    ) -> Option<Action> {
        let legals = RuleEngine::legal_actions(state);
        self.search_neural_with_exploration_and_legals(
            state,
            legals,
            evaluator,
            num_simulations,
            add_dirichlet,
            dirichlet_alpha,
            dirichlet_eps,
            temperature,
            rng,
        )
    }

    /// 执行带纯神经网络指导与 AlphaZero 探索机制的 MCTS 搜索 (支持复用外部已生成的合法动作列表与评估缓存)
    pub fn search_neural_with_exploration_and_legals<R: Rng + ?Sized>(
        &self,
        state: &GameState,
        legals: Vec<Action>,
        evaluator: &TractNeuralEvaluator,
        num_simulations: usize,
        add_dirichlet: bool,
        dirichlet_alpha: f32,
        dirichlet_eps: f32,
        temperature: f32,
        rng: &mut R,
    ) -> Option<Action> {
        self.search_neural_policy_with_legals(
            state,
            legals,
            evaluator,
            num_simulations,
            add_dirichlet,
            dirichlet_alpha,
            dirichlet_eps,
            temperature,
            rng,
        )
        .map(|(a, _)| a)
    }

    /// 执行带纯神经网络指导与 AlphaZero 探索机制的 MCTS 搜索并返回选择的动作以及完整的 288 维软策略分布
    pub fn search_neural_policy_with_legals<R: Rng + ?Sized>(
        &self,
        state: &GameState,
        legals: Vec<Action>,
        evaluator: &TractNeuralEvaluator,
        num_simulations: usize,
        add_dirichlet: bool,
        dirichlet_alpha: f32,
        dirichlet_eps: f32,
        temperature: f32,
        rng: &mut R,
    ) -> Option<(Action, [f32; ACTION_SIZE])> {
        if legals.is_empty() {
            return None;
        }
        if legals.len() == 1 {
            let mut policy = [0.0f32; ACTION_SIZE];
            let id = action_to_id(&legals[0]);
            if id < ACTION_SIZE {
                policy[id] = 1.0;
            }
            return Some((legals[0].clone(), policy));
        }

        let mut nodes: Vec<Node> = Vec::with_capacity(num_simulations * 2);
        let mut eval_cache: HashMap<u64, ([f32; ACTION_SIZE], NeuralPrediction)> =
            HashMap::with_capacity(num_simulations + 1);
        let root_idx = 0;
        let is_term = matches!(state.phase, TurnPhase::GameOver(_));

        let (mut root_edges, _root_pred) =
            Self::create_edges_with_neural_priors_cached(state, legals, evaluator, &mut eval_cache).ok()?;

        if add_dirichlet && root_edges.len() >= 2 {
            let alphas = vec![dirichlet_alpha; root_edges.len()];
            if let Ok(dir) = Dirichlet::new(&alphas) {
                let mut noise = vec![0.0f32; root_edges.len()];
                dir.sample_to_slice(rng, &mut noise);
                let mut sum = 0.0f32;
                for (i, edge) in root_edges.iter_mut().enumerate() {
                    edge.prior = (1.0 - dirichlet_eps) * edge.prior + dirichlet_eps * noise[i];
                    sum += edge.prior;
                }
                if sum > 1e-6 {
                    for edge in root_edges.iter_mut() {
                        edge.prior /= sum;
                    }
                }
            }
        }

        nodes.push(Node {
            player: state.current_player,
            visits: 1,
            edges: root_edges,
            is_terminal: is_term,
        });

        // 根节点如果也是游戏结束状态则直接返回
        if is_term {
            return None;
        }

        let mut base_sim_state = state.determinize_for_player(state.current_player, rng);
        for sim_idx in 0..num_simulations {
            if sim_idx > 0 && sim_idx % MIS_BLOCK_SIZE == 0 {
                base_sim_state = state.determinize_for_player(state.current_player, rng);
            }
            let mut sim_state = base_sim_state.clone();
            let mut curr_node_idx = root_idx;
            let mut path: Vec<(usize, usize)> = Vec::with_capacity(16);

            // 1. Selection
            while !nodes[curr_node_idx].is_terminal && !nodes[curr_node_idx].edges.is_empty() {
                let best_edge_idx = self.select_best_edge(&nodes[curr_node_idx]);
                let action = nodes[curr_node_idx].edges[best_edge_idx].action.clone();
                if GameEngine::step(&mut sim_state, &action).is_err() {
                    break;
                }
                collapse_deterministic_micro_steps(&mut sim_state);
                path.push((curr_node_idx, best_edge_idx));

                if let Some(child_idx) = nodes[curr_node_idx].edges[best_edge_idx].child_idx {
                    curr_node_idx = child_idx;
                } else {
                    break;
                }
            }

            if path.is_empty() {
                continue;
            }

            // 2. Expansion & Evaluation
            let &(last_node_idx, last_edge_idx) = path.last().unwrap();
            let next_is_term = matches!(sim_state.phase, TurnPhase::GameOver(_));
            let new_node_idx = nodes.len();

            let (next_edges, v_p0) = if next_is_term {
                let win_v = match sim_state.winner.map(|(w, _)| w) {
                    Some(0) => 1.0,
                    Some(1) => -1.0,
                    _ => 0.0,
                };
                (Vec::new(), win_v)
            } else {
                let next_legals = RuleEngine::legal_actions(&sim_state);
                if next_legals.is_empty() {
                    (Vec::new(), 0.0)
                } else {
                    match Self::create_edges_with_neural_priors_cached(
                        &sim_state,
                        next_legals,
                        evaluator,
                        &mut eval_cache,
                    ) {
                        Ok((edges, pred)) => {
                            let v_mover = pred.combined_value(LAMBDA_TURNS);
                            let vp0 = if sim_state.current_player == 0 {
                                v_mover
                            } else {
                                -v_mover
                            };
                            (edges, vp0)
                        }
                        Err(_) => (Vec::new(), 0.0),
                    }
                }
            };

            nodes.push(Node {
                player: sim_state.current_player,
                visits: 1,
                edges: next_edges,
                is_terminal: next_is_term,
            });
            nodes[last_node_idx].edges[last_edge_idx].child_idx = Some(new_node_idx);

            // 3. Backup
            for &(n_idx, e_idx) in path.iter() {
                nodes[n_idx].visits += 1;
                nodes[n_idx].edges[e_idx].visits += 1;
                nodes[n_idx].edges[e_idx].w_p0 += v_p0;
            }

            // 自适应早停判断：仅在确定性贪婪模式 (temperature <= 0.01) 下生效
            // 若根节点第一分支访问量 N1 与第二分支访问量 N2 的差值大于剩余推演次数，
            // 则即使剩余推演全部给 N2，N2 也绝对无法反超，提前安全截断
            if temperature <= 0.01 && nodes[root_idx].edges.len() >= 2 {
                let remaining = (num_simulations - 1 - sim_idx) as u32;
                let mut max_visits = 0;
                let mut second_max_visits = 0;
                for edge in &nodes[root_idx].edges {
                    if edge.visits > max_visits {
                        second_max_visits = max_visits;
                        max_visits = edge.visits;
                    } else if edge.visits > second_max_visits {
                        second_max_visits = edge.visits;
                    }
                }
                if max_visits.saturating_sub(second_max_visits) > remaining {
                    break;
                }
            }
        }

        Self::extract_policy_distribution(&nodes[root_idx].edges, temperature, rng)
    }

    /// 使用神经网络提供先验概率与状态估值 (带缓存支持)
    fn create_edges_with_neural_priors_cached(
        state: &GameState,
        legals: Vec<Action>,
        evaluator: &TractNeuralEvaluator,
        cache: &mut HashMap<u64, ([f32; ACTION_SIZE], NeuralPrediction)>,
    ) -> Result<(Vec<Edge>, NeuralPrediction), String> {
        let obs = encode_state(state);
        let key = hash_obs(&obs);

        let (logits, pred) = if let Some(cached) = cache.get(&key) {
            *cached
        } else {
            let res = evaluator.evaluate(&obs)?;
            cache.insert(key, res);
            res
        };

        let mut scores = Vec::with_capacity(legals.len());
        let mut max_logit = f32::NEG_INFINITY;

        for act in legals.iter() {
            let id = action_to_id(act);
            let logit = if id < ACTION_SIZE { logits[id] } else { 0.0 };
            max_logit = max_logit.max(logit);
            scores.push(logit);
        }

        let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_logit).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum::<f32>().max(1e-6);

        let edges = legals
            .into_iter()
            .enumerate()
            .map(|(i, action)| Edge {
                action,
                prior: exp_scores[i] / sum_exp,
                visits: 0,
                w_p0: 0.0,
                child_idx: None,
            })
            .collect();

        Ok((edges, pred))
    }

    #[allow(dead_code)]
    fn create_edges_with_neural_priors(
        state: &GameState,
        legals: Vec<Action>,
        evaluator: &TractNeuralEvaluator,
    ) -> Result<(Vec<Edge>, NeuralPrediction), String> {
        let mut cache = HashMap::new();
        Self::create_edges_with_neural_priors_cached(state, legals, evaluator, &mut cache)
    }

    /// 使用先验打分并做平滑 Softmax 归一化初始化分支
    fn create_edges_with_priors(state: &GameState, legals: Vec<Action>) -> Vec<Edge> {
        let mut scores = Vec::with_capacity(legals.len());
        let mut max_score = f32::NEG_INFINITY;

        for act in legals.iter() {
            // 获取动作评估分
            let s = HeuristicAI::evaluate_action(state, act);
            max_score = max_score.max(s);
            scores.push(s);
        }

        // 经由带温度的 Softmax 得到先验概率
        let temperature = 15.0; // 适当平滑，兼顾探索
        let exp_scores: Vec<f32> = scores
            .iter()
            .map(|&s| ((s - max_score) / temperature).exp())
            .collect();
        let sum_exp: f32 = exp_scores.iter().sum::<f32>().max(1e-6);

        legals
            .into_iter()
            .enumerate()
            .map(|(i, action)| Edge {
                action,
                prior: exp_scores[i] / sum_exp,
                visits: 0,
                w_p0: 0.0,
                child_idx: None,
            })
            .collect()
    }

    fn select_best_edge(&self, node: &Node) -> usize {
        let total_sqrt = (node.visits as f32).sqrt().max(1.0);
        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx = 0;

        for (i, edge) in node.edges.iter().enumerate() {
            // Q 从当前决策者 (node.player) 视角计算
            let q_mover = if node.player == 0 {
                edge.q_p0()
            } else {
                -edge.q_p0()
            };

            // PUCT 公式: Q + c * P * (sqrt(N) / (1 + n))
            let uct = self.c_puct * edge.prior * (total_sqrt / (1.0 + edge.visits as f32));
            let score = q_mover + uct;

            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        best_idx
    }

    /// 极速前瞻评估：终局直接判定，未终局向前启发式推演并计算终态局势差
    fn evaluate_or_rollout<R: Rng + ?Sized>(&self, state: &mut GameState, rng: &mut R) -> f32 {
        if let TurnPhase::GameOver(_) = state.phase {
            return match state.winner.map(|(w, _)| w) {
                Some(0) => 1.0,
                Some(1) => -1.0,
                _ => 0.0,
            };
        }

        for _ in 0..self.max_rollout_steps {
            if matches!(state.phase, TurnPhase::GameOver(_)) {
                break;
            }
            if let Some(act) = HeuristicAI::select_action(state, rng) {
                if GameEngine::step(state, &act).is_err() {
                    break;
                }
            } else {
                break;
            }
        }

        if let TurnPhase::GameOver(_) = state.phase {
            return match state.winner.map(|(w, _)| w) {
                Some(0) => 1.0,
                Some(1) => -1.0,
                _ => 0.0,
            };
        }

        // 平滑综合局势差
        let p0 = &state.players[0];
        let p1 = &state.players[1];

        let pt_diff = (p0.total_points as f32 - p1.total_points as f32) / 20.0;
        let crown_diff = (p0.total_crowns as f32 - p1.total_crowns as f32) / 10.0;
        let priv_diff = (p0.privileges as f32 - p1.privileges as f32) / 3.0;

        let eval = pt_diff + 0.6 * crown_diff + 0.2 * priv_diff;
        eval.clamp(-1.0, 1.0)
    }
}
