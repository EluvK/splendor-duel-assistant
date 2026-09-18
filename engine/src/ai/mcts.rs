use rand::prelude::*;
use rand_distr::multi::{Dirichlet, MultiDistribution};
use std::collections::HashMap;

use crate::ai::neural_evaluator::{NeuralPrediction, TractNeuralEvaluator};
use crate::bridge::{ACTION_SIZE, action_to_id, encode_state};
use crate::game_state::phase::TurnPhase;
use crate::game_state::state::GameState;
use crate::gameplay::engine::GameEngine;
use crate::gameplay::rules::RuleEngine;
use crate::model::action::Action;

/// 基于有效盘面状态的纳秒级极速 FNV-1a 哈希 (彻底消除 969 维 f32 全字节 SipHash 开销)
#[inline]
pub fn fast_state_hash(state: &GameState) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325; // FNV-1a 64-bit offset basis
    const PRIME: u64 = 0x100000001b3;

    macro_rules! hash_u64 {
        ($val:expr) => {
            h ^= ($val) as u64;
            h = h.wrapping_mul(PRIME);
        };
    }

    hash_u64!(state.current_player);
    hash_u64!(state.privilege_pool);
    hash_u64!(state.extra_turn_granted as u8);
    hash_u64!(state.turn_number);

    // 棋盘 25 格
    for r in 0..5 {
        for c in 0..5 {
            let v = match state.board.grid[r][c] {
                Some(g) => g.index() as u64 + 1,
                None => 0,
            };
            hash_u64!(v);
        }
    }

    // 金字塔明牌 ID (最多 12 张)
    for row in &state.pyramid {
        for card in row {
            hash_u64!(card.id);
        }
    }

    // 王室卡 ID (最多 4 张)
    for royal in &state.royal_cards {
        hash_u64!(royal.id);
    }

    // 双方玩家状态 (标记、声望、王冠、特权、永久加成、手牌与预留卡)
    for p in &state.players {
        for &cnt in &p.tokens.counts {
            hash_u64!(cnt);
        }
        hash_u64!(p.total_points);
        hash_u64!(p.total_crowns);
        hash_u64!(p.privileges);
        for &b in &p.bonuses {
            hash_u64!(b);
        }
        for c in &p.cards {
            hash_u64!(c.id);
        }
        for rc in &p.reserved_cards {
            hash_u64!(((rc.card.id as u64) << 1) | (rc.is_public as u64));
        }
    }

    h
}

/// 跨步共享的轻量固定容量神经网络评估转置表 (零锁竞争、快速命中)
#[derive(Debug, Clone)]
pub struct NeuralEvalCache {
    map: HashMap<u64, ([f32; ACTION_SIZE], NeuralPrediction)>,
    max_entries: usize,
}

impl Default for NeuralEvalCache {
    fn default() -> Self {
        Self::new(4096)
    }
}

impl NeuralEvalCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            map: HashMap::with_capacity(max_entries),
            max_entries,
        }
    }

    #[inline]
    pub fn get(&self, key: &u64) -> Option<&([f32; ACTION_SIZE], NeuralPrediction)> {
        self.map.get(key)
    }

    #[inline]
    pub fn insert(&mut self, key: u64, val: ([f32; ACTION_SIZE], NeuralPrediction)) {
        if self.map.len() >= self.max_entries {
            self.map.clear();
        }
        self.map.insert(key, val);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// 子节点边
#[derive(Clone, Debug)]
pub struct Edge {
    pub action: Action,
    pub prior: f32,               // 动作先验概率 P(s, a)
    pub visits: u32,              // 访问次数 N
    pub w_p0: f32,                // Player-0 绝对累积价值 W
    pub child_idx: Option<usize>, // 若已展开则指向子节点索引
}

impl Edge {
    #[inline]
    pub fn q_p0(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.w_p0 / (self.visits as f32)
        }
    }
}

/// 树节点
#[derive(Clone, Debug)]
pub struct Node {
    pub player: usize,
    pub visits: u32,
    pub edges: Vec<Edge>,
    pub is_terminal: bool,
}

impl Node {
    #[inline]
    pub fn select_best_edge(&self, c_puct: f32) -> usize {
        debug_assert!(!self.edges.is_empty());
        let c_puct_sqrt = c_puct * (self.visits as f32).sqrt().max(1.0);
        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx = 0;

        for (i, edge) in self.edges.iter().enumerate() {
            // Q 从当前决策者 (node.player) 视角计算
            let q = edge.q_p0();
            let q_mover = if self.player == 0 { q } else { -q };

            // PUCT 公式: Q + c * P * (sqrt(N) / (1 + n))
            let uct = c_puct_sqrt * edge.prior / (1.0 + edge.visits as f32);
            let score = q_mover + uct;

            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        best_idx
    }
}

/// MCTS 搜索综合价值中的时间敏感度惩罚系数 (鼓励快速斩杀，惩罚拖延苟活)
pub const LAMBDA_TURNS: f32 = 0.20;

/// 多重确定化信息集洗牌块大小 (MIS-MCTS: 每隔 K 次模拟重抽暗牌，兼顾无偏估计与局部备份一致性)
pub const MIS_BLOCK_SIZE: usize = 8;

/// 根据当前盘面阶段和合法动作数，计算动态自适应 MCTS 模拟预算
/// 对 OptionalActions、DiscardTokens、SelectRoyalCard 等简单微步阶段自适应下调模拟次数，
/// 在保证核心决策质量的同时砍掉单局近半的冗余推演开销
#[inline]
pub fn compute_adaptive_sims(phase: &TurnPhase, num_legals: usize, base_sims: usize) -> usize {
    if num_legals <= 1 || base_sims == 0 {
        return num_sims_clamp(num_legals, base_sims);
    }
    match phase {
        TurnPhase::OptionalActions | TurnPhase::DiscardTokens | TurnPhase::SelectRoyalCard => {
            (base_sims / 4).max(10).min(base_sims)
        }
        _ => {
            if num_legals <= 3 {
                (base_sims / 3).max(12).min(base_sims)
            } else {
                base_sims
            }
        }
    }
}

#[inline]
fn num_sims_clamp(num_legals: usize, base_sims: usize) -> usize {
    if num_legals <= 1 {
        1.min(base_sims)
    } else {
        base_sims
    }
}

/// 自动折叠确定性单选项微步 (单一合法支付确认、单一合法弃牌)，压缩搜索树无谓深度
#[inline]
pub fn collapse_deterministic_micro_steps(sim_state: &mut GameState) {
    loop {
        match sim_state.phase {
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

/// 从神经网络预测的 logits 和候选动作集合生成 MCTS 初始边
#[inline]
pub fn create_edges_from_logits(legals: &[Action], logits: &[f32; ACTION_SIZE]) -> Vec<Edge> {
    if legals.is_empty() {
        return Vec::new();
    }
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

    legals
        .iter()
        .enumerate()
        .map(|(i, action)| Edge {
            action: action.clone(),
            prior: exp_scores[i] / sum_exp,
            visits: 0,
            w_p0: 0.0,
            child_idx: None,
        })
        .collect()
}

/// 基于纯神经网络推理与 AlphaZero 探索机制的高性能 Rust 原生 MCTS
pub struct RustMCTS {
    c_puct: f32,
}

impl Default for RustMCTS {
    fn default() -> Self {
        Self { c_puct: 1.5 }
    }
}

impl RustMCTS {
    pub fn new(c_puct: f32) -> Self {
        Self { c_puct }
    }

    /// 便捷方法：执行神经网络 MCTS 并直接返回选定的最佳动作（无需完整策略分布）
    pub fn search_action<R: Rng + ?Sized>(
        &self,
        state: &GameState,
        legals: Vec<Action>,
        evaluator: &TractNeuralEvaluator,
        eval_cache: &mut NeuralEvalCache,
        num_simulations: usize,
        rng: &mut R,
    ) -> Option<Action> {
        self.search_policy(
            state,
            legals,
            evaluator,
            eval_cache,
            num_simulations,
            false,
            0.0,
            0.0,
            0.0,
            rng,
        )
        .map(|(a, _)| a)
    }

    /// 执行带跨步共享评估转置表的高性能神经网络 MCTS 搜索并返回选择的动作以及完整的策略分布
    pub fn search_policy<R: Rng + ?Sized>(
        &self,
        state: &GameState,
        legals: Vec<Action>,
        evaluator: &TractNeuralEvaluator,
        eval_cache: &mut NeuralEvalCache,
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
        let root_idx = 0;
        let is_term = matches!(state.phase, TurnPhase::GameOver(_));

        let (mut root_edges, _root_pred) =
            Self::create_edges_with_neural_priors_cached(state, legals, evaluator, eval_cache)
                .ok()?;

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

        // 批量预热 (Batch Prefetch): 对高优先级 Top-K 子分支进行批量前向推演，一次性载入缓存
        if num_simulations >= 15 && nodes[root_idx].edges.len() > 1 {
            let top_k = nodes[root_idx].edges.len().min(4);
            let mut sorted_indices: Vec<usize> = (0..nodes[root_idx].edges.len()).collect();
            sorted_indices.sort_unstable_by(|&a, &b| {
                nodes[root_idx].edges[b]
                    .prior
                    .partial_cmp(&nodes[root_idx].edges[a].prior)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut batch_obs = Vec::with_capacity(top_k);
            let mut batch_keys = Vec::with_capacity(top_k);

            for &idx in sorted_indices.iter().take(top_k) {
                let mut sim_s = *state;
                if GameEngine::step(&mut sim_s, &nodes[root_idx].edges[idx].action).is_ok() {
                    collapse_deterministic_micro_steps(&mut sim_s);
                    if !matches!(sim_s.phase, TurnPhase::GameOver(_)) {
                        let key = fast_state_hash(&sim_s);
                        if eval_cache.get(&key).is_none() {
                            batch_obs.push(encode_state(&sim_s));
                            batch_keys.push(key);
                        }
                    }
                }
            }

            if !batch_obs.is_empty() {
                if let Ok(preds) = evaluator.evaluate_batch(&batch_obs) {
                    for (k, pred) in batch_keys.into_iter().zip(preds.into_iter()) {
                        eval_cache.insert(k, pred);
                    }
                }
            }
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
                        eval_cache,
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

            // 自适应早停判断：
            // 1. 确定性贪婪模式 (temperature <= 0.01):
            //    若根节点第一分支访问量 N1 与第二分支访问量 N2 的差值大于剩余推演次数，
            //    则即使剩余推演全部给 N2，N2 也绝对无法反超，纯数学无损提前截断
            // 2. 自对弈残余温度探索模式 (add_dirichlet == false && 已完成 >= 60% 模拟):
            //    仅在脱离初始探索噪声后介入，当第一名优势不可逆反超且占有统治级访问比时提前收敛，
            //    既完全保护前期的探索多样性，又大幅节省平稳局面的无效推演
            if nodes[root_idx].edges.len() >= 2 {
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
                if temperature <= 0.01 {
                    if max_visits.saturating_sub(second_max_visits) > remaining {
                        break;
                    }
                } else if !add_dirichlet && sim_idx >= (num_simulations * 3) / 5 {
                    if max_visits.saturating_sub(second_max_visits) > remaining {
                        break;
                    }
                }
            }
        }

        Self::extract_policy_distribution(&nodes[root_idx].edges, temperature, rng)
    }
    /// 从根节点分支访问量中提取完整的策略分布并进行采样
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

    /// 使用神经网络提供先验概率与状态估值 (带极速状态哈希与跨步缓存支持)
    fn create_edges_with_neural_priors_cached(
        state: &GameState,
        legals: Vec<Action>,
        evaluator: &TractNeuralEvaluator,
        cache: &mut NeuralEvalCache,
    ) -> Result<(Vec<Edge>, NeuralPrediction), String> {
        let key = fast_state_hash(state);

        let (logits, pred) = if let Some(cached) = cache.get(&key) {
            *cached
        } else {
            let obs = encode_state(state);
            let res = evaluator.evaluate(&obs)?;
            cache.insert(key, res);
            res
        };

        let edges = create_edges_from_logits(&legals, &logits);
        Ok((edges, pred))
    }

    #[inline]
    fn select_best_edge(&self, node: &Node) -> usize {
        node.select_best_edge(self.c_puct)
    }
}
