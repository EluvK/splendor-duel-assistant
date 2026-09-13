use rand::prelude::*;
use rand_distr::multi::{Dirichlet, MultiDistribution};

use crate::ai::heuristic_ai::HeuristicAI;
use crate::game_state::phase::TurnPhase;
use crate::game_state::state::GameState;
use crate::gameplay::engine::GameEngine;
use crate::gameplay::rules::RuleEngine;
use crate::model::action::Action;

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
        let legals = RuleEngine::legal_actions(state);
        if legals.is_empty() {
            return None;
        }
        // 零开销快捷路径：单一动作免搜索直接返回
        if legals.len() == 1 {
            return Some(legals[0].clone());
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

        for _ in 0..num_simulations {
            let mut sim_state = state.clone();
            let mut curr_node_idx = root_idx;
            // 记录沿途 (node_idx, edge_idx)
            let mut path: Vec<(usize, usize)> = Vec::with_capacity(16);

            // 1. Selection: 沿着树向下选择 PUCT 最大分支
            while !nodes[curr_node_idx].is_terminal && !nodes[curr_node_idx].edges.is_empty() {
                let best_edge_idx = self.select_best_edge(&nodes[curr_node_idx]);
                path.push((curr_node_idx, best_edge_idx));

                let action = nodes[curr_node_idx].edges[best_edge_idx].action.clone();
                let _ = GameEngine::step(&mut sim_state, &action);

                // 若该边已有子节点，继续向下探索；否则在当前叶子停止展开
                if let Some(child_idx) = nodes[curr_node_idx].edges[best_edge_idx].child_idx {
                    curr_node_idx = child_idx;
                } else {
                    break;
                }
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
        }

        // 根据温度参数进行动作选取
        if temperature <= 0.01 {
            // 贪婪选择根节点下访问次数最多 (最稳健) 的动作
            let best_edge = nodes[root_idx]
                .edges
                .iter()
                .max_by_key(|e| e.visits);
            best_edge.map(|e| e.action.clone())
        } else {
            // 温度轮盘赌采样 (前 10~15 步破除死板套路)
            let root_edges = &nodes[root_idx].edges;
            let inv_temp = 1.0 / temperature;
            let mut exp_visits = Vec::with_capacity(root_edges.len());
            let mut sum_v = 0.0f32;
            for edge in root_edges.iter() {
                let v = (edge.visits as f32).powf(inv_temp);
                exp_visits.push(v);
                sum_v += v;
            }
            if sum_v <= 1e-6 {
                return nodes[root_idx].edges.first().map(|e| e.action.clone());
            }
            let mut pick = rng.random_range(0.0..sum_v);
            for (i, &v) in exp_visits.iter().enumerate() {
                if pick <= v {
                    return Some(root_edges[i].action.clone());
                }
                pick -= v;
            }
            root_edges.last().map(|e| e.action.clone())
        }
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
