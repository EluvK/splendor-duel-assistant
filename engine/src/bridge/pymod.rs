use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::SeedableRng;

use super::encode::{action_mask, action_to_id, encode_state, ACTION_SIZE, OBS_SIZE};
use crate::game_state::phase::TurnPhase;
use crate::game_state::state::GameState;
use crate::gameplay::engine::GameEngine;
use crate::gameplay::rules::RuleEngine;

/// 暴露给 Python 强化学习环境调用的 GameState 封装
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PyGameState {
    state: GameState,
}

#[pymethods]
impl PyGameState {
    #[new]
    #[pyo3(signature = (seed=42))]
    pub fn new(seed: u64) -> Self {
        Self {
            state: GameState::new_game(seed),
        }
    }

    /// 使用新的随机种子重置对局
    pub fn reset(&mut self, seed: u64) {
        self.state = GameState::new_game(seed);
    }

    /// 深拷贝当前状态（用于 MCTS 前向分支模拟）
    pub fn clone_state(&self) -> Self {
        self.clone()
    }

    /// 获取当前行动方规范化视角的 OBS_SIZE (742) 维状态观察向量
    pub fn observe(&self) -> Vec<f32> {
        encode_state(&self.state).to_vec()
    }

    /// 获取当前状态下的合法动作掩码 (ACTION_SIZE (288) 维 bool 数组)
    pub fn action_mask(&self) -> Vec<bool> {
        action_mask(&self.state).to_vec()
    }

    /// 获取当前所有合法动作的整数 ID 列表
    pub fn legal_action_ids(&self) -> Vec<usize> {
        RuleEngine::legal_actions(&self.state)
            .iter()
            .map(action_to_id)
            .collect()
    }

    /// 执行一个动作 ID (0..ACTION_SIZE-1)
    /// 返回元组: (next_obs, done, winner)
    pub fn step(&mut self, action_id: usize) -> PyResult<(Vec<f32>, bool, Option<usize>)> {
        if action_id >= ACTION_SIZE {
            return Err(PyValueError::new_err(format!(
                "Action ID {action_id} out of range [0, {ACTION_SIZE})"
            )));
        }

        let legals = RuleEngine::legal_actions(&self.state);
        let action = legals
            .into_iter()
            .find(|act| action_to_id(act) == action_id)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Action ID {action_id} is not legal in current phase {:?}",
                    self.state.phase
                ))
            })?;

        GameEngine::step(&mut self.state, &action)
            .map_err(|e| PyValueError::new_err(format!("Action execution error: {e}")))?;

        let done = matches!(self.state.phase, TurnPhase::GameOver(_));
        let winner = self.state.winner.map(|(w, _)| w);
        let obs = self.observe();

        Ok((obs, done, winner))
    }

    /// 当前游戏是否已终局
    pub fn is_done(&self) -> bool {
        matches!(self.state.phase, TurnPhase::GameOver(_))
    }

    /// 获胜玩家 (0 或 1)，未结束则返回 None
    pub fn winner(&self) -> Option<usize> {
        self.state.winner.map(|(w, _)| w)
    }

    /// 当前轮到的行动方 (0 或 1)
    pub fn current_player(&self) -> usize {
        self.state.current_player
    }

    /// 当前回合数
    pub fn turn_number(&self) -> u32 {
        self.state.turn_number
    }

    /// 当前轮数 (Round)
    pub fn round_number(&self) -> u32 {
        self.state.round_number()
    }

    /// 当前阶段名称字符串
    pub fn phase(&self) -> String {
        format!("{:?}", self.state.phase)
    }

    /// 双方当前声望总分 (p0_points, p1_points)
    pub fn scores(&self) -> (u8, u8) {
        (
            self.state.players[0].total_points,
            self.state.players[1].total_points,
        )
    }

    /// 双方当前王冠总数 (p0_crowns, p1_crowns)
    pub fn crowns(&self) -> (u8, u8) {
        (
            self.state.players[0].total_crowns,
            self.state.players[1].total_crowns,
        )
    }

    /// 获取启发式 AI 在当前盘面下选择的动作 ID
    #[pyo3(signature = (seed=42))]
    pub fn heuristic_action_id(&self, py: Python<'_>, seed: u64) -> Option<usize> {
        py.detach(|| {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
            crate::ai::HeuristicAI::select_action(&self.state, &mut rng).map(|a| action_to_id(&a))
        })
    }

    /// 调用底层 Rust 原生高性能 MCTS 进行推演并返回最佳动作 ID (微秒级响应，释放 GIL 允许 Python 全核并发)
    #[pyo3(signature = (num_sims=50, seed=None, add_dirichlet=false, dirichlet_alpha=0.3, dirichlet_eps=0.25, temperature=0.0, max_rollout_steps=15))]
    pub fn mcts_action_id(
        &self,
        py: Python<'_>,
        num_sims: usize,
        seed: Option<u64>,
        add_dirichlet: bool,
        dirichlet_alpha: f32,
        dirichlet_eps: f32,
        temperature: f32,
        max_rollout_steps: usize,
    ) -> Option<usize> {
        py.detach(|| {
            let mut rng = match seed {
                Some(s) => rand_chacha::ChaCha8Rng::seed_from_u64(s),
                None => rand_chacha::ChaCha8Rng::from_rng(&mut rand::rng()),
            };
            let mcts = crate::ai::RustMCTS::new(1.5, max_rollout_steps);
            mcts.search_with_exploration(
                &self.state,
                num_sims,
                add_dirichlet,
                dirichlet_alpha,
                dirichlet_eps,
                temperature,
                &mut rng,
            )
            .map(|a| action_to_id(&a))
        })
    }

    #[staticmethod]
    pub fn observation_space_size() -> usize {
        OBS_SIZE
    }

    #[staticmethod]
    pub fn action_space_size() -> usize {
        ACTION_SIZE
    }
}

/// 批量多线程并行生成启发式专家轨迹样本 (释放 GIL，全核并发)
#[pyfunction]
#[pyo3(signature = (num_games=1000, start_seed=42))]
pub fn generate_heuristic_samples(
    py: Python<'_>,
    num_games: usize,
    start_seed: u64,
) -> PyResult<(
    pyo3::Py<numpy::PyArray1<f32>>,
    pyo3::Py<numpy::PyArray1<u8>>,
    pyo3::Py<numpy::PyArray1<i32>>,
    pyo3::Py<numpy::PyArray1<f32>>,
    pyo3::Py<numpy::PyArray1<i32>>,
    usize,
)> {
    let batch = py.detach(|| crate::ai::sample_heuristic_games_parallel(num_games, start_seed));

    let (obs_arr, mask_arr, action_arr, value_arr, reason_arr) = Python::attach(|py| {
        (
            numpy::PyArray1::from_vec(py, batch.obs).unbind(),
            numpy::PyArray1::from_vec(py, batch.masks).unbind(),
            numpy::PyArray1::from_vec(py, batch.actions).unbind(),
            numpy::PyArray1::from_vec(py, batch.values).unbind(),
            numpy::PyArray1::from_vec(py, batch.reasons).unbind(),
        )
    });

    Ok((obs_arr, mask_arr, action_arr, value_arr, reason_arr, batch.total_steps))
}

/// 批量多线程并行生成带 MCTS 深度推演与 AlphaZero 探索机制的自博弈样本 (8 线程全速并发)
#[pyfunction]
#[pyo3(signature = (num_games=100, num_sims=30, start_seed=42, temp_steps=12, dirichlet_alpha=0.3, dirichlet_eps=0.25))]
pub fn generate_mcts_samples(
    py: Python<'_>,
    num_games: usize,
    num_sims: usize,
    start_seed: u64,
    temp_steps: usize,
    dirichlet_alpha: f32,
    dirichlet_eps: f32,
) -> PyResult<(
    pyo3::Py<numpy::PyArray1<f32>>,
    pyo3::Py<numpy::PyArray1<u8>>,
    pyo3::Py<numpy::PyArray1<i32>>,
    pyo3::Py<numpy::PyArray1<f32>>,
    pyo3::Py<numpy::PyArray1<i32>>,
    usize,
)> {
    let batch = py.detach(|| {
        crate::ai::sample_mcts_games_parallel_with_config(
            num_games,
            num_sims,
            start_seed,
            temp_steps,
            dirichlet_alpha,
            dirichlet_eps,
        )
    });

    let (obs_arr, mask_arr, action_arr, value_arr, reason_arr) = Python::attach(|py| {
        (
            numpy::PyArray1::from_vec(py, batch.obs).unbind(),
            numpy::PyArray1::from_vec(py, batch.masks).unbind(),
            numpy::PyArray1::from_vec(py, batch.actions).unbind(),
            numpy::PyArray1::from_vec(py, batch.values).unbind(),
            numpy::PyArray1::from_vec(py, batch.reasons).unbind(),
        )
    });

    Ok((obs_arr, mask_arr, action_arr, value_arr, reason_arr, batch.total_steps))
}

#[pyfunction]
#[pyo3(signature = (model_bytes, num_games=100, num_sims=30, start_seed=42, temp_steps=12, dirichlet_alpha=0.3, dirichlet_eps=0.25))]
pub fn generate_neural_mcts_samples(
    py: Python<'_>,
    model_bytes: &[u8],
    num_games: usize,
    num_sims: usize,
    start_seed: u64,
    temp_steps: usize,
    dirichlet_alpha: f32,
    dirichlet_eps: f32,
) -> PyResult<(
    pyo3::Py<numpy::PyArray1<f32>>,
    pyo3::Py<numpy::PyArray1<u8>>,
    pyo3::Py<numpy::PyArray1<i32>>,
    pyo3::Py<numpy::PyArray1<f32>>,
    pyo3::Py<numpy::PyArray1<i32>>,
    usize,
)> {
    let batch = py
        .detach(|| {
            crate::ai::sample_neural_mcts_games_parallel(
                model_bytes,
                num_games,
                num_sims,
                start_seed,
                temp_steps,
                dirichlet_alpha,
                dirichlet_eps,
            )
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let (obs_arr, mask_arr, action_arr, value_arr, reason_arr) = Python::attach(|py| {
        (
            numpy::PyArray1::from_vec(py, batch.obs).unbind(),
            numpy::PyArray1::from_vec(py, batch.masks).unbind(),
            numpy::PyArray1::from_vec(py, batch.actions).unbind(),
            numpy::PyArray1::from_vec(py, batch.values).unbind(),
            numpy::PyArray1::from_vec(py, batch.reasons).unbind(),
        )
    });

    Ok((obs_arr, mask_arr, action_arr, value_arr, reason_arr, batch.total_steps))
}

/// 纯 Rust 多线程 8 核并发成对严格换座对抗评测 (秒级极速完成门禁对抗，零 Python/CUDA 开销)
/// - model_bytes_1: None 或空字节时，对抗内置 HeuristicAI
/// - num_sims: 0 为极速纯直觉 PolicyNet 对决；> 0 时开启纯神经网络 MCTS 树搜索对抗
#[pyfunction]
#[pyo3(signature = (model_bytes_0, model_bytes_1=None, num_pairs=10, base_seed=1000, num_sims=0))]
pub fn evaluate_neural_match(
    py: Python<'_>,
    model_bytes_0: &[u8],
    model_bytes_1: Option<&[u8]>,
    num_pairs: usize,
    base_seed: u64,
    num_sims: usize,
) -> PyResult<(usize, usize, usize, usize, HashMap<String, usize>)> {
    let res = py
        .detach(|| {
            crate::ai::evaluate_neural_match_parallel(
                model_bytes_0,
                model_bytes_1,
                num_pairs,
                base_seed,
                num_sims,
            )
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let mut reasons = HashMap::new();
    reasons.insert("20_points".to_string(), res.reasons_20_pts);
    reasons.insert("10_crowns".to_string(), res.reasons_10_crowns);
    reasons.insert("10_color_points".to_string(), res.reasons_10_color);
    reasons.insert("draw".to_string(), res.draws);
    reasons.insert("p0_seat_wins".to_string(), res.p0_seat_wins);
    reasons.insert("p1_seat_wins".to_string(), res.p1_seat_wins);
    reasons.insert("total_steps".to_string(), res.total_steps);
    reasons.insert("total_rounds".to_string(), res.total_rounds);
    reasons.insert("agent0_win_steps".to_string(), res.agent0_win_steps);
    reasons.insert("agent0_win_rounds".to_string(), res.agent0_win_rounds);
    reasons.insert("agent0_lose_steps".to_string(), res.agent0_lose_steps);
    reasons.insert("agent0_lose_rounds".to_string(), res.agent0_lose_rounds);

    Ok((
        res.total_games,
        res.agent0_wins,
        res.agent1_wins,
        res.draws,
        reasons,
    ))
}
