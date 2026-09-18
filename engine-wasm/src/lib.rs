use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use splendor_duel_engine::ai::{
    format_action, DecisionDto, InteractiveSession, LegalActionDto, PlayerKind,
    PlayerType, ReplaySession, ReplayStep, ScoredActionDto, StateDto,
};
use splendor_duel_engine::bridge::{
    action_to_id, encode_state, ACTION_SIZE, OBS_SIZE,
};
use splendor_duel_engine::gameplay::{GameEngine, RuleEngine};
use splendor_duel_engine::model::Action;

/// 内部结构：单步摘要
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepSummary {
    pub index: usize,
    pub round: u32,
    pub player: usize,
    pub action: String,
    pub phase: String,
    pub score: Option<f32>,
    pub ai_type: Option<String>,
}

fn map_step_summaries(history: &[ReplayStep]) -> Vec<StepSummary> {
    history
        .iter()
        .map(|s| StepSummary {
            index: s.step_index,
            round: s.round_number,
            player: s.player,
            action: s.action_desc.clone(),
            phase: s.phase.clone(),
            score: s.decision.as_ref().and_then(|d| d.chosen_score),
            ai_type: s.decision.as_ref().map(|d| d.ai_type.clone()),
        })
        .collect()
}

/// 交互对战状态 DTO（与 /api/game/state 100% 契约一致）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameStateResponse {
    pub state: StateDto,
    pub player_kinds: [String; 2],
    pub current_player: usize,
    pub current_player_kind: String,
    pub is_human: bool,
    pub legal_actions: Vec<LegalActionDto>,
    pub neural_available: bool,
    pub mcts_available: bool,
    pub mcts_simulations: usize,
    pub history_len: usize,
    pub latest_step: Option<ReplayStep>,
    pub history: Vec<StepSummary>,
}

#[wasm_bindgen]
pub struct WasmGameSession {
    session: InteractiveSession,
    neural_ready: bool,
}

#[wasm_bindgen]
impl WasmGameSession {
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u64, p0_kind: &str, p1_kind: &str) -> Self {
        let p0 = PlayerKind::parse(p0_kind);
        let p1 = PlayerKind::parse(p1_kind);
        let session = InteractiveSession::new(seed, [p0, p1]);
        Self {
            session,
            neural_ready: false,
        }
    }

    /// 标记浏览器端 ONNX 神经网络推理器是否已就绪
    pub fn set_neural_ready(&mut self, ready: bool) {
        self.neural_ready = ready;
    }

    /// 设置 MCTS 默认推演次数
    pub fn set_mcts_simulations(&mut self, sims: usize) {
        self.session.set_mcts_simulations(sims);
    }

    /// 获取当前最新对局状态 JSON 字符串 (结构与 /api/game/state 严格一致)
    pub fn get_state_json(&self) -> String {
        let history = map_step_summaries(&self.session.history);
        let res = GameStateResponse {
            state: self.session.current_state(),
            player_kinds: [
                self.session.player_kinds[0].as_str().to_string(),
                self.session.player_kinds[1].as_str().to_string(),
            ],
            current_player: self.session.game.current_player,
            current_player_kind: self.session.current_player_kind().as_str().to_string(),
            is_human: self.session.is_current_player_human(),
            legal_actions: self.session.legal_actions_dto(),
            neural_available: self.neural_ready,
            mcts_available: false,
            mcts_simulations: self.session.mcts_simulations,
            history_len: self.session.history.len(),
            latest_step: self.session.history.last().cloned(),
            history,
        };
        serde_json::to_string(&res).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
    }

    /// 重置对局
    pub fn reset(&mut self, seed: u64, p0_kind: &str, p1_kind: &str) -> String {
        let p0 = PlayerKind::parse(p0_kind);
        let p1 = PlayerKind::parse(p1_kind);
        self.session.reset(seed, [p0, p1]);
        self.get_state_json()
    }

    /// 执行人类玩家动作（传入动作 JSON 或 payload 字符串）
    pub fn step_human(&mut self, action_json: &str) -> String {
        #[derive(Deserialize)]
        struct ActionWrapper {
            action: Action,
        }

        let action_res = if action_json.contains("\"action\"") {
            serde_json::from_str::<ActionWrapper>(action_json)
                .map(|w| w.action)
                .map_err(|e| e.to_string())
        } else {
            serde_json::from_str::<Action>(action_json).map_err(|e| e.to_string())
        };

        match action_res {
            Ok(action) => match self.session.step_human(action) {
                Ok(step) => serde_json::json!({
                    "ok": true,
                    "step": step,
                    "state": self.session.current_state(),
                    "player_kinds": [
                        self.session.player_kinds[0].as_str(),
                        self.session.player_kinds[1].as_str(),
                    ],
                    "current_player": self.session.game.current_player,
                    "is_human": self.session.is_current_player_human(),
                    "legal_actions": self.session.legal_actions_dto(),
                })
                .to_string(),
                Err(err) => serde_json::json!({ "ok": false, "error": err }).to_string(),
            },
            Err(e) => serde_json::json!({
                "ok": false,
                "error": format!("Action JSON parse error: {e}")
            })
            .to_string(),
        }
    }

    /// 执行当前非神经网络 AI（如 Heuristic 或 Random）
    pub fn step_ai(&mut self, mcts_sims: Option<usize>) -> String {
        match self.session.step_ai_with_sims(mcts_sims) {
            Ok(Some(step)) => serde_json::json!({
                "ok": true,
                "step": step,
                "state": self.session.current_state(),
                "player_kinds": [
                    self.session.player_kinds[0].as_str(),
                    self.session.player_kinds[1].as_str(),
                ],
                "current_player": self.session.game.current_player,
                "is_human": self.session.is_current_player_human(),
                "legal_actions": self.session.legal_actions_dto(),
            })
            .to_string(),
            Ok(None) => serde_json::json!({
                "ok": false,
                "error": "Current player is human or game over"
            })
            .to_string(),
            Err(err) => serde_json::json!({ "ok": false, "error": err }).to_string(),
        }
    }

    /// 获取当前盘面的 969 维浮点观测特征向量（供浏览器端 ONNX 模型推理）
    pub fn encode_observation(&self) -> Vec<f32> {
        let obs = encode_state(&self.session.game);
        obs.to_vec()
    }

    /// 获取当前盘面的全部合法动作 ID 列表 (0..1856)
    pub fn get_legal_action_ids(&self) -> Vec<u32> {
        let legals = RuleEngine::legal_actions(&self.session.game);
        legals
            .into_iter()
            .map(|a| action_to_id(&a) as u32)
            .collect()
    }

    /// 获取当前动作空间大小 (固定 1856) 与观测维度 (固定 969)
    pub fn get_spec(&self) -> Vec<u32> {
        vec![OBS_SIZE as u32, ACTION_SIZE as u32]
    }

    /// 接收浏览器端 ONNX 神经网络推理得到的最佳动作与决策详情并执行落子
    pub fn apply_neural_step(
        &mut self,
        best_action_id: usize,
        winrate: f32,
        top_candidates_json: &str,
    ) -> String {
        if matches!(
            self.session.game.phase,
            splendor_duel_engine::game_state::TurnPhase::GameOver(_)
        ) {
            return serde_json::json!({ "ok": false, "error": "Game is already over" }).to_string();
        }

        let legals = RuleEngine::legal_actions(&self.session.game);
        let chosen_action = legals
            .into_iter()
            .find(|a| action_to_id(a) == best_action_id);

        let action = match chosen_action {
            Some(a) => a,
            None => {
                return serde_json::json!({
                    "ok": false,
                    "error": format!("Action ID {} is not legal in current state", best_action_id)
                })
                .to_string();
            }
        };

        // 反序列化候选动作详情
        let top_candidates: Vec<ScoredActionDto> =
            serde_json::from_str(top_candidates_json).unwrap_or_default();

        let decision = DecisionDto {
            ai_type: "neural (onnx-web)".to_string(),
            chosen_score: Some(winrate * 100.0),
            top_candidates,
        };

        let round_number = self.session.game.turn_number;
        let player = self.session.game.current_player;
        let action_desc = format_action(&action);
        let phase_desc = format!("{:?}", self.session.game.phase);

        if let Err(e) = GameEngine::step(&mut self.session.game, &action) {
            return serde_json::json!({ "ok": false, "error": format!("Engine step error: {e}") })
                .to_string();
        }

        let next_step = ReplayStep {
            step_index: self.session.history.len(),
            round_number,
            player,
            action_desc,
            phase: phase_desc,
            state: StateDto::from(&self.session.game),
            decision: Some(decision),
        };

        self.session.history.push(next_step.clone());
        self.session.maybe_auto_skip_optional();

        serde_json::json!({
            "ok": true,
            "step": next_step,
            "state": self.session.current_state(),
            "player_kinds": [
                self.session.player_kinds[0].as_str(),
                self.session.player_kinds[1].as_str(),
            ],
            "current_player": self.session.game.current_player,
            "is_human": self.session.is_current_player_human(),
            "legal_actions": self.session.legal_actions_dto(),
        })
        .to_string()
    }

    /// 获取完整历史步骤 JSON
    pub fn get_history_json(&self) -> String {
        let steps = map_step_summaries(&self.session.history);
        serde_json::json!({
            "total_steps": steps.len(),
            "steps": steps,
        })
        .to_string()
    }

    /// 获取指定索引的单步详情 JSON
    pub fn get_step_json(&self, index: usize) -> String {
        match self.session.history.get(index) {
            Some(step) => serde_json::to_string(step).unwrap_or_default(),
            None => serde_json::json!({ "ok": false, "error": "Step not found" }).to_string(),
        }
    }

    /// 导出当前局势的快照与完整历史，用于复盘会话初始化
    pub fn export_replay_data(&self) -> String {
        let player_types = [
            self.session.player_kinds[0].to_replay_player_type().as_str(),
            self.session.player_kinds[1].to_replay_player_type().as_str(),
        ];
        serde_json::json!({
            "seed": self.session.seed,
            "player_types": player_types,
            "live_game": self.session.game,
            "history": self.session.history,
        })
        .to_string()
    }
}

/// 复盘回放 WASM 实例 (对应 replay.html)
#[wasm_bindgen]
pub struct WasmReplaySession {
    session: ReplaySession,
    neural_ready: bool,
}

#[wasm_bindgen]
impl WasmReplaySession {
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u64, p0_type: &str, p1_type: &str) -> Self {
        let p0 = PlayerType::parse(p0_type);
        let p1 = PlayerType::parse(p1_type);
        let session = ReplaySession::new_with_players(seed, [p0, p1]);
        Self {
            session,
            neural_ready: false,
        }
    }

    /// 标记浏览器端 ONNX 神经网络推理器是否已就绪
    pub fn set_neural_ready(&mut self, ready: bool) {
        self.neural_ready = ready;
    }

    /// 从导入的数据 (JSON 字符串，由 export_replay_data 产生) 创建复盘会话
    pub fn from_replay_data(data_json: &str) -> Result<WasmReplaySession, String> {
        #[derive(Deserialize)]
        struct ImportedData {
            seed: u64,
            player_types: [String; 2],
            #[serde(default)]
            live_game: Option<splendor_duel_engine::game_state::GameState>,
            history: Vec<ReplayStep>,
        }

        let imported: ImportedData = serde_json::from_str(data_json)
            .map_err(|e| format!("Failed to parse replay data: {e}"))?;

        let p0 = PlayerType::parse(&imported.player_types[0]);
        let p1 = PlayerType::parse(&imported.player_types[1]);

        let live_game = imported.live_game.unwrap_or_else(|| {
            splendor_duel_engine::game_state::GameState::new_game(imported.seed)
        });

        let sess = ReplaySession {
            seed: imported.seed,
            player_types: [p0, p1],
            live_game,
            rng: rand_chacha::ChaCha8Rng::seed_from_u64(imported.seed),
            history: imported.history,
        };

        Ok(Self {
            session: sess,
            neural_ready: false,
        })
    }

    /// 获取当前行动方玩家类型 ("neural", "heuristic", "random")
    pub fn current_player_type(&self) -> String {
        let p = self.session.live_game.current_player;
        self.session.player_types[p].as_str().to_string()
    }

    /// 获取当前盘面的 969 维特征向量 (供神经网络推理)
    pub fn encode_observation(&self) -> Vec<f32> {
        let obs = encode_state(&self.session.live_game);
        obs.to_vec()
    }

    /// 获取当前盘面的全部合法动作 ID 列表
    pub fn get_legal_action_ids(&self) -> Vec<u32> {
        let legals = RuleEngine::legal_actions(&self.session.live_game);
        legals.into_iter().map(|a| action_to_id(&a) as u32).collect()
    }

    /// 获取合法动作展示 DTO 列表
    pub fn get_legal_actions_dto(&self) -> String {
        let legals = RuleEngine::legal_actions(&self.session.live_game);
        let dtos: Vec<LegalActionDto> = legals
            .into_iter()
            .map(|a| LegalActionDto {
                desc: format_action(&a),
                category: splendor_duel_engine::ai::action_category(&a).to_string(),
                action: a,
            })
            .collect();
        serde_json::to_string(&dtos).unwrap_or_else(|_| "[]".to_string())
    }

    /// 接收神经网络推理结果执行单步推进 (用于复盘模式下的神经网络走步)
    pub fn step_with_neural(
        &mut self,
        best_action_id: usize,
        winrate: f32,
        top_candidates_json: &str,
    ) -> String {
        if matches!(
            self.session.live_game.phase,
            splendor_duel_engine::game_state::TurnPhase::GameOver(_)
        ) {
            return serde_json::json!({
                "advanced": false,
                "advanced_count": 0,
                "total_steps": self.session.history.len(),
                "total_rounds": self.session.history.last().map(|s| s.round_number).unwrap_or(1),
                "step": self.session.history.last(),
                "new_steps": [],
            })
            .to_string();
        }

        let legals = RuleEngine::legal_actions(&self.session.live_game);
        let chosen_action = legals
            .into_iter()
            .find(|a| action_to_id(a) == best_action_id);

        let action = match chosen_action {
            Some(a) => a,
            None => {
                return serde_json::json!({ "error": format!("Action ID {best_action_id} not legal") }).to_string();
            }
        };

        let top_candidates: Vec<ScoredActionDto> =
            serde_json::from_str(top_candidates_json).unwrap_or_default();

        let decision = DecisionDto {
            ai_type: "neural (onnx-web)".to_string(),
            chosen_score: Some(winrate * 100.0),
            top_candidates,
        };

        let player = self.session.live_game.current_player;
        let round_number = self.session.live_game.turn_number;
        let action_desc = format_action(&action);
        let phase_desc = format!("{:?}", self.session.live_game.phase);

        if let Err(e) = GameEngine::step(&mut self.session.live_game, &action) {
            return serde_json::json!({ "error": format!("Engine step error: {e}") }).to_string();
        }

        let next_step = ReplayStep {
            step_index: self.session.history.len(),
            round_number,
            player,
            action_desc: action_desc.clone(),
            phase: phase_desc.clone(),
            state: StateDto::from(&self.session.live_game),
            decision: Some(decision),
        };

        let step_summary = StepSummary {
            index: next_step.step_index,
            round: next_step.round_number,
            player: next_step.player,
            action: next_step.action_desc.clone(),
            phase: next_step.phase.clone(),
            score: next_step.decision.as_ref().and_then(|d| d.chosen_score),
            ai_type: next_step.decision.as_ref().map(|d| d.ai_type.clone()),
        };

        self.session.history.push(next_step.clone());

        serde_json::json!({
            "advanced": true,
            "advanced_count": 1,
            "total_steps": self.session.history.len(),
            "total_rounds": self.session.history.last().map(|s| s.round_number).unwrap_or(1),
            "step": next_step,
            "new_steps": [step_summary],
        })
        .to_string()
    }

    /// 获取当前最新复盘状态 JSON (结构与 /api/status 100% 对齐)
    pub fn get_status_json(&self) -> String {
        let last_step = self.session.history.last().cloned();
        serde_json::json!({
            "state": self.session.current_state(),
            "player_types": [
                self.session.player_types[0].as_str(),
                self.session.player_types[1].as_str(),
            ],
            "step": last_step,
            "neural_available": self.neural_ready,
        })
        .to_string()
    }

    /// 推进单步或 N 步 (结构与 POST /api/step 100% 对齐)
    pub fn step_forward(&mut self, count: usize) -> String {
        let prev_len = self.session.history.len();
        match self.session.step_n(count) {
            Ok(advanced_count) => {
                let last_step = self.session.history.last().cloned();
                let new_steps: Vec<StepSummary> = self.session.history[prev_len..]
                    .iter()
                    .map(|s| StepSummary {
                        index: s.step_index,
                        round: s.round_number,
                        player: s.player,
                        action: s.action_desc.clone(),
                        phase: s.phase.clone(),
                        score: s.decision.as_ref().and_then(|d| d.chosen_score),
                        ai_type: s.decision.as_ref().map(|d| d.ai_type.clone()),
                    })
                    .collect();

                serde_json::json!({
                    "advanced": advanced_count > 0,
                    "advanced_count": advanced_count,
                    "total_steps": self.session.history.len(),
                    "total_rounds": self.session.history.last().map(|s| s.round_number).unwrap_or(1),
                    "step": last_step,
                    "new_steps": new_steps,
                })
                .to_string()
            }
            Err(e) => serde_json::json!({ "error": e }).to_string(),
        }
    }

    /// 推进直到终局 (结构与 POST /api/play_to_end 100% 对齐)
    pub fn play_to_end(&mut self, max_steps: usize) -> String {
        let _ = self.session.play_to_end(max_steps);
        let last_step = self.session.history.last().cloned();
        serde_json::json!({
            "total_steps": self.session.history.len(),
            "total_rounds": self.session.history.last().map(|s| s.round_number).unwrap_or(1),
            "step": last_step,
        })
        .to_string()
    }

    /// 使用指定种子与配置重置对局 (结构与 POST /api/reset 100% 对齐)
    pub fn reset_with_players(&mut self, seed: u64, p0_type: &str, p1_type: &str) -> String {
        let p0 = PlayerType::parse(p0_type);
        let p1 = PlayerType::parse(p1_type);
        self.session.reset_with_players(seed, [p0, p1]);
        self.get_status_json()
    }

    /// 动态修改双方身份配置 (结构与 POST /api/set_players 100% 对齐)
    pub fn set_players(&mut self, p0_type: &str, p1_type: &str) -> String {
        let p0 = PlayerType::parse(p0_type);
        let p1 = PlayerType::parse(p1_type);
        self.session.set_player_type(0, p0);
        self.session.set_player_type(1, p1);
        serde_json::json!({
            "player_types": [p0.as_str(), p1.as_str()]
        })
        .to_string()
    }

    pub fn get_history_json(&self) -> String {
        let steps: Vec<_> = self
            .session
            .history
            .iter()
            .map(|s| StepSummary {
                index: s.step_index,
                round: s.round_number,
                player: s.player,
                action: s.action_desc.clone(),
                phase: s.phase.clone(),
                score: s.decision.as_ref().and_then(|d| d.chosen_score),
                ai_type: s.decision.as_ref().map(|d| d.ai_type.clone()),
            })
            .collect();
        serde_json::json!({
            "seed": self.session.seed,
            "total_steps": steps.len(),
            "total_rounds": self.session.history.last().map(|s| s.round_number).unwrap_or(1),
            "player_types": [
                self.session.player_types[0].as_str(),
                self.session.player_types[1].as_str(),
            ],
            "steps": steps,
        })
        .to_string()
    }

    pub fn get_step_json(&self, index: usize) -> String {
        match self.session.get_step(index) {
            Some(step) => serde_json::to_string(step).unwrap_or_default(),
            None => serde_json::json!({ "ok": false, "error": "Step not found" }).to_string(),
        }
    }
}
