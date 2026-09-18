use serde_json::Value;
use splendor_duel_wasm::{WasmGameSession, WasmReplaySession};

#[test]
fn test_wasm_game_session_lifecycle() {
    let session = WasmGameSession::new(42, "human", "neural");

    // 1. 获取盘面状态 JSON 并校验结构契约
    let state_raw = session.get_state_json();
    let state_json: Value = serde_json::from_str(&state_raw).expect("state_raw 必须是合法 JSON");

    assert_eq!(state_json["current_player"], 0);
    assert_eq!(state_json["is_human"], true);
    assert_eq!(state_json["player_kinds"][0], "human");
    assert_eq!(state_json["player_kinds"][1], "neural");

    let legals = state_json["legal_actions"].as_array().expect("legal_actions 必须为数组");
    assert!(!legals.is_empty(), "开局必须有合法动作");

    // 2. 特征编码与规格元数据验证
    let obs = session.encode_observation();
    assert_eq!(obs.len(), 969, "观测特征向量维度必须严格为 969");

    let spec = session.get_spec();
    assert_eq!(spec, vec![969, 1856], "规范元数据维度应为 [969, 1856]");

    let action_ids = session.get_legal_action_ids();
    assert_eq!(action_ids.len(), legals.len());
    for id in action_ids {
        assert!(id < 1856, "合法动作 ID 必须在 0..1856 范围内");
    }
}

#[test]
fn test_wasm_game_session_human_and_ai_step() {
    let mut session = WasmGameSession::new(42, "human", "heuristic");

    let state_raw = session.get_state_json();
    let state_json: Value = serde_json::from_str(&state_raw).unwrap();
    let first_legal = &state_json["legal_actions"][0]["action"];

    // 1. 人类走合法第一步
    let step_res_raw = session.step_human(&first_legal.to_string());
    let step_res: Value = serde_json::from_str(&step_res_raw).unwrap();
    assert_eq!(step_res["ok"], true, "人类执行合法动作必须成功: {:?}", step_res);

    // 2. 尝试执行明显非法的动作字符串
    let invalid_res_raw = session.step_human("{\"TakeTokens\":{\"count\":99,\"positions\":[[0,0],[0,0],[0,0]]}}");
    let invalid_res: Value = serde_json::from_str(&invalid_res_raw).unwrap();
    assert_eq!(invalid_res["ok"], false, "非法动作必须被拒绝");

    // 3. 启发式 AI 执行走步
    let ai_step_raw = session.step_ai(None);
    let ai_step: Value = serde_json::from_str(&ai_step_raw).unwrap();
    assert_eq!(ai_step["ok"], true, "AI 走步必须成功: {:?}", ai_step);
}

#[test]
fn test_wasm_replay_export_import_roundtrip() {
    let mut game_session = WasmGameSession::new(12345, "heuristic", "heuristic");

    // 推进 4 步走局
    for _ in 0..4 {
        let res_raw = game_session.step_ai(None);
        let res: Value = serde_json::from_str(&res_raw).unwrap();
        assert_eq!(res["ok"], true);
    }

    let game_state_before_export = game_session.get_state_json();
    let parsed_before: Value = serde_json::from_str(&game_state_before_export).unwrap();
    let total_steps_before = parsed_before["history_len"].as_u64().unwrap() as usize;
    assert!(total_steps_before >= 4);

    // 导出复盘数据
    let export_data = game_session.export_replay_data();

    // 从导出的数据构建复盘会话
    let mut replay_session = WasmReplaySession::from_replay_data(&export_data)
        .expect("从导出的数据创建复盘会话必须成功");

    let replay_status_raw = replay_session.get_status_json();
    let replay_status: Value = serde_json::from_str(&replay_status_raw).unwrap();

    // 验证状态一致性：棋盘、行动方必须与对战导出时完全一致，而不是被初始化回开局状态
    assert_eq!(
        replay_status["state"]["current_player"],
        parsed_before["current_player"],
        "复盘会话的行动方必须继承对局当时行动方"
    );
    assert_eq!(
        replay_status["state"]["turn_number"],
        parsed_before["state"]["turn_number"],
        "回合数必须一致"
    );

    // 验证复盘会话基于当前恢复的局面能继续正常向前推演
    let step_forward_res_raw = replay_session.step_forward(1);
    let step_forward_res: Value = serde_json::from_str(&step_forward_res_raw).unwrap();
    assert_eq!(step_forward_res["advanced"], true, "恢复局势后应能继续向前推演");
}

#[test]
fn test_wasm_replay_step_with_neural_and_winrate() {
    let mut replay = WasmReplaySession::new(42, "neural", "neural");

    // 验证初始与设定的 neural_available 状态一致性
    let init_status_raw = replay.get_status_json();
    let init_status: Value = serde_json::from_str(&init_status_raw).unwrap();
    assert_eq!(init_status["neural_available"], false);

    replay.set_neural_ready(true);
    let ready_status_raw = replay.get_status_json();
    let ready_status: Value = serde_json::from_str(&ready_status_raw).unwrap();
    assert_eq!(ready_status["neural_available"], true);

    // 模拟前端 ONNX 模型推理给出的最佳动作与胜率
    let legal_ids = replay.get_legal_action_ids();
    assert!(!legal_ids.is_empty());
    let best_id = legal_ids[0] as usize;
    let winrate = 0.652; // 65.2% 胜率

    let candidates_json = "[{\"action_desc\":\"Action #0\",\"score\":65.2,\"is_chosen\":true}]";
    let res_raw = replay.step_with_neural(best_id, winrate, candidates_json);
    let res: Value = serde_json::from_str(&res_raw).expect("step_with_neural 必须返回合法 JSON");

    assert_eq!(res["advanced"], true);
    assert_eq!(res["step"]["decision"]["ai_type"], "neural (onnx-web)");

    // 校验历史列表中输出的胜率过程数据
    let history_raw = replay.get_history_json();
    let history: Value = serde_json::from_str(&history_raw).unwrap();
    let last_summary = history["steps"].as_array().unwrap().last().unwrap();

    assert_eq!(last_summary["ai_type"], "neural (onnx-web)");
    let score_f = last_summary["score"].as_f64().expect("胜率过程数据 score 必须存在");
    assert!((score_f - 65.2).abs() < 1e-3, "胜率分数必须为真实的 65.2% 百分比，当前为: {}", score_f);
}

#[test]
fn test_wasm_game_session_save_and_restore() {
    let mut session = WasmGameSession::new(777, "human", "heuristic");

    // 人类执行第一步
    let state_raw = session.get_state_json();
    let state_json: Value = serde_json::from_str(&state_raw).unwrap();
    let first_legal = &state_json["legal_actions"][0]["action"];
    let step_res_raw = session.step_human(&first_legal.to_string());
    let step_res: Value = serde_json::from_str(&step_res_raw).unwrap();
    assert_eq!(step_res["ok"], true);

    // AI 执行第二步
    let ai_step_raw = session.step_ai(None);
    let ai_step: Value = serde_json::from_str(&ai_step_raw).unwrap();
    assert_eq!(ai_step["ok"], true);

    // 导出持久化状态
    let state_before_raw = session.get_state_json();
    let state_before: Value = serde_json::from_str(&state_before_raw).unwrap();

    let saved_state = session.export_saved_state();
    assert!(!saved_state.is_empty());

    // 模拟新实例通过 saved_state 恢复
    let restored_session = WasmGameSession::from_saved_state(&saved_state)
        .expect("从保存状态恢复对局必须成功");

    let restored_state_raw = restored_session.get_state_json();
    let restored_state: Value = serde_json::from_str(&restored_state_raw).unwrap();

    assert_eq!(restored_state["history_len"], state_before["history_len"]);
    assert_eq!(restored_state["current_player"], state_before["current_player"]);
    assert_eq!(restored_state["state"]["turn_number"], state_before["state"]["turn_number"]);
    assert_eq!(restored_state["state"]["board"], state_before["state"]["board"], "恢复前后棋盘盘面必须完全无损吻合");
    assert_eq!(restored_state["state"]["players"], state_before["state"]["players"], "恢复前后双方玩家手牌筹码必须完全一致");
    assert_eq!(restored_state["legal_actions"], state_before["legal_actions"], "恢复后当前合法动作列表必须完全一致");
}

