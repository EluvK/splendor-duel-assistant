use std::sync::Arc;
use _engine::{InteractiveSession, PlayerKind, TractNeuralEvaluator};

#[test]
fn test_neural_mcts_step_with_evaluator() {
    // 读取 checkpoints/best.pt 导出的 onnx 如果存在
    let onnx_path = std::path::Path::new("../checkpoints/best.onnx");
    let local_onnx = if onnx_path.exists() {
        std::fs::read(onnx_path).ok()
    } else {
        std::fs::read("checkpoints/best.onnx").ok()
    };

    if let Some(bytes) = local_onnx {
        if let Ok(evaluator) = TractNeuralEvaluator::from_bytes(&bytes) {
            let eval_arc = Arc::new(evaluator);
            let mut sess = InteractiveSession::new(42, [PlayerKind::Neural, PlayerKind::Human]);
            sess.set_evaluator(Some(eval_arc.clone()));

            // 1. 指定 20 次模拟推演
            let step_res = sess.step_ai_with_sims(Some(20));
            assert!(step_res.is_ok());
            let step = step_res.unwrap().unwrap();
            let decision = step.decision.unwrap();
            assert!(
                decision.ai_type.contains("neural mcts (20 sims)"),
                "决策类型必须标明 20 sims: {}",
                decision.ai_type
            );

            // 2. 验证 0 次模拟（纯直觉神经网络前向推理）
            let mut sess_zero = InteractiveSession::new(43, [PlayerKind::Neural, PlayerKind::Human]);
            sess_zero.set_evaluator(Some(eval_arc.clone()));
            let zero_res = sess_zero.step_ai_with_sims(Some(0));
            assert!(zero_res.is_ok());
            let zero_step = zero_res.unwrap().unwrap();
            let zero_dec = zero_step.decision.unwrap();
            assert!(
                zero_dec.ai_type.contains("neural"),
                "0 次必须为纯直觉神经网络推演: {}",
                zero_dec.ai_type
            );

            // 3. 验证通过 set_mcts_simulations 设定的默认推演次数生效
            let mut sess_default = InteractiveSession::new(44, [PlayerKind::Neural, PlayerKind::Human]);
            sess_default.set_evaluator(Some(eval_arc.clone()));
            sess_default.set_mcts_simulations(75);
            let def_res = sess_default.step_ai_with_sims(None);
            assert!(def_res.is_ok());
            let def_step = def_res.unwrap().unwrap();
            let def_dec = def_step.decision.unwrap();
            assert!(
                def_dec.ai_type.contains("neural mcts (75 sims)"),
                "未指定推演参数时必须默认使用 session.mcts_simulations: {}",
                def_dec.ai_type
            );

            println!("✅ Neural MCTS Step 验证通过: {}", step.action_desc);
        }
    }
}
