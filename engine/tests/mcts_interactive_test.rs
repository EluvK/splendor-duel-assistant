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
            let mut sess = InteractiveSession::new(42, [PlayerKind::Neural, PlayerKind::Human]);
            sess.set_evaluator(Some(Arc::new(evaluator)));
            let step_res = sess.step_ai_with_sims(Some(20));
            assert!(step_res.is_ok());
            let step = step_res.unwrap().unwrap();
            let decision = step.decision.unwrap();
            assert!(
                decision.ai_type.contains("neural mcts"),
                "决策类型必须标明 neural mcts: {}",
                decision.ai_type
            );
            println!("✅ Neural MCTS Step 验证通过: {}", step.action_desc);
        }
    }
}
