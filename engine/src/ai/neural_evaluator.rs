use std::io::Cursor;
use std::sync::Arc;
use tract_onnx::prelude::*;

use crate::bridge::{ACTION_SIZE, OBS_SIZE};

pub type RunnableModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// 神经网络多任务预测结果 (解耦胜负、剩余步数与终局胜因)
#[derive(Debug, Clone, Copy)]
pub struct NeuralPrediction {
    pub win_value: f32,        // 纯胜率预期 [-1.0, 1.0]
    pub turns_value: f32,      // 归一化剩余轮数预期 [0.0, 1.0] (0..80 轮)
    pub reason_probs: [f32; 4], // 胜因概率分布: [20_points, 10_crowns, 10_color, draw]
}

impl NeuralPrediction {
    /// 计算用于 MCTS 驱动的综合搜索价值 (结合时间敏感度惩罚)
    #[inline]
    pub fn combined_value(&self, lambda_turns: f32) -> f32 {
        self.win_value - lambda_turns * self.turns_value
    }
}

/// 基于 tract-onnx 的纯 Rust 神经网络评估器
#[derive(Clone)]
pub struct TractNeuralEvaluator {
    plan: Arc<RunnableModel>,
}

impl TractNeuralEvaluator {
    /// 从 ONNX 字节切片构建并优化模型
    pub fn from_bytes(onnx_bytes: &[u8]) -> Result<Self, String> {
        let mut cursor = Cursor::new(onnx_bytes);
        let model = tract_onnx::onnx()
            .model_for_read(&mut cursor)
            .map_err(|e| format!("Failed to parse ONNX: {e}"))?
            .with_input_fact(0, f32::fact([1, OBS_SIZE]).into())
            .map_err(|e| format!("Failed to set input fact: {e}"))?
            .into_optimized()
            .map_err(|e| format!("Failed to optimize model: {e}"))?
            .into_runnable()
            .map_err(|e| format!("Failed to build runnable plan: {e}"))?;

        Ok(Self {
            plan: Arc::new(model),
        })
    }

    /// 评估单一步观察向量 (OBS_SIZE 维)
    /// 返回 (ACTION_SIZE 维 policy_logits, NeuralPrediction)
    pub fn evaluate(&self, obs: &[f32; OBS_SIZE]) -> Result<([f32; ACTION_SIZE], NeuralPrediction), String> {
        let input_tensor: Tensor = tract_ndarray::ArrayView2::from_shape((1, OBS_SIZE), obs.as_slice())
            .map_err(|e| format!("Failed to create ndarray view: {e}"))?
            .to_owned()
            .into();

        let outputs = self
            .plan
            .run(tvec!(input_tensor.into()))
            .map_err(|e| format!("Failed to run ONNX inference: {e}"))?;

        if outputs.len() < 4 {
            return Err(format!(
                "Expected 4 outputs (policy_logits, win_value, turns_value, reason_logits), got {}",
                outputs.len()
            ));
        }

        // outputs[0]: policy_logits [1, ACTION_SIZE]
        let logits_view = outputs[0]
            .to_array_view::<f32>()
            .map_err(|e| format!("Failed to cast logits output: {e}"))?;
        let mut logits = [0.0f32; ACTION_SIZE];
        if logits_view.len() != ACTION_SIZE {
            return Err(format!("Expected {} logits, got {}", ACTION_SIZE, logits_view.len()));
        }
        for (i, &v) in logits_view.iter().enumerate() {
            logits[i] = v;
        }

        // outputs[1]: win_value [1, 1]
        let win_view = outputs[1]
            .to_array_view::<f32>()
            .map_err(|e| format!("Failed to cast win_value output: {e}"))?;
        let win_value = if !win_view.is_empty() {
            win_view[[0, 0]]
        } else {
            0.0
        };

        // outputs[2]: turns_value [1, 1]
        let turns_view = outputs[2]
            .to_array_view::<f32>()
            .map_err(|e| format!("Failed to cast turns_value output: {e}"))?;
        let turns_value = if !turns_view.is_empty() {
            turns_view[[0, 0]]
        } else {
            0.5
        };

        // outputs[3]: reason_logits [1, 4]
        let reason_view = outputs[3]
            .to_array_view::<f32>()
            .map_err(|e| format!("Failed to cast reason_logits output: {e}"))?;
        let mut reason_probs = [0.25f32; 4];
        if reason_view.len() >= 4 {
            let max_logit = reason_view.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for (i, &l) in reason_view.iter().take(4).enumerate() {
                let exp_val = (l - max_logit).exp();
                reason_probs[i] = exp_val;
                sum_exp += exp_val;
            }
            if sum_exp > 1e-6 {
                for p in reason_probs.iter_mut() {
                    *p /= sum_exp;
                }
            }
        }

        Ok((
            logits,
            NeuralPrediction {
                win_value,
                turns_value,
                reason_probs,
            },
        ))
    }
}
