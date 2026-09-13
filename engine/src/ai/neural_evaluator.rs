use std::io::Cursor;
use std::sync::Arc;
use tract_onnx::prelude::*;

use crate::bridge::{ACTION_SIZE, OBS_SIZE};

pub type RunnableModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

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
    /// 返回 (ACTION_SIZE 维 policy_logits, value [-1.0, 1.0])
    pub fn evaluate(&self, obs: &[f32; OBS_SIZE]) -> Result<([f32; ACTION_SIZE], f32), String> {
        let input_tensor: Tensor = tract_ndarray::Array2::from_shape_vec(
            (1, OBS_SIZE),
            obs.to_vec(),
        )
        .map_err(|e| format!("Failed to create ndarray: {e}"))?
        .into();

        let outputs = self
            .plan
            .run(tvec!(input_tensor.into()))
            .map_err(|e| format!("Failed to run ONNX inference: {e}"))?;

        if outputs.len() < 2 {
            return Err(format!("Expected 2 outputs (logits, value), got {}", outputs.len()));
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

        // outputs[1]: value [1, 1]
        let value_view = outputs[1]
            .to_array_view::<f32>()
            .map_err(|e| format!("Failed to cast value output: {e}"))?;
        let value = if !value_view.is_empty() {
            value_view[[0, 0]]
        } else {
            0.0
        };

        Ok((logits, value))
    }
}
