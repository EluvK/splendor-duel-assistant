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
    pub reason_probs: [f32; 3], // 多标签独立胜因概率: [20_points, 10_crowns, 10_color]
}

impl NeuralPrediction {
    /// 计算用于 MCTS 驱动的综合搜索价值 (结合时间敏感度惩罚与逆风拖延激励，采用平滑连续激活消除非线性阶跃)
    #[inline]
    pub fn combined_value(&self, lambda_turns: f32) -> f32 {
        // 使用连续激活实现双向平滑门控：
        // 1. 均势局 (win_value -> 0) 时门控平滑趋向 0，完全消除 +/- 0.05 处的突变阶跃；
        // 2. 优势局 (win_value > 0) 时连续增强速胜奖励 [win_value, win_value + lambda * (1 - turns)]；
        // 3. 劣势局 (win_value < 0) 时连续增强逆风拖延激励 [win_value, win_value + lambda * turns]，
        //    始终保持正向补偿，确保下界不跌破 -1.0。
        let bias = if self.win_value >= 0.0 {
            let gate = (2.5 * self.win_value).tanh();
            gate * (1.0 - self.turns_value)
        } else {
            let gate = (-2.5 * self.win_value).tanh();
            gate * self.turns_value
        };
        self.win_value + lambda_turns * bias
    }
}

/// 基于 tract-onnx 的纯 Rust 神经网络评估器
#[derive(Clone)]
pub struct TractNeuralEvaluator {
    plan: Arc<RunnableModel>,
}

impl TractNeuralEvaluator {
    /// 从 ONNX 字节切片构建并优化模型 (支持任意批大小 [B, OBS_SIZE] 的动态推理)
    pub fn from_bytes(onnx_bytes: &[u8]) -> Result<Self, String> {
        let mut cursor = Cursor::new(onnx_bytes);
        let model = tract_onnx::onnx()
            .model_for_read(&mut cursor)
            .map_err(|e| format!("Failed to parse ONNX: {e}"))?;

        let batch = model.sym("batch");
        let plan = model
            .with_input_fact(0, f32::fact([batch.to_dim(), OBS_SIZE.to_dim()]).into())
            .map_err(|e| format!("Failed to set dynamic input fact: {e}"))?
            .into_optimized()
            .map_err(|e| format!("Failed to optimize model: {e}"))?
            .into_runnable()
            .map_err(|e| format!("Failed to build runnable plan: {e}"))?;

        Ok(Self {
            plan: Arc::new(plan),
        })
    }

    /// 评估单一步观察向量 (OBS_SIZE 维)
    /// 返回 (ACTION_SIZE 维 policy_logits, NeuralPrediction)
    pub fn evaluate(&self, obs: &[f32; OBS_SIZE]) -> Result<([f32; ACTION_SIZE], NeuralPrediction), String> {
        let input_tensor = unsafe {
            let mut tensor = Tensor::uninitialized::<f32>(&[1, OBS_SIZE])
                .map_err(|e| format!("Failed to allocate tensor: {e}"))?;
            std::ptr::copy_nonoverlapping(
                obs.as_ptr(),
                tensor.as_slice_mut_unchecked::<f32>().as_mut_ptr(),
                OBS_SIZE,
            );
            tensor
        };

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
        let logits_slice = outputs[0]
            .as_slice::<f32>()
            .map_err(|e| format!("Failed to access logits output slice: {e}"))?;
        if logits_slice.len() != ACTION_SIZE {
            return Err(format!("Expected {} logits, got {}", ACTION_SIZE, logits_slice.len()));
        }
        let mut logits = [0.0f32; ACTION_SIZE];
        logits.copy_from_slice(logits_slice);

        // outputs[1]: win_value [1, 1]
        let win_slice = outputs[1]
            .as_slice::<f32>()
            .map_err(|e| format!("Failed to access win_value slice: {e}"))?;
        let win_value = win_slice.first().copied().unwrap_or(0.0);

        // outputs[2]: turns_value [1, 1]
        let turns_slice = outputs[2]
            .as_slice::<f32>()
            .map_err(|e| format!("Failed to access turns_value slice: {e}"))?;
        let turns_value = turns_slice.first().copied().unwrap_or(0.5);

        // outputs[3]: reason_logits [1, 3]
        let reason_slice = outputs[3]
            .as_slice::<f32>()
            .map_err(|e| format!("Failed to access reason_logits slice: {e}"))?;
        let mut reason_probs = [0.0f32; 3];
        if reason_slice.len() >= 3 {
            for i in 0..3 {
                // Sigmoid 映射为独立多标签胜因概率
                reason_probs[i] = 1.0 / (1.0 + (-reason_slice[i]).exp());
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

    /// 批量并行评估多个观察向量 (形状 [N, OBS_SIZE])
    /// 相比逐样本单步调用，批处理大幅摊薄前向传播与状态初始化开销
    pub fn evaluate_batch(
        &self,
        obs_list: &[[f32; OBS_SIZE]],
    ) -> Result<Vec<([f32; ACTION_SIZE], NeuralPrediction)>, String> {
        let n = obs_list.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n == 1 {
            let res = self.evaluate(&obs_list[0])?;
            return Ok(vec![res]);
        }

        let input_tensor = unsafe {
            let mut tensor = Tensor::uninitialized::<f32>(&[n, OBS_SIZE])
                .map_err(|e| format!("Failed to allocate batch tensor: {e}"))?;
            std::ptr::copy_nonoverlapping(
                obs_list.as_ptr() as *const f32,
                tensor.as_slice_mut_unchecked::<f32>().as_mut_ptr(),
                n * OBS_SIZE,
            );
            tensor
        };

        let outputs = self
            .plan
            .run(tvec!(input_tensor.into()))
            .map_err(|e| format!("Failed to run ONNX batch inference: {e}"))?;

        if outputs.len() < 4 {
            return Err(format!(
                "Expected 4 outputs (policy_logits, win_value, turns_value, reason_logits), got {}",
                outputs.len()
            ));
        }

        // outputs[0]: policy_logits [n, ACTION_SIZE]
        let logits_slice = outputs[0]
            .as_slice::<f32>()
            .map_err(|e| format!("Failed to access batch logits: {e}"))?;

        // outputs[1]: win_value [n, 1]
        let win_slice = outputs[1]
            .as_slice::<f32>()
            .map_err(|e| format!("Failed to access batch win_value: {e}"))?;

        // outputs[2]: turns_value [n, 1]
        let turns_slice = outputs[2]
            .as_slice::<f32>()
            .map_err(|e| format!("Failed to access batch turns_value: {e}"))?;

        // outputs[3]: reason_logits [n, 3]
        let reason_slice = outputs[3]
            .as_slice::<f32>()
            .map_err(|e| format!("Failed to access batch reason_logits: {e}"))?;

        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            let mut logits = [0.0f32; ACTION_SIZE];
            logits.copy_from_slice(&logits_slice[i * ACTION_SIZE..(i + 1) * ACTION_SIZE]);

            let win_value = win_slice.get(i).copied().unwrap_or(0.0);
            let turns_value = turns_slice.get(i).copied().unwrap_or(0.5);

            let mut reason_probs = [0.0f32; 3];
            let r_start = i * 3;
            if reason_slice.len() >= r_start + 3 {
                for j in 0..3 {
                    reason_probs[j] = 1.0 / (1.0 + (-reason_slice[r_start + j]).exp());
                }
            }

            results.push((
                logits,
                NeuralPrediction {
                    win_value,
                    turns_value,
                    reason_probs,
                },
            ));
        }

        Ok(results)
    }
}
