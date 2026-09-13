use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

use crate::bridge::{action_mask, action_to_id, encode_state};
use crate::game_state::state::GameState;
use crate::gameplay::rules::RuleEngine;
use crate::model::action::Action;

#[derive(Serialize)]
struct PredictRequest<'a> {
    obs: &'a [f32],
    mask: &'a [bool],
    temperature: f32,
}

#[derive(Deserialize)]
struct TopAction {
    action_id: usize,
    prob: f32,
    #[allow(dead_code)]
    logit: f32,
}

#[derive(Deserialize)]
struct PredictResponse {
    best_action_id: usize,
    value: f32,
    top_actions: Vec<TopAction>,
    epoch: Option<u32>,
}

pub struct NeuralPrediction {
    pub best_action: Action,
    pub winrate: f32, // [-1.0, 1.0]
    pub top_candidates: Vec<(Action, f32, bool)>, // (Action, prob 0..1, is_chosen)
    pub epoch: u32,
}

pub struct NeuralAI;

impl NeuralAI {
    const DEFAULT_ADDR: &'static str = "127.0.0.1:8088";

    /// 探测神经网络推理微服务是否就绪
    pub fn is_available() -> bool {
        let addr: SocketAddr = Self::DEFAULT_ADDR.parse().unwrap();
        if let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_millis(200)) {
            let req = "GET /health HTTP/1.1\r\nHost: 127.0.0.1:8088\r\nConnection: close\r\n\r\n";
            if stream.write_all(req.as_bytes()).is_ok() {
                let mut buf = [0u8; 512];
                if let Ok(n) = stream.read(&mut buf) {
                    let resp = String::from_utf8_lossy(&buf[..n]);
                    return resp.contains("200 OK");
                }
            }
        }
        false
    }

    /// 向推理服务请求预测动作与胜率评估
    pub fn predict_action(
        state: &GameState,
        temperature: f32,
    ) -> Result<NeuralPrediction, String> {
        let legals = RuleEngine::legal_actions(state);
        if legals.is_empty() {
            return Err("No legal actions available".to_string());
        }

        let obs = encode_state(state);
        let mask = action_mask(state);

        let req_payload = PredictRequest {
            obs: &obs[..],
            mask: &mask[..],
            temperature,
        };

        let req_json = serde_json::to_vec(&req_payload)
            .map_err(|e| format!("Failed to serialize predict request: {e}"))?;

        let addr: SocketAddr = Self::DEFAULT_ADDR
            .parse()
            .map_err(|e| format!("Invalid address: {e}"))?;

        let mut stream = TcpStream::connect_timeout(&addr, Duration::from_millis(300))
            .map_err(|e| format!("Inference server connection failed: {e}"))?;

        let _ = stream.set_read_timeout(Some(Duration::from_millis(600)));
        let _ = stream.set_write_timeout(Some(Duration::from_millis(300)));

        let header = format!(
            "POST /predict HTTP/1.1\r\nHost: 127.0.0.1:8088\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            req_json.len()
        );

        stream
            .write_all(header.as_bytes())
            .map_err(|e| format!("Failed to send HTTP header: {e}"))?;
        stream
            .write_all(&req_json)
            .map_err(|e| format!("Failed to send request body: {e}"))?;

        let mut resp_bytes = Vec::with_capacity(4096);
        stream
            .read_to_end(&mut resp_bytes)
            .map_err(|e| format!("Failed to read HTTP response: {e}"))?;

        let resp_str = String::from_utf8_lossy(&resp_bytes);
        let body = if let Some(idx) = resp_str.find("\r\n\r\n") {
            &resp_str[idx + 4..]
        } else {
            return Err("Invalid HTTP response: header separator not found".to_string());
        };

        let pred_res: PredictResponse = serde_json::from_str(body)
            .map_err(|e| format!("Failed to parse JSON response: {e}. Raw body: {body}"))?;

        // 将离散动作 ID 映射回合法高层 Action
        let mut legal_id_map: Vec<(usize, Action)> = Vec::with_capacity(legals.len());
        for act in legals {
            let id = action_to_id(&act);
            legal_id_map.push((id, act));
        }

        // 优先匹配 best_action_id
        let chosen_action = legal_id_map
            .iter()
            .find(|(id, _)| *id == pred_res.best_action_id)
            .map(|(_, a)| a.clone())
            .or_else(|| {
                // 若未直接命中，取合法动作中在 top_actions 排名最靠前者
                for top in &pred_res.top_actions {
                    if let Some((_, a)) = legal_id_map.iter().find(|(id, _)| *id == top.action_id) {
                        return Some(a.clone());
                    }
                }
                // 兜底返回第一个合法动作
                legal_id_map.first().map(|(_, a)| a.clone())
            })
            .ok_or_else(|| "No matched legal action found".to_string())?;

        let chosen_id = action_to_id(&chosen_action);

        let mut top_candidates = Vec::with_capacity(pred_res.top_actions.len());
        for top in pred_res.top_actions {
            if let Some((_, act)) = legal_id_map.iter().find(|(id, _)| *id == top.action_id) {
                top_candidates.push((act.clone(), top.prob, top.action_id == chosen_id));
            }
        }

        Ok(NeuralPrediction {
            best_action: chosen_action,
            winrate: pred_res.value,
            top_candidates,
            epoch: pred_res.epoch.unwrap_or(1),
        })
    }
}
