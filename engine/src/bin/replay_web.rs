use _engine::{
    Action, InteractiveSession, LegalActionDto, NeuralAI, PlayerKind, PlayerType, ReplaySession,
    ReplayStep, StateDto,
};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Child;
use std::sync::{Arc, Mutex, RwLock};

static NEURAL_CHILD: Mutex<Option<Child>> = Mutex::new(None);

#[derive(Serialize, Clone)]
struct StepSummary {
    index: usize,
    round: u32,
    player: usize,
    action: String,
    phase: String,
    score: Option<f32>,
    ai_type: Option<String>,
}

#[derive(Serialize)]
struct StatusResponse {
    state: _engine::StateDto,
    player_types: [String; 2],
    step: _engine::ReplayStep,
    neural_available: bool,
}

#[derive(Serialize)]
struct HistoryResponse {
    seed: u64,
    total_steps: usize,
    total_rounds: u32,
    player_types: [String; 2],
    steps: Vec<StepSummary>,
}

#[derive(Serialize)]
struct GameStateResponse {
    state: StateDto,
    player_kinds: [String; 2],
    current_player: usize,
    current_player_kind: String,
    is_human: bool,
    legal_actions: Vec<LegalActionDto>,
    neural_available: bool,
    history_len: usize,
    latest_step: Option<ReplayStep>,
    history: Vec<StepSummary>,
}

fn map_step_summaries(steps: &[ReplayStep]) -> Vec<StepSummary> {
    steps
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

struct AppState {
    replay: RwLock<ReplaySession>,
    game: RwLock<InteractiveSession>,
}

fn respond(stream: &mut TcpStream, status: &str, body: &[u8], content_type: &str) {
    let _ = write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(body);
}

fn respond_headers_only(stream: &mut TcpStream, status: &str, length: usize, content_type: &str) {
    let _ = write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {length}\r\nAccess-Control-Allow-Origin: *\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
    );
}

fn find_web_root() -> PathBuf {
    let candidates = [
        PathBuf::from("engine/web"),
        PathBuf::from("web"),
        PathBuf::from("../engine/web"),
    ];
    for c in candidates {
        if c.exists() && (c.join("index.html").exists() || c.join("play.html").exists()) {
            return c;
        }
    }
    PathBuf::from("engine/web")
}

fn extract_body(request_str: &str) -> &str {
    if let Some(idx) = request_str.find("\r\n\r\n") {
        &request_str[idx + 4..]
    } else if let Some(idx) = request_str.find("\n\n") {
        &request_str[idx + 2..]
    } else {
        ""
    }
}

fn handle_client(mut stream: TcpStream, state: &Arc<AppState>, web_root: &Path) {
    let mut buffer = [0u8; 8192];
    let bytes_read = match stream.read(&mut buffer) {
        Ok(n) if n > 0 => n,
        _ => return,
    };

    let request_str = String::from_utf8_lossy(&buffer[..bytes_read]);
    let first_line = request_str.lines().next().unwrap_or("");
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    if parts.len() < 2 {
        respond(&mut stream, "400 Bad Request", b"bad request", "text/plain");
        return;
    }

    let method = parts[0];
    let raw_url = parts[1];
    let (path, query) = if let Some(idx) = raw_url.find('?') {
        (&raw_url[..idx], &raw_url[idx + 1..])
    } else {
        (raw_url, "")
    };

    let body_str = extract_body(&request_str);

    match (method, path) {
        // ================= 交互对战 API =================
        ("GET", "/api/game/state") => {
            let game = state.game.read().unwrap();
            let history = map_step_summaries(&game.history);
            let res = GameStateResponse {
                state: game.current_state(),
                player_kinds: [
                    game.player_kinds[0].as_str().to_string(),
                    game.player_kinds[1].as_str().to_string(),
                ],
                current_player: game.game.current_player,
                current_player_kind: game.current_player_kind().as_str().to_string(),
                is_human: game.is_current_player_human(),
                legal_actions: game.legal_actions_dto(),
                neural_available: NeuralAI::is_available(),
                history_len: game.history.len(),
                latest_step: game.history.last().cloned(),
                history,
            };
            let json = serde_json::to_vec(&res).unwrap();
            respond(&mut stream, "200 OK", &json, "application/json");
        }

        // 获取对战完整步骤历史列表
        ("GET", "/api/game/history") => {
            let game = state.game.read().unwrap();
            let steps = map_step_summaries(&game.history);
            let res = serde_json::json!({
                "total_steps": steps.len(),
                "steps": steps,
            });
            let json = serde_json::to_vec(&res).unwrap();
            respond(&mut stream, "200 OK", &json, "application/json");
        }

        // 获取对战指定单步的完整信息（含 AI 决策候选详情）
        ("GET", "/api/game/step") => {
            let mut step_index: Option<usize> = None;
            for param in query.split('&') {
                let mut kv = param.split('=');
                if let (Some(k), Some(v)) = (kv.next(), kv.next()) {
                    if k == "index" || k == "step" {
                        step_index = v.parse::<usize>().ok();
                    }
                }
            }
            let game = state.game.read().unwrap();
            let step = match step_index {
                Some(idx) => game.history.get(idx).cloned(),
                None => game.history.last().cloned(),
            };
            if let Some(s) = step {
                let json = serde_json::to_vec(&s).unwrap();
                respond(&mut stream, "200 OK", &json, "application/json");
            } else {
                let err = serde_json::json!({ "ok": false, "error": "Step not found" });
                let json = serde_json::to_vec(&err).unwrap();
                respond(&mut stream, "404 Not Found", &json, "application/json");
            }
        }

        // 人类玩家提交执行动作
        ("POST", "/api/game/action") => {
            #[derive(Deserialize)]
            struct ActionRequest {
                action: Action,
            }

            let action_res: Result<Action, String> = if body_str.contains("\"action\"") {
                serde_json::from_str::<ActionRequest>(body_str)
                    .map(|r| r.action)
                    .map_err(|e| format!("ActionRequest parse failed: {e}"))
            } else {
                serde_json::from_str::<Action>(body_str)
                    .map_err(|e| format!("Action parse failed: {e}"))
            };

            match action_res {
                Ok(action) => {
                    let mut game = state.game.write().unwrap();
                    match game.step_human(action) {
                        Ok(step) => {
                            let res = serde_json::json!({
                                "ok": true,
                                "step": step,
                                "state": game.current_state(),
                                "current_player": game.game.current_player,
                                "is_human": game.is_current_player_human(),
                                "legal_actions": game.legal_actions_dto(),
                            });
                            let json = serde_json::to_vec(&res).unwrap();
                            respond(&mut stream, "200 OK", &json, "application/json");
                        }
                        Err(err) => {
                            let res = serde_json::json!({ "ok": false, "error": err });
                            let json = serde_json::to_vec(&res).unwrap();
                            respond(&mut stream, "400 Bad Request", &json, "application/json");
                        }
                    }
                }
                Err(e) => {
                    let res = serde_json::json!({ "ok": false, "error": format!("Invalid action JSON: {e}") });
                    let json = serde_json::to_vec(&res).unwrap();
                    respond(&mut stream, "400 Bad Request", &json, "application/json");
                }
            }
        }

        // 触发 AI 执行一步
        ("POST", "/api/game/ai_step") | ("GET", "/api/game/ai_step") => {
            let mut game = state.game.write().unwrap();
            match game.step_ai() {
                Ok(Some(step)) => {
                    let res = serde_json::json!({
                        "ok": true,
                        "step": step,
                        "state": game.current_state(),
                        "current_player": game.game.current_player,
                        "is_human": game.is_current_player_human(),
                        "legal_actions": game.legal_actions_dto(),
                    });
                    let json = serde_json::to_vec(&res).unwrap();
                    respond(&mut stream, "200 OK", &json, "application/json");
                }
                Ok(None) => {
                    let res = serde_json::json!({
                        "ok": false,
                        "error": "Current player is human or game over"
                    });
                    let json = serde_json::to_vec(&res).unwrap();
                    respond(&mut stream, "200 OK", &json, "application/json");
                }
                Err(err) => {
                    let res = serde_json::json!({ "ok": false, "error": err });
                    let json = serde_json::to_vec(&res).unwrap();
                    respond(&mut stream, "400 Bad Request", &json, "application/json");
                }
            }
        }

        // 重置交互对战对局
        ("POST", "/api/game/reset") | ("GET", "/api/game/reset") | ("POST", "/api/game/new") => {
            let mut seed: u64 = 42;
            let mut p0 = PlayerKind::Human;
            let mut p1 = PlayerKind::Neural;

            for param in query.split('&') {
                let mut kv = param.split('=');
                if let (Some(k), Some(v)) = (kv.next(), kv.next()) {
                    if k == "seed" {
                        seed = v.parse::<u64>().unwrap_or(42);
                    } else if k == "p0" {
                        p0 = PlayerKind::parse(v);
                    } else if k == "p1" {
                        p1 = PlayerKind::parse(v);
                    }
                }
            }

            let mut game = state.game.write().unwrap();
            game.reset(seed, [p0, p1]);
            let history = map_step_summaries(&game.history);

            let res = GameStateResponse {
                state: game.current_state(),
                player_kinds: [
                    game.player_kinds[0].as_str().to_string(),
                    game.player_kinds[1].as_str().to_string(),
                ],
                current_player: game.game.current_player,
                current_player_kind: game.current_player_kind().as_str().to_string(),
                is_human: game.is_current_player_human(),
                legal_actions: game.legal_actions_dto(),
                neural_available: NeuralAI::is_available(),
                history_len: game.history.len(),
                latest_step: game.history.last().cloned(),
                history,
            };
            let json = serde_json::to_vec(&res).unwrap();
            respond(&mut stream, "200 OK", &json, "application/json");
        }

        // 将当前对局一键导出到复盘系统
        ("POST", "/api/game/to_replay") | ("GET", "/api/game/to_replay") => {
            let game = state.game.read().unwrap();
            let replay_sess = game.to_replay_session();
            let mut replay = state.replay.write().unwrap();
            *replay = replay_sess;

            let res = serde_json::json!({
                "ok": true,
                "total_steps": replay.history.len(),
                "redirect": "replay.html"
            });
            let json = serde_json::to_vec(&res).unwrap();
            respond(&mut stream, "200 OK", &json, "application/json");
        }

        // ================= 现有回放 API =================
        ("GET", "/api/status") => {
            let sess = state.replay.read().unwrap();
            let last_step = sess.history.last().unwrap().clone();
            let res = StatusResponse {
                state: sess.current_state(),
                player_types: [
                    sess.player_types[0].as_str().to_string(),
                    sess.player_types[1].as_str().to_string(),
                ],
                step: last_step,
                neural_available: NeuralAI::is_available(),
            };
            let json = serde_json::to_vec(&res).unwrap();
            respond(&mut stream, "200 OK", &json, "application/json");
        }

        ("GET", "/api/neural_status") => {
            let available = NeuralAI::is_available();
            let details = if available { NeuralAI::get_status() } else { None };
            let res = serde_json::json!({
                "available": available,
                "details": details,
            });
            let json = serde_json::to_vec(&res).unwrap();
            respond(&mut stream, "200 OK", &json, "application/json");
        }

        ("POST", "/api/neural_reload") | ("GET", "/api/neural_reload") => {
            match NeuralAI::reload_checkpoint() {
                Ok(val) => {
                    let json = serde_json::to_vec(&val).unwrap();
                    respond(&mut stream, "200 OK", &json, "application/json");
                }
                Err(e) => {
                    let err = serde_json::json!({ "ok": false, "error": e });
                    let json = serde_json::to_vec(&err).unwrap();
                    respond(&mut stream, "500 Internal Error", &json, "application/json");
                }
            }
        }

        ("POST", "/api/step") | ("GET", "/api/step_next") => {
            let mut count: usize = 1;
            for param in query.split('&') {
                let mut kv = param.split('=');
                if let (Some(k), Some(v)) = (kv.next(), kv.next()) {
                    if k == "count" {
                        count = v.parse::<usize>().unwrap_or(1);
                    }
                }
            }

            let mut sess = state.replay.write().unwrap();
            let prev_len = sess.history.len();
            match sess.step_n(count) {
                Ok(advanced_count) => {
                    let last_step = sess.history.last().unwrap().clone();
                    let new_steps: Vec<StepSummary> = sess.history[prev_len..]
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

                    #[derive(Serialize)]
                    struct StepResult {
                        advanced: bool,
                        advanced_count: usize,
                        total_steps: usize,
                        total_rounds: u32,
                        step: _engine::ReplayStep,
                        new_steps: Vec<StepSummary>,
                    }
                    let res = StepResult {
                        advanced: advanced_count > 0,
                        advanced_count,
                        total_steps: sess.history.len(),
                        total_rounds: sess.history.last().map(|s| s.round_number).unwrap_or(1),
                        step: last_step,
                        new_steps,
                    };
                    let json = serde_json::to_vec(&res).unwrap();
                    respond(&mut stream, "200 OK", &json, "application/json");
                }
                Err(e) => {
                    let err_json = serde_json::to_vec(&serde_json::json!({ "error": e })).unwrap();
                    respond(&mut stream, "400 Bad Request", &err_json, "application/json");
                }
            }
        }

        ("POST", "/api/play_to_end") | ("GET", "/api/play_to_end") => {
            let mut sess = state.replay.write().unwrap();
            let _ = sess.play_to_end(2000);
            let last_step = sess.history.last().unwrap().clone();
            #[derive(Serialize)]
            struct EndResult {
                total_steps: usize,
                total_rounds: u32,
                step: _engine::ReplayStep,
            }
            let res = EndResult {
                total_steps: sess.history.len(),
                total_rounds: sess.history.last().map(|s| s.round_number).unwrap_or(1),
                step: last_step,
            };
            let json = serde_json::to_vec(&res).unwrap();
            respond(&mut stream, "200 OK", &json, "application/json");
        }

        ("POST", "/api/set_players") | ("GET", "/api/set_players") => {
            let mut sess = state.replay.write().unwrap();
            for param in query.split('&') {
                let mut kv = param.split('=');
                if let (Some(k), Some(v)) = (kv.next(), kv.next()) {
                    if k == "p0" {
                        sess.set_player_type(0, PlayerType::parse(v));
                    } else if k == "p1" {
                        sess.set_player_type(1, PlayerType::parse(v));
                    }
                }
            }
            let res = serde_json::json!({
                "player_types": [
                    sess.player_types[0].as_str(),
                    sess.player_types[1].as_str(),
                ]
            });
            let json = serde_json::to_vec(&res).unwrap();
            respond(&mut stream, "200 OK", &json, "application/json");
        }

        ("POST", "/api/reset") | ("GET", "/api/reset") => {
            let mut seed: u64 = 42;
            let mut play_to_end = false;
            let mut p0 = PlayerType::Heuristic;
            let mut p1 = PlayerType::Heuristic;

            for param in query.split('&') {
                let mut kv = param.split('=');
                if let (Some(k), Some(v)) = (kv.next(), kv.next()) {
                    if k == "seed" {
                        seed = v.parse::<u64>().unwrap_or(42);
                    } else if k == "play_to_end" {
                        play_to_end = v == "true" || v == "1";
                    } else if k == "p0" {
                        p0 = PlayerType::parse(v);
                    } else if k == "p1" {
                        p1 = PlayerType::parse(v);
                    }
                }
            }

            let mut sess = state.replay.write().unwrap();
            sess.reset_with_players(seed, [p0, p1]);
            if play_to_end {
                let _ = sess.play_to_end(2000);
            }
            let last_step = sess.history.last().unwrap().clone();
            let res = StatusResponse {
                state: sess.current_state(),
                player_types: [
                    sess.player_types[0].as_str().to_string(),
                    sess.player_types[1].as_str().to_string(),
                ],
                step: last_step,
                neural_available: NeuralAI::is_available(),
            };
            let json = serde_json::to_vec(&res).unwrap();
            respond(&mut stream, "200 OK", &json, "application/json");
        }

        ("GET", "/api/step") => {
            let mut index = 0;
            for param in query.split('&') {
                let mut kv = param.split('=');
                if let (Some(k), Some(v)) = (kv.next(), kv.next()) {
                    if k == "index" {
                        index = v.parse::<usize>().unwrap_or(0);
                    }
                }
            }

            let sess = state.replay.read().unwrap();
            if let Some(step) = sess.get_step(index) {
                let json = serde_json::to_vec(step).unwrap();
                respond(&mut stream, "200 OK", &json, "application/json");
            } else {
                respond(&mut stream, "404 Not Found", b"Step not found", "text/plain");
            }
        }

        ("GET", "/api/history") => {
            let sess = state.replay.read().unwrap();
            let steps: Vec<_> = sess
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
            let res = HistoryResponse {
                seed: sess.seed,
                total_steps: sess.history.len(),
                total_rounds: sess.history.last().map(|s| s.round_number).unwrap_or(1),
                player_types: [
                    sess.player_types[0].as_str().to_string(),
                    sess.player_types[1].as_str().to_string(),
                ],
                steps,
            };
            let json = serde_json::to_vec(&res).unwrap();
            respond(&mut stream, "200 OK", &json, "application/json");
        }

        // ================= 静态文件服务 =================
        ("GET", _) | ("HEAD", _) => {
            let clean_path = path.trim_start_matches('/');
            let rel_path = if path == "/" || path == "/index.html" || path == "/play" || path == "/play.html" {
                if web_root.join("play.html").exists() {
                    "play.html".to_string()
                } else {
                    "index.html".to_string()
                }
            } else if path == "/replay" {
                "replay.html".to_string()
            } else {
                clean_path.to_string()
            };

            let file_path = web_root.join(&rel_path);
            if file_path.is_file() {
                let ext = file_path
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("");
                let content_type = match ext {
                    "html" => "text/html; charset=utf-8",
                    "js" => "application/javascript; charset=utf-8",
                    "css" => "text/css; charset=utf-8",
                    "webp" => "image/webp",
                    "png" => "image/png",
                    "jpg" | "jpeg" => "image/jpeg",
                    "svg" => "image/svg+xml",
                    "json" => "application/json",
                    _ => "application/octet-stream",
                };

                if let Ok(content) = fs::read(&file_path) {
                    if method == "HEAD" {
                        respond_headers_only(&mut stream, "200 OK", content.len(), content_type);
                    } else {
                        respond(&mut stream, "200 OK", &content, content_type);
                    }
                } else {
                    respond(&mut stream, "500 Internal Error", b"read failed", "text/plain");
                }
            } else {
                respond(&mut stream, "404 Not Found", b"File not found", "text/plain");
            }
        }

        _ => {
            respond(&mut stream, "405 Method Not Allowed", b"not allowed", "text/plain");
        }
    }
}

fn ensure_neural_server_running() {
    if NeuralAI::is_available() {
        println!("🧠 神经网络推理微服务已就绪 (127.0.0.1:8088)");
        return;
    }

    let candidates = [
        PathBuf::from("checkpoints/best.pt"),
        PathBuf::from("../checkpoints/best.pt"),
    ];

    let mut pt_path = None;
    for c in &candidates {
        if c.exists() {
            pt_path = Some(c.clone());
            break;
        }
    }

    let checkpoint = match pt_path {
        Some(p) => p,
        None => {
            println!("ℹ️ 未检测到 checkpoints/best.pt，神经网络对局功能将暂时回退为启发式 AI");
            return;
        }
    };

    println!("🚀 正在自动拉起神经网络推理后台服务 (端口 8088)...");

    let py_candidates = [
        PathBuf::from(".venv/Scripts/python.exe"),
        PathBuf::from(".venv/bin/python"),
        PathBuf::from("python.exe"),
        PathBuf::from("python"),
    ];

    let mut python_bin = None;
    for p in &py_candidates {
        if p.exists() {
            python_bin = Some(p.clone());
            break;
        }
    }

    let py_cmd = python_bin.unwrap_or_else(|| PathBuf::from("python"));
    let server_script = if Path::new("python/splendor_ai/server.py").exists() {
        "python/splendor_ai/server.py"
    } else {
        "../python/splendor_ai/server.py"
    };

    let current_pid = std::process::id();
    let mut spawn_cmd = std::process::Command::new(py_cmd);
    spawn_cmd.args([
        server_script,
        "--port",
        "8088",
        "--checkpoint",
        checkpoint.to_str().unwrap(),
        "--parent-pid",
        &current_pid.to_string(),
    ]);

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        spawn_cmd.creation_flags(CREATE_NO_WINDOW);
    }

    match spawn_cmd.spawn() {
        Ok(child) => {
            if let Ok(mut guard) = NEURAL_CHILD.lock() {
                *guard = Some(child);
            }
            println!("⏳ 等待神经网络推理微服务启动 (已绑定父进程 PID: {current_pid})...");
            for _ in 0..40 {
                std::thread::sleep(std::time::Duration::from_millis(100));
                if NeuralAI::is_available() {
                    println!("✅ 神经网络推理微服务已成功建立连接！");
                    return;
                }
            }
            println!("⚠️ 启动超时，稍后请求时可自动重试连接。");
        }
        Err(e) => {
            println!("⚠️ 自动启动 Python 推理服务失败: {e}");
        }
    }
}

fn main() {
    // 捕获 Ctrl+C，在主服务退出时优雅终止关联的 Python 推理微服务
    ctrlc::set_handler(move || {
        println!("\n🛑 收到退出信号，正在停止 replay_web 服务...");
        if let Ok(mut guard) = NEURAL_CHILD.lock() {
            if let Some(mut child) = guard.take() {
                println!("🛑 正在终止关联的 Python 神经网络推理子进程...");
                let _ = child.kill();
            }
        }
        std::process::exit(0);
    }).expect("Error setting Ctrl-C handler");

    let args: Vec<String> = env::args().collect();
    let port = args.get(1).and_then(|p| p.parse().ok()).unwrap_or(8080);
    let addr = format!("127.0.0.1:{port}");

    let listener = match TcpListener::bind(&addr) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("❌ 端口绑定失败 {addr}: {e}");
            std::process::exit(1);
        }
    };

    let web_root = find_web_root();

    // 尝试拉起神经网络推理微服务
    ensure_neural_server_running();

    let replay_session = ReplaySession::new(42);
    let interactive_session = InteractiveSession::new(42, [PlayerKind::Human, PlayerKind::Neural]);

    let app_state = Arc::new(AppState {
        replay: RwLock::new(replay_session),
        game: RwLock::new(interactive_session),
    });

    println!("\n========================================================");
    println!("💎 璀璨宝石：对决 (Splendor Duel) 全功能服务已启动！");
    println!("🎮 实时人机/双人对战:   http://{addr}/play.html");
    println!("🎬 AI 自博弈复盘分析:   http://{addr}/replay.html");
    println!("📁 静态网页根目录:      {}", web_root.display());
    println!("========================================================\n");

    for stream in listener.incoming() {
        if let Ok(stream) = stream {
            let app_state_clone = Arc::clone(&app_state);
            let root_clone = web_root.clone();
            std::thread::spawn(move || {
                handle_client(stream, &app_state_clone, &root_clone);
            });
        }
    }
}
