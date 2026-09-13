use _engine::{NeuralAI, PlayerType, ReplaySession};
use serde::Serialize;
use std::env;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

#[derive(Serialize)]
struct StepSummary {
    index: usize,
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
    player_types: [String; 2],
    steps: Vec<StepSummary>,
}

fn respond(stream: &mut TcpStream, status: &str, body: &[u8], content_type: &str) {
    let _ = write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(body);
}

fn find_web_root() -> PathBuf {
    let candidates = [
        PathBuf::from("engine/web"),
        PathBuf::from("web"),
        PathBuf::from("../engine/web"),
    ];
    for c in candidates {
        if c.exists() && c.join("index.html").exists() {
            return c;
        }
    }
    PathBuf::from("engine/web")
}

fn handle_client(mut stream: TcpStream, session: &Arc<RwLock<ReplaySession>>, web_root: &Path) {
    let mut buffer = [0u8; 4096];
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

    match (method, path) {
        // API: 获取最新状态
        ("GET", "/api/status") => {
            let sess = session.read().unwrap();
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

        // API: 检测神经网络服务健康状态
        ("GET", "/api/neural_status") => {
            let available = NeuralAI::is_available();
            let res = serde_json::json!({ "available": available });
            let json = serde_json::to_vec(&res).unwrap();
            respond(&mut stream, "200 OK", &json, "application/json");
        }

        // API: 单步推进或批量推进
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

            let mut sess = session.write().unwrap();
            let prev_len = sess.history.len();
            match sess.step_n(count) {
                Ok(advanced_count) => {
                    let last_step = sess.history.last().unwrap().clone();
                    let new_steps: Vec<StepSummary> = sess.history[prev_len..]
                        .iter()
                        .map(|s| StepSummary {
                            index: s.step_index,
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
                        step: _engine::ReplayStep,
                        new_steps: Vec<StepSummary>,
                    }
                    let res = StepResult {
                        advanced: advanced_count > 0,
                        advanced_count,
                        total_steps: sess.history.len(),
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

        // API: 一键推至终局
        ("POST", "/api/play_to_end") | ("GET", "/api/play_to_end") => {
            let mut sess = session.write().unwrap();
            let _ = sess.play_to_end(2000);
            let last_step = sess.history.last().unwrap().clone();
            #[derive(Serialize)]
            struct EndResult {
                total_steps: usize,
                step: _engine::ReplayStep,
            }
            let res = EndResult {
                total_steps: sess.history.len(),
                step: last_step,
            };
            let json = serde_json::to_vec(&res).unwrap();
            respond(&mut stream, "200 OK", &json, "application/json");
        }

        // API: 动态修改玩家 AI 类型
        ("POST", "/api/set_players") | ("GET", "/api/set_players") => {
            let mut sess = session.write().unwrap();
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

        // API: 重置对局 (支持设置双方 AI 类型与一键生成全局)
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

            let mut sess = session.write().unwrap();
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

        // API: 获取指定步数快照
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

            let sess = session.read().unwrap();
            if let Some(step) = sess.get_step(index) {
                let json = serde_json::to_vec(step).unwrap();
                respond(&mut stream, "200 OK", &json, "application/json");
            } else {
                respond(&mut stream, "404 Not Found", b"Step not found", "text/plain");
            }
        }

        // API: 获取历史列表
        ("GET", "/api/history") => {
            let sess = session.read().unwrap();
            let steps: Vec<_> = sess
                .history
                .iter()
                .map(|s| StepSummary {
                    index: s.step_index,
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
                player_types: [
                    sess.player_types[0].as_str().to_string(),
                    sess.player_types[1].as_str().to_string(),
                ],
                steps,
            };
            let json = serde_json::to_vec(&res).unwrap();
            respond(&mut stream, "200 OK", &json, "application/json");
        }

        // 静态文件服务
        ("GET", _) => {
            let rel_path = if path == "/" || path == "/index.html" {
                "index.html"
            } else {
                path.trim_start_matches('/')
            };

            let file_path = web_root.join(rel_path);
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
                    respond(&mut stream, "200 OK", &content, content_type);
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

    let mut found_ckpt = None;
    for c in &candidates {
        if c.exists() {
            found_ckpt = Some(c.clone());
            break;
        }
    }

    if let Some(ckpt) = found_ckpt {
        println!("🔍 检测到模型权重 {}，正在自动唤起后台 Python 推理微服务...", ckpt.display());
        let python_bins = [
            ".venv/Scripts/python.exe",
            "../.venv/Scripts/python.exe",
            ".venv/bin/python",
            "python",
        ];

        let mut py_exec = "python";
        for py in &python_bins {
            if Path::new(py).exists() {
                py_exec = py;
                break;
            }
        }

        let script = if Path::new("python/splendor_ai/server.py").exists() {
            "python/splendor_ai/server.py"
        } else {
            "../python/splendor_ai/server.py"
        };

        let child = std::process::Command::new(py_exec)
            .args([script, "--port", "8088", "--checkpoint", ckpt.to_str().unwrap()])
            .spawn();

        if let Ok(_) = child {
            // 等待 Python 进程初始化
            for _ in 0..15 {
                std::thread::sleep(std::time::Duration::from_millis(200));
                if NeuralAI::is_available() {
                    println!("✨ 神经网络推理微服务已成功自动启动并就绪 (127.0.0.1:8088)！");
                    return;
                }
            }
            println!("⏳ 神经网络推理微服务已发起启动，正在后台加载模型权重...");
        } else {
            println!("💡 提示: 未能自动拉起 Python，可通过命令行手动启动: python python/splendor_ai/server.py");
        }
    }
}

fn main() {
    let port = env::args()
        .nth(1)
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(8080);
    let addr = format!("127.0.0.1:{port}");

    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("无法绑定地址 {addr}: {e}");
        std::process::exit(1);
    });

    let web_root = find_web_root();
    println!("==================================================");
    println!("✨ 璀璨宝石：对决 (Splendor Duel) 可视化对局回放服务已启动！");
    println!("🌐 本地访问地址: http://{addr}");
    println!("📁 静态网页目录: {}", web_root.display());
    println!("==================================================");

    ensure_neural_server_running();

    let session = Arc::new(RwLock::new(ReplaySession::new(42)));

    for stream in listener.incoming() {
        if let Ok(stream) = stream {
            let session = Arc::clone(&session);
            let web_root = web_root.clone();
            std::thread::spawn(move || {
                handle_client(stream, &session, &web_root);
            });
        }
    }
}
