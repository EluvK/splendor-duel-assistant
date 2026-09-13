use _engine::{GameEngine, GameState, HeuristicAI, RandomAI, TurnPhase, VictoryReason};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::time::Instant;

fn main() {
    let num_threads = 8;
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .expect("Failed to build rayon thread pool");

    println!("============================================================");
    println!("🚀 《璀璨宝石：对决》Rust 性能基准测试 (Release 模式)");
    println!("⚙️  并发配置: {num_threads} 线程并行");
    println!("============================================================");

    // -------------------------------------------------------------
    // 测试组 1: 纯随机 AI vs 纯随机 AI (10,000 局)
    // -------------------------------------------------------------
    println!("\n▶️  [测试组 1] 纯随机 AI 自博弈 (10,000 局)");
    run_benchmark("RandomAI vs RandomAI", 10_000, |game, rng| {
        RandomAI::select_action(game, rng)
    });

    // -------------------------------------------------------------
    // 测试组 2: 启发式 AI vs 启发式 AI (20,000 局)
    // -------------------------------------------------------------
    println!("\n▶️  [测试组 2] 启发式 AI 自博弈 (20,000 局)");
    run_benchmark("HeuristicAI vs HeuristicAI", 20_000, |game, rng| {
        HeuristicAI::select_action(game, rng)
    });
}

fn run_benchmark<F>(name: &str, total_games: usize, select_fn: F)
where
    F: Fn(&GameState, &mut ChaCha8Rng) -> Option<_engine::Action> + Sync + Send,
{
    // 预热 500 局
    (0..500).into_par_iter().for_each(|seed| {
        let mut game = GameState::new_game(seed);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        while !matches!(game.phase, TurnPhase::GameOver(_)) {
            if let Some(action) = select_fn(&game, &mut rng) {
                let _ = GameEngine::step(&mut game, &action);
            } else {
                break;
            }
        }
    });

    let start_time = Instant::now();

    let results: Vec<(VictoryReason, u32)> = (0..total_games)
        .into_par_iter()
        .map(|seed| {
            let mut game = GameState::new_game(seed as u64);
            let mut rng = ChaCha8Rng::seed_from_u64(seed as u64);
            let mut steps: u32 = 0;

            while !matches!(game.phase, TurnPhase::GameOver(_)) {
                steps += 1;
                if steps > 2000 {
                    break;
                }
                if let Some(action) = select_fn(&game, &mut rng) {
                    let _ = GameEngine::step(&mut game, &action);
                } else {
                    break;
                }
            }

            let reason = match game.phase {
                TurnPhase::GameOver(r) => r,
                _ => VictoryReason::TwentyPrestigePoints,
            };

            (reason, steps)
        })
        .collect();

    let elapsed_sec = start_time.elapsed().as_secs_f64();

    let mut total_steps: u64 = 0;
    let mut reason_20_pts = 0;
    let mut reason_10_crowns = 0;
    let mut reason_single_color = 0;
    let mut step_list = Vec::with_capacity(total_games);

    for &(reason, steps) in results.iter() {
        total_steps += steps as u64;
        step_list.push(steps);
        match reason {
            VictoryReason::TwentyPrestigePoints => reason_20_pts += 1,
            VictoryReason::TenCrowns => reason_10_crowns += 1,
            VictoryReason::TenPointsSameColor(_) => reason_single_color += 1,
        }
    }

    step_list.sort_unstable();

    let avg_steps = total_steps as f64 / total_games as f64;
    let games_per_sec = total_games as f64 / elapsed_sec;
    let steps_per_sec = total_steps as f64 / elapsed_sec;

    let p50 = step_list[total_games * 50 / 100];
    let p90 = step_list[total_games * 90 / 100];
    let p99 = step_list[total_games * 99 / 100];
    let min_steps = step_list[0];
    let max_steps = step_list[total_games - 1];

    println!("------------------ {name} 汇总 ------------------");
    println!("⏱️  总耗时:           {elapsed_sec:.3} 秒");
    println!("🔥  对局吞吐量 (GPS):   {games_per_sec:.1} 局/秒");
    println!("⚡  状态转移吞吐 (SPS): {steps_per_sec:.1} 步/秒 ({:.2} 万步/秒)", steps_per_sec / 10000.0);
    println!("📊  平均对局步数:       {avg_steps:.1} 步 (P50: {p50}, P90: {p90}, P99: {p99}, Min: {min_steps}, Max: {max_steps})");
    println!("🏆  胜利条件分布:");
    println!("    - 20 声望获胜:      {reason_20_pts} 局 ({:.2}%)", reason_20_pts as f64 / total_games as f64 * 100.0);
    println!("    - 10 王冠获胜:      {reason_10_crowns} 局 ({:.2}%)", reason_10_crowns as f64 / total_games as f64 * 100.0);
    println!("    - 单色 10 分获胜:   {reason_single_color} 局 ({:.2}%)", reason_single_color as f64 / total_games as f64 * 100.0);
}
