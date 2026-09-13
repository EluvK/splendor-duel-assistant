use _engine::{GameEngine, GameState, RandomAI, TurnPhase, VictoryReason};
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
    println!("🚀 《璀璨宝石：对决》Rust 游戏引擎性能基准测试 (Release 模式)");
    println!("⚙️  并发配置: {num_threads} 线程并行");
    println!("============================================================");

    // 预热 1,000 局
    println!("正在预热 JIT 与 CPU 缓存 (1,000 局)...");
    (0..1_000).into_par_iter().for_each(|seed| {
        let mut game = GameState::new_game(seed);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        while !matches!(game.phase, TurnPhase::GameOver(_)) {
            if let Some(action) = RandomAI::select_action(&game, &mut rng) {
                let _ = GameEngine::step(&mut game, &action);
            } else {
                break;
            }
        }
    });

    let total_games: usize = 100_000;
    println!("开始正式基准测试: 总对局数 = {total_games} 局...\n");

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
                if let Some(action) = RandomAI::select_action(&game, &mut rng) {
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

    let duration = start_time.elapsed();
    let elapsed_sec = duration.as_secs_f64();

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

    println!("================== 测试结果汇总 ==================");
    println!("⏱️  总耗时:           {elapsed_sec:.3} 秒");
    println!("🔥  对局吞吐量 (GPS):   {games_per_sec:.1} 局/秒");
    println!("⚡  状态转移吞吐 (SPS): {steps_per_sec:.1} 步/秒 ({:.2} 万步/秒)", steps_per_sec / 10000.0);
    println!("--------------------------------------------------");
    println!("📊  平均对局步数:       {avg_steps:.1} 步");
    println!("    - 最短对局:         {min_steps} 步");
    println!("    - 中位数 (P50):     {p50} 步");
    println!("    - P90 步数:         {p90} 步");
    println!("    - P99 步数:         {p99} 步");
    println!("    - 最长对局:         {max_steps} 步");
    println!("--------------------------------------------------");
    println!("🏆  胜利条件分布:");
    println!("    - 20 声望获胜:      {reason_20_pts} 局 ({:.2}%)", reason_20_pts as f64 / total_games as f64 * 100.0);
    println!("    - 10 王冠获胜:      {reason_10_crowns} 局 ({:.2}%)", reason_10_crowns as f64 / total_games as f64 * 100.0);
    println!("    - 单色 10 分获胜:   {reason_single_color} 局 ({:.2}%)", reason_single_color as f64 / total_games as f64 * 100.0);
    println!("==================================================");
}
