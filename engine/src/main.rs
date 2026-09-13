use _engine::{GameEngine, GameState, RandomAI, TurnPhase};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

fn main() {
    println!("=== 璀璨宝石：对决 (Splendor Duel) Rust 游戏引擎 ===");
    let mut game = GameState::new_game(2026);
    let mut rng = ChaCha8Rng::seed_from_u64(2026);

    println!("游戏初始化完成！");
    println!("起始棋盘标记数: {}", game.board.count_tokens());
    println!("后手玩家特权数: {}", game.players[1].privileges);

    let mut step_count = 0;
    while !matches!(game.phase, TurnPhase::GameOver(_)) {
        step_count += 1;
        if step_count > 1000 {
            println!("单局超过 1000 步，强制终止保护。");
            break;
        }

        if let Some(action) = RandomAI::select_action(&game, &mut rng) {
            if let Err(e) = GameEngine::step(&mut game, &action) {
                eprintln!("执行动作错误: {e:?}");
                break;
            }
        } else {
            eprintln!("当前阶段 {:?} 无合法动作，可能出现死锁！", game.phase);
            break;
        }
    }

    if let TurnPhase::GameOver(reason) = game.phase {
        println!("\n=== 游戏结束！===");
        let winner = game.winner.map(|(p, _)| p).unwrap_or(game.current_player);
        println!("获胜玩家: Player {winner}");
        println!("获胜原因: {reason:?}");
        println!("总步数: {step_count}");
        println!(
            "Player 0: 分数 = {}, 王冠 = {}",
            game.players[0].total_points, game.players[0].total_crowns
        );
        println!(
            "Player 1: 分数 = {}, 王冠 = {}",
            game.players[1].total_points, game.players[1].total_crowns
        );
    }
}
