"""Round-Robin Tournament and Cycle Diagnostic Evaluator for Splendor Duel Checkpoints.

Evaluates all or selected checkpoints (iter_*.pt) in a full round-robin tournament
using ultra-fast 0-sim PolicyNet (or Neural MCTS) via the native Rust engine.

Key Diagnostics:
- Full pairwise win-rate matrix & Elo ratings
- Immediate predecessor progress (does iter_N beat iter_{N-1}?)
- Historical regressions (does iter_N lose to earlier iter_M where M < N?)
- Intransitive cycles (Rock-Paper-Scissors loops: A > B > C > A)
- Monotonicity score across self-play training
- Export to CSV, JSON, and standalone interactive HTML heatmap
"""

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import torch

# 修复 Windows 控制台下 GBK 编码输出 Emoji 时崩溃的问题
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from splendor_ai._engine import evaluate_neural_match
from splendor_ai.net import SplendorNet


@dataclass
class CheckpointInfo:
    name: str
    path: Optional[Path]
    iteration: Optional[int]
    onnx_bytes: Optional[bytes]  # None for HeuristicAI
    meta: Dict[str, Any]


@dataclass
class MatchupRecord:
    model_a: str
    model_b: str
    a_wins: int
    b_wins: int
    draws: int
    total_games: int
    a_win_rate: float
    elapsed_sec: float
    avg_rounds: float
    avg_steps: float
    reasons: Dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Splendor Duel Round-Robin Tournament & Cycle Diagnostic Evaluator"
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="checkpoints",
        help="Directory containing checkpoint files (default: checkpoints)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=r"^iter_(\d+)\.pt$",
        help="Regex pattern to match checkpoint files (default: ^iter_(\\d+)\\.pt$)",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=25,
        help="Number of paired games per matchup (total = 2 * pairs, default: 25 = 50 games)",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=0,
        help="MCTS simulations per step (default: 0 = pure PolicyNet intuition, ultra fast)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Stride step for sampling checkpoints (e.g. 5 samples every 5th checkpoint)",
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=None,
        help="Only evaluate the latest N checkpoints",
    )
    parser.add_argument(
        "--start-iter",
        type=int,
        default=None,
        help="Minimum iteration number to include (inclusive)",
    )
    parser.add_argument(
        "--end-iter",
        type=int,
        default=None,
        help="Maximum iteration number to include (inclusive)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Explicit list of model paths/names to evaluate (e.g. checkpoints/iter_004.pt checkpoints/iter_057.pt)",
    )
    parser.add_argument(
        "--include-heuristic",
        action="store_true",
        help="Include HeuristicAI (rule-based expert anchor) in the tournament",
    )
    parser.add_argument(
        "--include-best",
        action="store_true",
        help="Include checkpoints/best.pt in the tournament",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="checkpoints/tournament_results",
        help="Directory to save CSV, JSON, and HTML reports (default: checkpoints/tournament_results)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable ONNX bytes disk caching in checkpoints/.onnx_cache",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducible matchups (default: 42)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress live per-match output, show only progress summary and results",
    )

    return parser.parse_args()


def get_cached_onnx_bytes(ckpt_path: Path, use_cache: bool = True) -> Tuple[bytes, Dict[str, Any]]:
    """读取或导出 ONNX 字节流，并使用磁盘缓存加速二次加载."""
    cache_dir = ckpt_path.parent / ".onnx_cache"
    mtime = int(ckpt_path.stat().st_mtime)
    cache_file = cache_dir / f"{ckpt_path.stem}_{mtime}.onnx"

    meta: Dict[str, Any] = {}
    if use_cache and cache_file.exists() and cache_file.stat().st_size > 0:
        try:
            # 尝试快速加载元数据
            ckpt = torch.load(ckpt_path, map_location="cpu")
            meta = ckpt.get("meta", {})
            return cache_file.read_bytes(), meta
        except Exception:
            pass

    # 重新从 PyTorch checkpoint 导出
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    net = SplendorNet()
    net.load_state_dict(ckpt["model_state"])
    onnx_bytes = net.export_onnx_bytes()

    if use_cache:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(onnx_bytes)
        except Exception as e:
            print(f"   ⚠️ 写入 ONNX 缓存失败: {e}")

    return onnx_bytes, meta


def discover_checkpoints(
    ckpt_dir: Path,
    pattern_str: str,
    step: int = 1,
    latest: Optional[int] = None,
    start_iter: Optional[int] = None,
    end_iter: Optional[int] = None,
    explicit_models: Optional[List[str]] = None,
    include_heuristic: bool = False,
    include_best: bool = False,
    use_cache: bool = True,
) -> List[CheckpointInfo]:
    """发现并加载待对决的 checkpoint 列表并转换为 ONNX 字节."""
    pattern = re.compile(pattern_str)
    discovered: List[Tuple[int, Path]] = []

    if explicit_models:
        for m in explicit_models:
            p = Path(m)
            if not p.exists():
                raise FileNotFoundError(f"指定的模型不存在: {m}")
            match = pattern.match(p.name)
            it = int(match.group(1)) if match else 999999
            discovered.append((it, p))
    else:
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint 目录不存在: {ckpt_dir}")

        for p in ckpt_dir.iterdir():
            if not p.is_file() or not p.name.endswith(".pt"):
                continue
            match = pattern.match(p.name)
            if match:
                it = int(match.group(1))
                if start_iter is not None and it < start_iter:
                    continue
                if end_iter is not None and it > end_iter:
                    continue
                discovered.append((it, p))

    # 按迭代轮次升序排序
    discovered.sort(key=lambda x: x[0])

    if step > 1 and len(discovered) > 0:
        # 步长采样，并保留最后一个 checkpoint
        sampled = discovered[::step]
        if discovered[-1] not in sampled:
            sampled.append(discovered[-1])
        discovered = sampled

    if latest is not None and latest > 0:
        discovered = discovered[-latest:]

    items: List[CheckpointInfo] = []
    total_to_load = len(discovered) + (1 if include_best else 0)
    print(f"📦 正在准备模型 ONNX 运行时数据 (共 {total_to_load} 个候选模型)...")

    for idx, (it, p) in enumerate(discovered, start=1):
        t0 = time.time()
        b, meta = get_cached_onnx_bytes(p, use_cache=use_cache)
        items.append(
            CheckpointInfo(
                name=p.stem,
                path=p,
                iteration=it,
                onnx_bytes=b,
                meta=meta,
            )
        )
        print(f"   [{idx}/{total_to_load}] 已加载 {p.name:<18} (迭代: {it:>3} | 耗时: {(time.time()-t0)*1000:.1f}ms)")

    if include_best:
        best_path = ckpt_dir / "best.pt"
        if best_path.exists():
            b, meta = get_cached_onnx_bytes(best_path, use_cache=use_cache)
            items.append(
                CheckpointInfo(
                    name="best",
                    path=best_path,
                    iteration=meta.get("iteration"),
                    onnx_bytes=b,
                    meta=meta,
                )
            )
            print(f"   [+] 已加载 best.pt (迭代: {meta.get('iteration')})")

    if include_heuristic:
        items.append(
            CheckpointInfo(
                name="HeuristicAI",
                path=None,
                iteration=None,
                onnx_bytes=None,  # Rust 引擎识别为 None 时走内置专家启发式 AI
                meta={"type": "heuristic"},
            )
        )
        print("   [+] 已挂载 HeuristicAI 启发式基准锚点")

    return items


def compute_elo_ratings(
    models: List[str],
    matchups: List[MatchupRecord],
    base_elo: float = 1000.0,
    k_factor: float = 32.0,
    max_iters: int = 50,
) -> Dict[str, float]:
    """基于所有对抗结果的 Bradley-Terry / 迭代 Elo 等级分计算."""
    elo = {m: base_elo for m in models}
    if not matchups:
        return elo

    for _ in range(max_iters):
        delta_max = 0.0
        for m in matchups:
            r1 = elo[m.model_a]
            r2 = elo[m.model_b]
            e1 = 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))
            e2 = 1.0 - e1

            tot = m.total_games
            if tot == 0:
                continue

            s1 = (m.a_wins + 0.5 * m.draws) / tot
            s2 = (m.b_wins + 0.5 * m.draws) / tot

            weight = min(tot / 50.0, 2.0)
            d1 = k_factor * (s1 - e1) * weight
            d2 = k_factor * (s2 - e2) * weight

            elo[m.model_a] += d1
            elo[m.model_b] += d2
            delta_max = max(delta_max, abs(d1), abs(d2))

        if delta_max < 0.05:
            break

    # 规范化：让均分保持为 base_elo
    mean_elo = sum(elo.values()) / max(len(elo), 1)
    for m in elo:
        elo[m] = round(elo[m] + (base_elo - mean_elo), 1)

    return elo


def detect_cycles_and_regressions(
    models: List[CheckpointInfo],
    matrix: Dict[str, Dict[str, float]],
    raw_wins: Dict[str, Dict[str, Tuple[int, int, int, int]]],  # (a_wins, b_wins, draws, total)
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
    """检测策略循环与历史倒退.
    
    1. 历史倒退 (Regressions): 新模型 M_i 战胜前一代 M_{i-1}，但败给更早代际 M_j (j < i - 1)。
    2. 策略三角死循环 (3-Cycles): A 胜 B, B 胜 C, C 胜 A (剪刀石头布循环)。
    3. 全局单调性得分: 所有 (新代 vs 老代) 对决中新代胜出的比例。
    """
    model_names = [m.name for m in models]
    n = len(models)

    # 1. 历史倒退检测
    regressions: List[Dict[str, Any]] = []
    total_older_pairs = 0
    newer_wins_older = 0

    for i in range(1, n):
        curr_m = models[i]
        prev_m = models[i - 1]
        
        # 仅针对具有明确 iteration 递增的模型比较
        if curr_m.iteration is None or prev_m.iteration is None:
            continue
        if curr_m.iteration <= prev_m.iteration:
            continue

        prev_wr = matrix[curr_m.name].get(prev_m.name, 0.5)
        beat_prev = prev_wr > 0.5

        lost_older_list = []
        for j in range(i - 1):
            older_m = models[j]
            if older_m.iteration is None or older_m.iteration >= curr_m.iteration:
                continue

            total_older_pairs += 1
            wr = matrix[curr_m.name].get(older_m.name, 0.5)
            if wr > 0.5:
                newer_wins_older += 1
            elif wr < 0.5:
                w1, w2, d, tot = raw_wins[curr_m.name][older_m.name]
                lost_older_list.append({
                    "older_model": older_m.name,
                    "older_iter": older_m.iteration,
                    "win_rate": wr,
                    "score": f"{w1}-{w2}" + (f" (平{d})" if d > 0 else ""),
                })

        if beat_prev and lost_older_list:
            w1_prev, w2_prev, d_prev, _ = raw_wins[curr_m.name][prev_m.name]
            regressions.append({
                "model": curr_m.name,
                "iteration": curr_m.iteration,
                "prev_model": prev_m.name,
                "prev_iter": prev_m.iteration,
                "prev_win_rate": prev_wr,
                "prev_score": f"{w1_prev}-{w2_prev}" + (f" (平{d_prev})" if d_prev > 0 else ""),
                "regressions_count": len(lost_older_list),
                "lost_older": sorted(lost_older_list, key=lambda x: x["win_rate"]),
            })

    monotonicity_score = newer_wins_older / max(total_older_pairs, 1)

    # 2. 三元循环检测 (A > B, B > C, C > A)
    cycles: List[Dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                m_a = model_names[i]
                m_b = model_names[j]
                m_c = model_names[k]

                w_ab = matrix[m_a][m_b]
                w_ba = matrix[m_b][m_a]
                w_bc = matrix[m_b][m_c]
                w_cb = matrix[m_c][m_b]
                w_ca = matrix[m_c][m_a]
                w_ac = matrix[m_a][m_c]

                # 方向 1: A > B, B > C, C > A
                if w_ab > 0.50 and w_bc > 0.50 and w_ca > 0.50:
                    margin = min(w_ab, w_bc, w_ca) - 0.50
                    cycles.append({
                        "cycle": f"{m_a} ➔ {m_b} ➔ {m_c} ➔ {m_a}",
                        "a": m_a,
                        "b": m_b,
                        "c": m_c,
                        "margin": round(margin * 100, 1),
                        "rates": f"{m_a}胜{m_b}: {w_ab*100:.1f}% | {m_b}胜{m_c}: {w_bc*100:.1f}% | {m_c}胜{m_a}: {w_ca*100:.1f}%",
                    })
                # 方向 2: A > C, C > B, B > A
                elif w_ac > 0.50 and w_cb > 0.50 and w_ba > 0.50:
                    margin = min(w_ac, w_cb, w_ba) - 0.50
                    cycles.append({
                        "cycle": f"{m_a} ➔ {m_c} ➔ {m_b} ➔ {m_a}",
                        "a": m_a,
                        "b": m_c,
                        "c": m_b,
                        "margin": round(margin * 100, 1),
                        "rates": f"{m_a}胜{m_c}: {w_ac*100:.1f}% | {m_c}胜{m_b}: {w_cb*100:.1f}% | {m_b}胜{m_a}: {w_ba*100:.1f}%",
                    })

    cycles.sort(key=lambda x: x["margin"], reverse=True)

    return regressions, cycles, monotonicity_score


def generate_html_report(
    models: List[CheckpointInfo],
    matchups: List[MatchupRecord],
    matrix: Dict[str, Dict[str, float]],
    raw_wins: Dict[str, Dict[str, Tuple[int, int, int, int]]],
    elo_ratings: Dict[str, float],
    regressions: List[Dict[str, Any]],
    cycles: List[Dict[str, Any]],
    monotonicity: float,
    elapsed_total: float,
    pairs: int,
    sims: int,
    out_path: Path,
) -> None:
    """生成完全独立的自包含交互式 HTML 战力热力图与循环诊断报告."""
    total_games = len(matchups) * pairs * 2
    model_names = [m.name for m in models]
    n = len(models)

    # 构造表格行
    table_rows = []
    for rank, (name, elo) in enumerate(
        sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True), start=1
    ):
        tot_wins = sum(raw_wins[name][other][0] for other in model_names if other != name)
        tot_losses = sum(raw_wins[name][other][1] for other in model_names if other != name)
        tot_draws = sum(raw_wins[name][other][2] for other in model_names if other != name)
        tot_g = tot_wins + tot_losses + tot_draws
        wr = (tot_wins + 0.5 * tot_draws) / max(tot_g, 1)

        table_rows.append(
            f"<tr>"
            f"<td><b>#{rank}</b></td>"
            f"<td><b>{name}</b></td>"
            f"<td><span class='badge elo'>{elo:.0f}</span></td>"
            f"<td><b>{wr*100:.1f}%</b></td>"
            f"<td>{tot_wins} / {tot_losses} / {tot_draws}</td>"
            f"<td>{tot_g}</td>"
            f"</tr>"
        )

    # 构造矩阵表头与单元格
    header_cells = "".join(f"<th title='{name}'>{name}</th>" for name in model_names)
    matrix_rows = []
    for m1 in model_names:
        cells = [f"<th class='row-header' title='{m1}'>{m1}</th>"]
        for m2 in model_names:
            if m1 == m2:
                cells.append("<td class='cell-diag'>-</td>")
            else:
                wr = matrix[m1][m2]
                w1, w2, d, tot = raw_wins[m1][m2]
                pct = wr * 100
                tooltip = f"{m1} vs {m2}&#10;胜率: {pct:.1f}%&#10;战绩: {w1}胜 {w2}负 {d}平 (共{tot}局)"

                if pct > 50.0:
                    alpha = min(0.85, 0.2 + (pct - 50.0) / 50.0 * 0.65)
                    bg = f"background-color: rgba(46, 204, 113, {alpha:.2f});"
                elif pct < 50.0:
                    alpha = min(0.85, 0.2 + (50.0 - pct) / 50.0 * 0.65)
                    bg = f"background-color: rgba(231, 76, 60, {alpha:.2f});"
                else:
                    bg = "background-color: rgba(149, 165, 166, 0.3);"

                cells.append(
                    f"<td style='{bg}' title='{tooltip}' class='cell-match'>{pct:.0f}%</td>"
                )
        matrix_rows.append("<tr>" + "".join(cells) + "</tr>")

    # 历史倒退列表 HTML
    reg_items = []
    for reg in regressions[:20]:
        lost_str = ", ".join(
            f"<b>{o['older_model']}</b> ({o['win_rate']*100:.1f}%)" for o in reg["lost_older"]
        )
        reg_items.append(
            f"<div class='card card-reg'>"
            f"<h4>⚡ {reg['model']} (胜前任 {reg['prev_model']}: {reg['prev_win_rate']*100:.1f}%) "
            f"<span class='badge danger'>倒退 {reg['regressions_count']} 处</span></h4>"
            f"<p class='desc'>被更早的模型反杀: {lost_str}</p>"
            f"</div>"
        )
    if not reg_items:
        reg_items.append("<div class='card success'><p>🎉 未检测到显著的历史倒退现象，代际晋升表现单调递增！</p></div>")

    # 循环死锁列表 HTML
    cycle_items = []
    for c in cycles[:15]:
        cycle_items.append(
            f"<div class='card card-cycle'>"
            f"<h4>🔄 三角克制循环 (循环强度: +{c['margin']}%)</h4>"
            f"<p class='mono'>{c['cycle']}</p>"
            f"<p class='desc'>{c['rates']}</p>"
            f"</div>"
        )
    if not cycle_items:
        cycle_items.append("<div class='card success'><p>🎉 未检测到强三角克制死循环 (No RPS cycles found)！</p></div>")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>璀璨宝石：对决 Checkpoints 循环赛与策略退化诊断报告</title>
    <style>
        :root {{
            --bg: #0f172a;
            --surface: #1e293b;
            --border: #334155;
            --text: #f8fafc;
            --text-dim: #94a3b8;
            --accent: #38bdf8;
            --green: #22c55e;
            --red: #ef4444;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            padding: 24px;
            line-height: 1.5;
        }}
        .header {{
            margin-bottom: 24px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
        }}
        .header h1 {{ font-size: 26px; margin-bottom: 8px; color: var(--accent); }}
        .header p {{ color: var(--text-dim); font-size: 14px; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .stat-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }}
        .stat-card .val {{ font-size: 24px; font-weight: bold; color: var(--text); }}
        .stat-card .lbl {{ font-size: 12px; color: var(--text-dim); text-transform: uppercase; margin-top: 4px; }}
        .section {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 24px;
        }}
        .section h2 {{
            font-size: 18px;
            margin-bottom: 16px;
            color: var(--accent);
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th, td {{
            padding: 8px 10px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{ color: var(--text-dim); font-weight: 600; background: rgba(0,0,0,0.2); }}
        .matrix-wrap {{
            overflow: auto;
            max-height: 680px;
            border: 1px solid var(--border);
            border-radius: 6px;
        }}
        .matrix-table th, .matrix-table td {{
            text-align: center;
            min-width: 44px;
            max-width: 58px;
            padding: 6px 2px;
            font-size: 11px;
            cursor: default;
        }}
        .matrix-table th {{
            position: sticky;
            top: 0;
            z-index: 2;
            background: #1e293b;
        }}
        .matrix-table th.row-header {{
            position: sticky;
            left: 0;
            z-index: 1;
            text-align: right;
            padding-right: 8px;
            background: #1e293b;
            min-width: 90px;
        }}
        .cell-diag {{ background: #334155; color: #64748b; }}
        .badge {{
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
        }}
        .badge.elo {{ background: #0284c7; color: #fff; }}
        .badge.danger {{ background: #dc2626; color: #fff; }}
        .card {{
            background: rgba(0,0,0,0.2);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px 16px;
            margin-bottom: 10px;
        }}
        .card.card-reg {{ border-left: 4px solid #ef4444; }}
        .card.card-cycle {{ border-left: 4px solid #f59e0b; }}
        .card.success {{ border-left: 4px solid #22c55e; }}
        .card h4 {{ font-size: 14px; margin-bottom: 4px; }}
        .card .mono {{ font-family: monospace; color: #38bdf8; font-size: 13px; margin: 4px 0; }}
        .card .desc {{ font-size: 12px; color: var(--text-dim); }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⚔️ 《璀璨宝石：对决》全代际 Checkpoints 循环对决与策略循环诊断报告</h1>
        <p>评估配置: 纯直觉网络 (sims={sims}) | 每对决局数: {pairs*2} 局严格换座 | 参战模型: {n} 个 | 总耗时: {elapsed_total:.1f}s</p>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="val">{n}</div>
            <div class="lbl">参评模型总数</div>
        </div>
        <div class="stat-card">
            <div class="val">{len(matchups)}</div>
            <div class="lbl">循环对决场次 ({total_games} 局)</div>
        </div>
        <div class="stat-card">
            <div class="val" style="color: {'#22c55e' if monotonicity >= 0.75 else '#ef4444'}">{monotonicity*100:.1f}%</div>
            <div class="lbl">全局单调递增率 (新代胜老代)</div>
        </div>
        <div class="stat-card">
            <div class="val" style="color: {'#ef4444' if len(regressions) > 0 else '#22c55e'}">{len(regressions)}</div>
            <div class="lbl">历史倒退点 (胜前任却负更早)</div>
        </div>
        <div class="stat-card">
            <div class="val" style="color: {'#f59e0b' if len(cycles) > 0 else '#22c55e'}">{len(cycles)}</div>
            <div class="lbl">发现三角剪刀石头布循环</div>
        </div>
    </div>

    <div class="section">
        <h2>📊 综合战力天梯排行榜 (按 Elo 等级分排序)</h2>
        <table>
            <thead>
                <tr>
                    <th>名次</th>
                    <th>模型名称</th>
                    <th>Elo 等级分</th>
                    <th>总胜率</th>
                    <th>胜 / 负 / 平</th>
                    <th>总局数</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>🔥 相互胜率交叉热力图 (行作为主角 对战 列作为对手)</h2>
        <p style="font-size: 12px; color: var(--text-dim); margin-bottom: 12px;">说明: 绿色代表该行模型胜率 > 50%，红色代表胜率 < 50%。鼠标悬停可查看局数与精准数据。</p>
        <div class="matrix-wrap">
            <table class="matrix-table">
                <thead>
                    <tr>
                        <th class="row-header">Model \\ Opp</th>
                        {header_cells}
                    </tr>
                </thead>
                <tbody>
                    {''.join(matrix_rows)}
                </tbody>
            </table>
        </div>
    </div>

    <div class="section">
        <h2>⚡ 历史倒退诊断 (新一代战胜前任，但却被更早代际反杀)</h2>
        <p style="font-size: 12px; color: var(--text-dim); margin-bottom: 12px;">这是自博弈强化学习中常见的策略过拟合或循环现象：新模型针对前任发展出了特化克制策略，却重开了对古老策略的防御漏洞。</p>
        {''.join(reg_items)}
    </div>

    <div class="section">
        <h2>🔄 策略死循环与非传递性结构 (Rock-Paper-Scissors 3-Cycles)</h2>
        <p style="font-size: 12px; color: var(--text-dim); margin-bottom: 12px;">检测到的非传递性闭环三角 (A 胜 B 且 B 胜 C 且 C 胜 A)：</p>
        {''.join(cycle_items)}
    </div>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def run_tournament(
    ckpt_dir: str = "checkpoints",
    pattern: str = r"^iter_(\d+)\.pt$",
    pairs: int = 25,
    sims: int = 0,
    step: int = 1,
    latest: Optional[int] = None,
    start_iter: Optional[int] = None,
    end_iter: Optional[int] = None,
    models: Optional[List[str]] = None,
    include_heuristic: bool = False,
    include_best: bool = False,
    out_dir: str = "checkpoints/tournament_results",
    use_cache: bool = True,
    seed: int = 42,
    quiet: bool = False,
) -> None:
    """运行多模型两两循环对弈评测与诊断."""
    ckpt_path = Path(ckpt_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    t0_start = time.time()
    discovered_models = discover_checkpoints(
        ckpt_dir=ckpt_path,
        pattern_str=pattern,
        step=step,
        latest=latest,
        start_iter=start_iter,
        end_iter=end_iter,
        explicit_models=models,
        include_heuristic=include_heuristic,
        include_best=include_best,
        use_cache=use_cache,
    )

    n = len(discovered_models)
    if n < 2:
        print(f"❌ 至少需要 2 个模型进行循环赛，当前仅找到 {n} 个模型。")
        return

    total_matchups = n * (n - 1) // 2
    total_games_per_match = pairs * 2
    total_games = total_matchups * total_games_per_match

    mode_str = f"Neural MCTS ({sims} 次/步)" if sims > 0 else "极速纯直觉 PolicyNet (0 次推演)"
    print("\n" + "=" * 80)
    print(f"⚔️ 《璀璨宝石：对决》全代际循环赛对抗评测 (Tournament Round-Robin)")
    print(f"   • 参战模型: {n} 个 ({discovered_models[0].name} ... {discovered_models[-1].name})")
    print(f"   • 对决场次: {total_matchups} 组两两对决 (每组 {total_games_per_match} 局严格换座 | 累计: {total_games} 局)")
    print(f"   • 决策引擎: {mode_str} | 并行加速: Rayon 纯 Rust 多线程引擎")
    print(f"   • 结果输出: {out_path.resolve()}")
    print("=" * 80 + "\n")

    # 初始化矩阵数据结构
    model_names = [m.name for m in discovered_models]
    matrix: Dict[str, Dict[str, float]] = {m: {other: 0.5 for other in model_names} for m in model_names}
    raw_wins: Dict[str, Dict[str, Tuple[int, int, int, int]]] = {
        m: {other: (0, 0, 0, 0) for other in model_names} for m in model_names
    }
    matchup_records: List[MatchupRecord] = []

    match_idx = 0
    t_start_matches = time.time()

    for i in range(n):
        for j in range(i + 1, n):
            match_idx += 1
            m_a = discovered_models[i]
            m_b = discovered_models[j]

            match_seed = (seed + match_idx * 10007) & 0x7FFFFFFF
            t_m0 = time.time()

            # 执行 Rust Rayon 并行对弈
            tot, a_wins, b_wins, draws, reasons = evaluate_neural_match(
                m_a.onnx_bytes,
                m_b.onnx_bytes,
                num_pairs=pairs,
                base_seed=match_seed,
                num_sims=sims,
            )
            m_dur = time.time() - t_m0

            wr_a = (a_wins + 0.5 * draws) / max(tot, 1)
            wr_b = (b_wins + 0.5 * draws) / max(tot, 1)

            matrix[m_a.name][m_b.name] = wr_a
            matrix[m_b.name][m_a.name] = wr_b

            raw_wins[m_a.name][m_b.name] = (a_wins, b_wins, draws, tot)
            raw_wins[m_b.name][m_a.name] = (b_wins, a_wins, draws, tot)

            rec = MatchupRecord(
                model_a=m_a.name,
                model_b=m_b.name,
                a_wins=a_wins,
                b_wins=b_wins,
                draws=draws,
                total_games=tot,
                a_win_rate=wr_a,
                elapsed_sec=m_dur,
                avg_rounds=reasons.get("total_rounds", 0) / max(tot, 1),
                avg_steps=reasons.get("total_steps", 0) / max(tot, 1),
                reasons=reasons,
            )
            matchup_records.append(rec)

            # 进度计算
            elapsed_matches = time.time() - t_start_matches
            avg_match_time = elapsed_matches / match_idx
            remaining_sec = avg_match_time * (total_matchups - match_idx)
            pct_done = match_idx / total_matchups * 100

            mins_rem, secs_rem = divmod(int(remaining_sec), 60)
            eta_str = f"{mins_rem:02d}:{secs_rem:02d}"

            if not quiet or match_idx == total_matchups or match_idx % 10 == 0:
                score_str = f"{a_wins:>2} : {b_wins:<2}"
                if draws > 0:
                    score_str += f" (平 {draws})"
                rate_str = f"{m_a.name} 胜率 {wr_a*100:5.1f}%"
                print(
                    f"[{match_idx:>3}/{total_matchups:>3}] ({pct_done:5.1f}%) ⏳ 剩余: {eta_str} | "
                    f"{m_a.name:<16} vs {m_b.name:<16} ➔ {score_str:<12} ({rate_str}) [{m_dur:.2f}s]"
                )

    total_time = time.time() - t0_start

    print("\n" + "=" * 80)
    print("📈 循环赛全部完成！正在执行 Elo 等级分收敛与策略循环诊断...")
    print("=" * 80)

    # 1. 计算 Elo 等级分
    elo_ratings = compute_elo_ratings(model_names, matchup_records)

    # 2. 检测历史倒退与三角死循环
    regressions, cycles, monotonicity = detect_cycles_and_regressions(
        discovered_models, matrix, raw_wins
    )

    # 3. 打印天梯榜单
    print("\n" + "─" * 80)
    print("🏆 【全代际战力天梯榜单】(按照全场综合 Elo 评级排序)")
    print("─" * 80)
    print(f"{'排名':<4} {'模型名称':<20} {'Elo等级分':<10} {'总胜率':<8} {'胜-负-平':<14} {'对前任胜率':<12} {'历史倒退次数'}")
    print("─" * 80)

    sorted_by_elo = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, elo) in enumerate(sorted_by_elo, start=1):
        tot_w = sum(raw_wins[name][other][0] for other in model_names if other != name)
        tot_l = sum(raw_wins[name][other][1] for other in model_names if other != name)
        tot_d = sum(raw_wins[name][other][2] for other in model_names if other != name)
        tot_g = tot_w + tot_l + tot_d
        wr = (tot_w + 0.5 * tot_d) / max(tot_g, 1)

        # 找到其在 discovered_models 中的索引
        idx = next((i for i, m in enumerate(discovered_models) if m.name == name), None)
        vs_prev_str = "-"
        reg_count_str = "-"
        if idx is not None and idx > 0 and discovered_models[idx].iteration is not None:
            prev_name = discovered_models[idx - 1].name
            prev_wr = matrix[name][prev_name]
            vs_prev_str = f"{prev_wr*100:5.1f}%"

            # 统计其落败于更早模型的次数
            curr_it = discovered_models[idx].iteration
            reg_cnt = sum(
                1 for j in range(idx)
                if discovered_models[j].iteration is not None
                and discovered_models[j].iteration < curr_it
                and matrix[name][discovered_models[j].name] < 0.50
            )
            reg_count_str = f"{reg_cnt} 次" if reg_cnt > 0 else "0 (无倒退)"

        print(
            f"#{rank:<3} {name:<20} {elo:>6.0f}     {wr*100:5.1f}%   "
            f"{f'{tot_w}-{tot_l}-{tot_d}':<14} {vs_prev_str:<12} {reg_count_str}"
        )
    print("─" * 80)

    # 4. 打印核心诊断：历史倒退
    print("\n" + "─" * 80)
    print(f"⚡ 【策略循环与历史倒退诊断】(全局单调递增率: {monotonicity*100:.1f}%)")
    print("─" * 80)
    if regressions:
        print(f"⚠️  共发现 {len(regressions)} 处典型代际倒退 (即: 胜过上一代，却败给更早代际):")
        for reg in regressions[:10]:
            lost_detail = ", ".join(
                f"{o['older_model']}({o['win_rate']*100:.1f}%)" for o in reg["lost_older"][:4]
            )
            if len(reg["lost_older"]) > 4:
                lost_detail += f" 等共 {len(reg['lost_older'])} 个"
            print(
                f"   • 💥 {reg['model']:<15} 胜过前任 {reg['prev_model']} ({reg['prev_win_rate']*100:.1f}%)，"
                f"却败给更早代际: {lost_detail}"
            )
        if len(regressions) > 10:
            print(f"   ... 其余 {len(regressions)-10} 处详见生成的 HTML/JSON 报告。")
    else:
        print("   ✅ 未发现显著的历史倒退现象，代际性能稳健提升！")

    # 5. 打印三角克制死循环
    if cycles:
        print(f"\n🔄 共检测到 {len(cycles)} 个策略三元死循环 (A 胜 B, B 胜 C, C 胜 A):")
        for c in cycles[:6]:
            print(f"   • 🔄 强度 +{c['margin']}%: {c['cycle']}")
            print(f"     └─ {c['rates']}")
        if len(cycles) > 6:
            print(f"   ... 其余 {len(cycles)-6} 个死循环详见 HTML 报告。")
    else:
        print("\n✅ 未检测到明显的策略三角死循环结构。")

    # 6. 若模型数量 <= 15，在终端打印矩阵
    if n <= 15:
        print("\n" + "─" * 80)
        print("📊 【相互胜率交叉矩阵】(行 vs 列胜率 %)")
        print("─" * 80)
        hdr = f"{'Model':<12} " + " ".join(f"{m[:8]:>8}" for m in model_names)
        print(hdr)
        for m1 in model_names:
            row_str = f"{m1:<12} "
            for m2 in model_names:
                if m1 == m2:
                    row_str += "       - "
                else:
                    wr = matrix[m1][m2] * 100
                    row_str += f"  {wr:5.1f}%"
            print(row_str)
        print("─" * 80)

    # 7. 导出数据文件 (CSV, JSON, HTML)
    matrix_csv_path = out_path / "matrix.csv"
    with open(matrix_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + model_names)
        for m1 in model_names:
            writer.writerow([m1] + [f"{matrix[m1][m2]:.4f}" if m1 != m2 else "0.5" for m2 in model_names])

    leaderboard_csv_path = out_path / "leaderboard.csv"
    with open(leaderboard_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "model", "iteration", "elo", "total_games", "wins", "losses", "draws", "win_rate"])
        for rank, (name, elo) in enumerate(sorted_by_elo, start=1):
            tot_w = sum(raw_wins[name][other][0] for other in model_names if other != name)
            tot_l = sum(raw_wins[name][other][1] for other in model_names if other != name)
            tot_d = sum(raw_wins[name][other][2] for other in model_names if other != name)
            tot_g = tot_w + tot_l + tot_d
            wr = (tot_w + 0.5 * tot_d) / max(tot_g, 1)
            it = next((m.iteration for m in discovered_models if m.name == name), None)
            writer.writerow([rank, name, it if it is not None else "", f"{elo:.1f}", tot_g, tot_w, tot_l, tot_d, f"{wr:.4f}"])

    json_path = out_path / "tournament_results.json"
    json_data = {
        "metadata": {
            "total_models": n,
            "total_matchups": total_matchups,
            "pairs_per_match": pairs,
            "games_per_match": total_games_per_match,
            "sims": sims,
            "monotonicity_score": monotonicity,
            "elapsed_seconds": total_time,
        },
        "elo_ratings": elo_ratings,
        "regressions": regressions,
        "cycles": cycles,
        "matrix": matrix,
        "matchups": [asdict(m) for m in matchup_records],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    html_path = out_path / "report.html"
    generate_html_report(
        models=discovered_models,
        matchups=matchup_records,
        matrix=matrix,
        raw_wins=raw_wins,
        elo_ratings=elo_ratings,
        regressions=regressions,
        cycles=cycles,
        monotonicity=monotonicity,
        elapsed_total=total_time,
        pairs=pairs,
        sims=sims,
        out_path=html_path,
    )

    print(f"\n💾 战报与全量数据已成功生成至: {out_path.resolve()}")
    print(f"   • 🌐 交互式热力图报告: {html_path.resolve()}")
    print(f"   • 📑 胜率交叉矩阵 CSV: {matrix_csv_path.resolve()}")
    print(f"   • 📊 战力天梯排行 CSV: {leaderboard_csv_path.resolve()}")
    print(f"   • 📦 完整结构化结果 JSON: {json_path.resolve()}")
    print(f"⏱️ 全程总耗时: {total_time:.2f}s (平均每局推演: {total_time/max(total_games,1)*1000:.1f}ms)\n")


def main() -> None:
    args = parse_args()
    run_tournament(
        ckpt_dir=args.ckpt_dir,
        pattern=args.pattern,
        pairs=args.pairs,
        sims=args.sims,
        step=args.step,
        latest=args.latest,
        start_iter=args.start_iter,
        end_iter=args.end_iter,
        models=args.models,
        include_heuristic=args.include_heuristic,
        include_best=args.include_best,
        out_dir=args.out_dir,
        use_cache=not args.no_cache,
        seed=args.seed,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
