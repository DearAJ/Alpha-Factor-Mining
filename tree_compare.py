"""
tree_compare.py - 跨轮次 MCTS 搜索树对比分析

用法:
    # 独立使用：指定 glob 模式加载多轮树 JSON，输出对比 HTML
    python tree_compare.py --load "viz_output/cycle_*_tree.json"

    # 集成至 run.py（推荐）：指定 --max_cycles 跑完后自动输出对比报告
    python run.py --method_config ./method_config/alpha_jungle_mcts.yaml --max_cycles 5
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

from tree_viz import (
    SearchTreeVisualizer,
    DEFAULT_DIMENSIONS,
)

# plotly 可选导入（与 tree_viz 风格一致）
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

# 维度中文标签（复用 tree_viz 的映射）
DIM_LABELS = {
    "effectiveness": "有效性",
    "diversity": "多样性",
    "stability": "稳定性",
    "turnover": "换手率",
    "overfit": "过拟合",
}


# ═════════════════════════════════════════════════════════════════════
#  数据加载
# ═════════════════════════════════════════════════════════════════════

def load_cycle_trees(pattern: str = "viz_output/cycle_*_tree.json") -> list[dict]:
    """
    批量加载各轮次的树 JSON，按轮次编号排序后返回。

    返回 list[dict]，每个 dict 包含:
      - cycle: int        轮次编号
      - path: str         原始 JSON 路径
      - viz: SearchTreeVisualizer  可视化器实例
      - stats: dict       get_stats() 结果
      - best_node: dict   该轮 reward 最高的节点
    """
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[tree_compare] 未找到匹配 {pattern} 的树文件")
        return []

    cycle_data = []
    for fpath in files:
        m = re.search(r"cycle_(\d+)_tree\.json$", fpath)
        if not m:
            continue
        cycle = int(m.group(1))

        viz = SearchTreeVisualizer.from_json(fpath)
        stats = viz.get_stats()

        # 该轮 reward 最高的节点
        best_node = (
            max(viz.nodes_data, key=lambda n: n.get("reward", 0) or 0)
            if viz.nodes_data else None
        )

        cycle_data.append({
            "cycle": cycle,
            "path": fpath,
            "viz": viz,
            "stats": stats,
            "best_node": best_node,
        })

    cycle_data.sort(key=lambda x: x["cycle"])
    return cycle_data


# ═════════════════════════════════════════════════════════════════════
#  可视化图表
# ═════════════════════════════════════════════════════════════════════

def build_metrics_trend(cycle_data: list[dict]) -> go.Figure:
    """
    2×2 指标趋势面板：
    - 左上: Best Reward 折线
    - 右上: Avg Reward 折线
    - 左下: 节点数 柱状
    - 右下: 语义等价节点数 柱状
    """
    cycles = [d["cycle"] for d in cycle_data]
    best_rewards = [d["stats"]["best_reward"] for d in cycle_data]
    avg_rewards = [d["stats"]["avg_reward"] for d in cycle_data]
    total_nodes = [d["stats"]["total_nodes"] for d in cycle_data]
    equiv_nodes = [d["stats"]["equivalent_nodes"] for d in cycle_data]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Best Reward", "Avg Reward", "节点数", "语义等价节点数"),
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    # TL: Best Reward
    fig.add_trace(
        go.Scatter(x=cycles, y=best_rewards, mode="lines+markers",
                   name="Best Reward",
                   line=dict(color="#27ae60", width=2),
                   marker=dict(size=10, color="#27ae60")),
        row=1, col=1,
    )
    # 标注具体数值
    for c, v in zip(cycles, best_rewards):
        fig.add_annotation(x=c, y=v, text=f"{v:.3f}", showarrow=False,
                           yshift=10, font=dict(size=9, color="#27ae60"),
                           row=1, col=1)

    # TR: Avg Reward
    fig.add_trace(
        go.Scatter(x=cycles, y=avg_rewards, mode="lines+markers",
                   name="Avg Reward",
                   line=dict(color="#2980b9", width=2),
                   marker=dict(size=10, color="#2980b9")),
        row=1, col=2,
    )
    for c, v in zip(cycles, avg_rewards):
        fig.add_annotation(x=c, y=v, text=f"{v:.3f}", showarrow=False,
                           yshift=10, font=dict(size=9, color="#2980b9"),
                           row=1, col=2)

    # BL: Total Nodes
    fig.add_trace(
        go.Bar(x=cycles, y=total_nodes, name="节点数",
               marker_color="#f39c12", opacity=0.8),
        row=2, col=1,
    )

    # BR: Equivalent Nodes
    fig.add_trace(
        go.Bar(x=cycles, y=equiv_nodes, name="等价节点数",
               marker_color="#e74c3c", opacity=0.8),
        row=2, col=2,
    )

    fig.update_layout(
        title=dict(text="跨轮次搜索指标趋势", x=0.5, font=dict(size=16)),
        height=520,
        width=860,
        showlegend=False,
        font=dict(family="Arial, sans-serif"),
        margin=dict(l=50, r=30, t=60, b=50),
        plot_bgcolor="white",
    )

    # X 轴显示整数轮次
    for row, col in [(2, 1), (2, 2)]:
        fig.update_xaxes(dtick=1, row=row, col=col)

    return fig


def build_best_factor_table(cycle_data: list[dict]) -> go.Figure:
    """
    各轮次最佳因子对比表：
    轮次 | 表达式 | Reward | 5 维评分 (5列) | IC IR
    """
    headers = [
        "轮次", "最佳表达式", "Reward",
        "有效性", "多样性", "稳定性", "换手率", "过拟合", "IC IR",
    ]

    rows = []
    for d in cycle_data:
        n = d["best_node"]
        if n is None:
            continue

        expr = n.get("expression", "")
        expr_short = expr[:45] + "..." if len(expr) > 45 else expr

        reward = n.get("reward", 0) or 0
        dims = n.get("dimension_scores", {})

        metrics = n.get("metrics", {})
        ic_ir_v = metrics.get("ic_ir")
        ic_ir_str = f"{ic_ir_v:.3f}" if isinstance(ic_ir_v, (int, float)) else "N/A"

        rows.append([
            str(d["cycle"]),
            f"<code>{expr_short}</code>",
            f"{reward:.4f}",
            f"{dims.get('effectiveness', 0):.2f}",
            f"{dims.get('diversity', 0):.2f}",
            f"{dims.get('stability', 0):.2f}",
            f"{dims.get('turnover', 0):.2f}",
            f"{dims.get('overfit', 0):.2f}",
            ic_ir_str,
        ])

    if not rows:
        return go.Figure()

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=headers,
            fill_color="#2c3e50",
            font=dict(color="white", size=11),
            align="left",
            height=32,
        ),
        cells=dict(
            values=list(zip(*rows)),
            fill_color=["#f9f9f9", "white"],
            font=dict(size=10),
            align="left",
            height=26,
            # 交替行颜色
            format=[None] * len(headers),
        ),
    )])

    fig.update_layout(
        title=dict(text="各轮次最佳因子对比", x=0.5, font=dict(size=16)),
        height=100 + len(rows) * 30,
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Arial, sans-serif"),
    )
    return fig


def build_dimension_radar_overlay(cycle_data: list[dict]) -> go.Figure:
    """
    各轮次最佳因子 5 维评分雷达图叠加。
    方便直观对比不同轮次搜索焦点的变化。
    """
    colors = ["#27ae60", "#2980b9", "#f39c12", "#e74c3c", "#8e44ad", "#1abc9c"]

    fig = go.Figure()
    for idx, d in enumerate(cycle_data):
        n = d["best_node"]
        if n is None:
            continue
        dims = n.get("dimension_scores", {})
        vals = [dims.get(k, 0) for k in DEFAULT_DIMENSIONS]
        vals_closed = vals + [vals[0]]
        labels = [DIM_LABELS.get(k, k) for k in DEFAULT_DIMENSIONS] + [DIM_LABELS.get(DEFAULT_DIMENSIONS[0], DEFAULT_DIMENSIONS[0])]

        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=labels,
            fill="toself",
            fillcolor=colors[idx % len(colors)].replace(")", ", 0.15)").replace("rgb", "rgba"),
            line=dict(color=colors[idx % len(colors)], width=2),
            name=f"Cycle {d['cycle']}",
        ))

    fig.update_layout(
        title=dict(text="各轮次最佳因子 5 维评分对比", x=0.5, font=dict(size=16)),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        showlegend=True,
        legend=dict(x=1.1, y=0.5),
        height=500,
        width=700,
        font=dict(family="Arial, sans-serif"),
        margin=dict(l=80, r=100, t=60, b=40),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════
#  报告聚合
# ═════════════════════════════════════════════════════════════════════

def create_comparison_report(
    cycle_data: list[dict],
    output_path: str | Path = "viz_output/comparison_report.html",
) -> str:
    """
    生成跨轮次对比 HTML 报告，包含：
    - 搜索概览摘要
    - 指标趋势图（Best / Avg Reward, 节点数, 等价节点）
    - 最佳因子对比表
    - 5 维评分雷达叠加图
    """
    if len(cycle_data) < 1:
        print("[tree_compare] 没有数据，跳过报告生成")
        return ""
    if len(cycle_data) < 2:
        print("[tree_compare] 只有 1 轮数据，跳过对比报告（已保存单轮 HTML）")
        return ""

    trend_fig = build_metrics_trend(cycle_data)
    table_fig = build_best_factor_table(cycle_data)
    radar_fig = build_dimension_radar_overlay(cycle_data)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 概览统计
    first, last = cycle_data[0], cycle_data[-1]
    best_reward_change = last["stats"]["best_reward"] - first["stats"]["best_reward"]
    change_symbol = "↑" if best_reward_change >= 0 else "↓"

    html_parts = [
        trend_fig.to_html(full_html=False, include_plotlyjs=True),
        "<br><hr><br>",
        radar_fig.to_html(full_html=False, include_plotlyjs=False),
        "<br><hr><br>",
        table_fig.to_html(full_html=False, include_plotlyjs=False),
    ]

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>跨轮次 MCTS 搜索树对比</title>
<style>
  body {{ margin: 0; padding: 20px; font-family: Arial, sans-serif; background: #fff; }}
  h1, h2, h3 {{ color: #2c3e50; }}
  hr {{ border: none; border-top: 1px solid #ddd; }}
  .summary {{ background: #f0f4f8; padding: 15px 20px; border-radius: 6px;
              margin: 10px 0; line-height: 1.8; }}
  .stat {{ display: inline-block; margin-right: 24px; }}
  .stat-label {{ font-size: 12px; color: #666; }}
  .stat-value {{ font-size: 18px; font-weight: bold; color: #2c3e50; }}
</style>
</head>
<body>
<h1 style="text-align:center;">MCTS 搜索树跨轮次对比报告</h1>
<div class="summary">
  <div class="stat"><div class="stat-label">总轮次</div><div class="stat-value">{len(cycle_data)}</div></div>
  <div class="stat"><div class="stat-label">轮次范围</div><div class="stat-value">Cycle {first['cycle']} → Cycle {last['cycle']}</div></div>
  <div class="stat"><div class="stat-label">Best Reward</div><div class="stat-value">{first['stats']['best_reward']:.3f} → {last['stats']['best_reward']:.3f} <span style="color:{'#27ae60' if best_reward_change >= 0 else '#e74c3c'}">{change_symbol}{abs(best_reward_change):.3f}</span></div></div>
  <div class="stat"><div class="stat-label">Avg Reward</div><div class="stat-value">{first['stats']['avg_reward']:.3f} → {last['stats']['avg_reward']:.3f}</div></div>
  <div class="stat"><div class="stat-label">语义等价组</div><div class="stat-value">{first['stats']['equivalent_groups']} → {last['stats']['equivalent_groups']}</div></div>
</div>
""" + "\n".join(html_parts) + """
</body>
</html>"""

    output_path.write_text(full_html, encoding="utf-8")
    print(f"跨轮次对比报告已保存: {output_path}")
    return str(output_path)


# ═════════════════════════════════════════════════════════════════════
#  CLI 入口
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="跨轮次 MCTS 搜索树对比分析")
    parser.add_argument(
        "--load", type=str, default="viz_output/cycle_*_tree.json",
        help="树 JSON 文件 glob 模式（默认 viz_output/cycle_*_tree.json）",
    )
    parser.add_argument(
        "--output", type=str, default="viz_output/comparison_report.html",
        help="输出 HTML 路径",
    )
    args = parser.parse_args()

    cycle_data = load_cycle_trees(args.load)
    if not cycle_data:
        print("没有找到树文件，请先运行 MCTS 挖掘。")
        sys.exit(1)

    print(f"\n加载了 {len(cycle_data)} 轮数据:")
    print(f"  {'轮次':<6} {'节点数':<8} {'深度':<6} {'等价组':<8} {'Best Reward':<12} {'Avg Reward':<12}")
    print(f"  {'-'*52}")
    for d in cycle_data:
        s = d["stats"]
        print(f"  Cycle {d['cycle']:<2} {s['total_nodes']:<8} {s['max_depth']:<6} "
              f"{s['equivalent_groups']:<8} {s['best_reward']:<12.4f} {s['avg_reward']:<12.4f}")

    create_comparison_report(cycle_data, args.output)


if __name__ == "__main__":
    main()
