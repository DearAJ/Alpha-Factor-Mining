"""
tree_viz.py - MCTS 搜索树可视化模块

唯一推荐用法 —— `SearchTreeVisualizer` API：

  from tree_viz import SearchTreeVisualizer

  # 方式 A：MCTS 挖完后直接可视化
  viz = SearchTreeVisualizer.from_mcts_method(method)
  viz.save_html("mcts_tree.html")
  viz.save_json("mcts_tree.json")
  print(viz.get_stats())

  # 方式 B：从已保存的 JSON 加载后可视化
  viz = SearchTreeVisualizer.from_json("mcts_tree.json")
  viz.save_html("mcts_tree.html")
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Conditionally import plotly ──────────────────────────────────────────
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


# ═════════════════════════════════════════════════════════════════════════
#  JSON 序列化
# ═════════════════════════════════════════════════════════════════════════

class NumpyEncoder(json.JSONEncoder):
    """处理 numpy 类型的 JSON 编码器。"""
    def default(self, o):
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) or np.isinf(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


# ═════════════════════════════════════════════════════════════════════════
#  语义等价表达式检测
# ═════════════════════════════════════════════════════════════════════════

def expression_signature(expr: str) -> str:
    """
    生成表达式的规范签名，归一化交换律操作（+ 和 *）。

    例如 "Rank(A) + Mean(B)" 和 "Mean(B) + Rank(A)" 会返回相同签名。
    用于检测 LLM 生成的语义等价但语法不同的表达式。
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval").body
    except SyntaxError:
        return expr.strip().lower()

    def _sig(node) -> tuple:
        """递归构建规范签名，交换律算子排序子节点。"""
        COMMUTATIVE = (ast.Add, ast.Mult)

        if isinstance(node, ast.BinOp):
            left = _sig(node.left)
            right = _sig(node.right)
            op = type(node.op).__name__
            if isinstance(node.op, COMMUTATIVE):
                left, right = sorted([left, right], key=str)
            return ("BinOp", op, left, right)
        elif isinstance(node, ast.UnaryOp):
            return ("UnaryOp", type(node.op).__name__, _sig(node.operand))
        elif isinstance(node, ast.Call):
            func = _sig(node.func)
            args = tuple(_sig(a) for a in node.args)
            return ("Call", func, args)
        elif isinstance(node, ast.Name):
            return ("Name", node.id.lower())
        elif isinstance(node, ast.Attribute):
            return ("Attr", _sig(node.value), node.attr.lower())
        elif isinstance(node, ast.Constant):
            return ("Const", repr(node.value))
        elif isinstance(node, ast.Compare):
            ops = tuple(type(o).__name__ for o in node.ops)
            parts = tuple(_sig(x) for x in [node.left] + node.comparators)
            return ("Compare", ops, parts)
        else:
            return (type(node).__name__,)

    return str(_sig(tree))


def find_equivalent_groups(nodes_data: list[dict]) -> list[list[str]]:
    """
    找出语义等价的节点组。

    nodes_data: flatten_tree() 返回的节点列表
    返回: [[node_id1, node_id2, ...], ...]  每组内节点语义等价
    """
    groups: dict[str, list[str]] = {}
    for node in nodes_data:
        expr = node.get("expression", "")
        sig = expression_signature(expr)
        groups.setdefault(sig, []).append(node["id"])
    return [g for g in groups.values() if len(g) > 1]


# ═════════════════════════════════════════════════════════════════════════
#  树结构与布局计算
# ═════════════════════════════════════════════════════════════════════════

def flatten_tree(tree_dict: dict) -> tuple[list[dict], list[dict]]:
    """
    DFS 遍历树字典，拍平为节点列表和边列表。
    返回 (nodes_data, edges_data)
    """
    nodes = []
    edges = []

    def dfs(node):
        nodes.append({
            "id": node["node_id"],
            "parent_id": node.get("parent_id"),
            "depth": node.get("depth", 0),
            "reward": node.get("reward", 0.0),
            "q_value": node.get("q_value", 0.0),
            "visits": node.get("visits", 1),
            "target_dimension": node.get("target_dimension", ""),
            "refinement_suggestion": node.get("refinement_suggestion", ""),
            "expression": node.get("candidate", {}).get("expression", ""),
            "explanation": node.get("candidate", {}).get("explanation", ""),
            "dimension_scores": node.get("dimension_scores", {}),
            "metrics": node.get("candidate", {}).get("metrics", {}),
            "extra": node.get("candidate", {}).get("extra", {}),
        })
        for child in node.get("children", []):
            edges.append({
                "from": node["node_id"],
                "to": child["node_id"],
                "dimension": child.get("target_dimension", ""),
                "suggestion": child.get("refinement_suggestion", ""),
            })
            dfs(child)

    dfs(tree_dict)
    return nodes, edges


def compute_tree_layout(
    nodes_data: list[dict],
    edges_data: list[dict],
    y_spacing: float = 0.13,
) -> dict[str, tuple[float, float]]:
    """
    计算树布局坐标（顶部根节点，自顶向下）。
    使用叶节点计数按比例分配水平空间，避免节点重叠。

    返回: {node_id: (x, y)}，x, y ∈ [0, 1]
    """
    # 构建邻接表
    children_of: dict[str, list[str]] = {}
    parents: dict[str, str] = {}
    for n in nodes_data:
        children_of.setdefault(n["id"], [])
    for e in edges_data:
        children_of.setdefault(e["from"], []).append(e["to"])
        parents[e["to"]] = e["from"]

    # 找根节点
    root_id = next(
        (n["id"] for n in nodes_data if n["id"] not in parents),
        nodes_data[0]["id"] if nodes_data else None,
    )
    if root_id is None:
        return {}

    # 统计每棵子树的叶节点数（用于分配水平空间）
    leaf_count: dict[str, int] = {}

    def _count_leaves(nid: str) -> int:
        ch = children_of.get(nid, [])
        if not ch:
            leaf_count[nid] = 1
        else:
            leaf_count[nid] = sum(_count_leaves(c) for c in ch)
        return leaf_count[nid]

    _count_leaves(root_id)
    total_leaves = leaf_count.get(root_id, 1)

    positions: dict[str, tuple[float, float]] = {}

    def _assign(nid: str, x0: float, x1: float, depth: int):
        x = (x0 + x1) / 2.0
        y = 1.0 - depth * y_spacing
        if y < 0.05:
            y = 0.05
        positions[nid] = (x, y)

        n_leaf = leaf_count.get(nid, 1)
        cx = x0
        for child in children_of.get(nid, []):
            c_leaf = leaf_count.get(child, 1)
            span = (x1 - x0) * c_leaf / n_leaf
            _assign(child, cx, cx + span, depth + 1)
            cx += span

    _assign(root_id, 0.0, 1.0, 0)
    return positions


# ═════════════════════════════════════════════════════════════════════════
#  Plotly 可视化
# ═════════════════════════════════════════════════════════════════════════

DEFAULT_DIMENSIONS = ["effectiveness", "diversity", "stability", "turnover", "overfit"]

# 中英维度名称映射（用于图表标签）
DIMENSION_LABELS = {
    "effectiveness": "有效性",
    "diversity": "多样性",
    "stability": "稳定性",
    "turnover": "换手率",
    "overfit": "过拟合",
}


def _short_expr(expr: str, max_len: int = 28) -> str:
    """截断表达式至指定长度。"""
    if len(expr) <= max_len:
        return expr
    return expr[: max_len - 3] + "..."


def _truncate(text: str, max_len: int = 80) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def build_tree_figure(
    nodes_data: list[dict],
    edges_data: list[dict],
    positions: dict[str, tuple[float, float]],
    equivalent_groups: list[list[str]] | None = None,
    title: str = "MCTS 因子表达式搜索树",
) -> go.Figure:
    """
    构建 Plotly 树形图。

    参数:
        nodes_data: flatten_tree 返回的节点列表
        edges_data: flatten_tree 返回的边列表
        positions: compute_tree_layout 返回的位置字典
        equivalent_groups: find_equivalent_groups 返回的等价组
        title: 图表标题

    返回:
        Plotly Figure（可 .write_html() 导出）
    """
    if not _HAS_PLOTLY:
        raise ImportError("需要安装 plotly: pip install plotly")

    # ── 构造等价节点的高亮集合 ──
    highlighted_ids: set[str] = set()
    if equivalent_groups:
        for group in equivalent_groups:
            highlighted_ids.update(group)

    fig = go.Figure()

    # ── 画边 ──
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_annotations = []

    for e in edges_data:
        if e["from"] not in positions or e["to"] not in positions:
            continue
        x0, y0 = positions[e["from"]]
        x1, y1 = positions[e["to"]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        # 边中点标注改进维度
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        dim_label = DIMENSION_LABELS.get(e["dimension"], e["dimension"])
        edge_annotations.append(dict(
            x=mid_x,
            y=mid_y,
            text=f"↑ {dim_label}",
            showarrow=False,
            font=dict(size=9, color="#666"),
            xanchor="center",
            yanchor="bottom",
        ))

    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(color="#bbb", width=2),
        hoverinfo="none",
        showlegend=False,
    ))

    # ── 画节点 ──
    if not nodes_data:
        return fig

    xs = []
    ys = []
    colors = []
    sizes = []
    texts = []
    hover_texts = []
    marker_symbols = []
    marker_line_colors = []

    max_reward = max((n["reward"] or 0) for n in nodes_data)
    min_reward = min((n["reward"] or 0) for n in nodes_data)
    reward_range = max_reward - min_reward if max_reward > min_reward else 1.0

    for n in nodes_data:
        nid = n["id"]
        if nid not in positions:
            continue
        x, y = positions[nid]
        xs.append(x)
        ys.append(y)

        reward = n["reward"] or 0.0
        colors.append(reward)

        depth = n["depth"] or 0
        sizes.append(max(14, 32 - depth * 4))

        texts.append(f"{_short_expr(n['expression'])}<br>R={reward:.3f}")

        # 悬停详细信息
        dim_scores = n.get("dimension_scores", {})
        dim_lines = []
        for k in DEFAULT_DIMENSIONS:
            v = dim_scores.get(k, None)
            label = DIMENSION_LABELS.get(k, k)
            if v is not None:
                dim_lines.append(f"  {label} ({k}): {v:.3f}")
            else:
                dim_lines.append(f"  {label} ({k}): N/A")

        metrics = n.get("metrics", {})
        ic_mean = metrics.get("ic_mean", None)
        ic_ir = metrics.get("ic_ir", None)

        suggestion = _truncate(n.get("refinement_suggestion", ""), 120)
        target_dim = DIMENSION_LABELS.get(n.get("target_dimension", ""), n.get("target_dimension", ""))

        hover = (
            f"<b>{nid}</b><br>"
            f"<b>深度:</b> {depth}<br><br>"
            f"<b>表达式:</b><br>"
            f"<code style='font-size:11px'>{n['expression']}</code><br><br>"
            f"<b>综合 Reward:</b> {reward:.4f}<br>"
            f"<b>Q-Value:</b> {n.get('q_value', 0):.4f}<br>"
            f"<b>访问次数:</b> {n.get('visits', 1)}<br><br>"
            f"<b>目标维度:</b> {target_dim}<br>"
            f"<b>改进建议:</b> {suggestion}<br><br>"
            f"<b>5 维评分:</b><br>"
            + "<br>".join(dim_lines) + "<br><br>"
            f"<b>IC(5) mean:</b> {ic_mean if ic_mean is not None else 'N/A'}<br>"
            f"<b>IC IR:</b> {ic_ir if ic_ir is not None else 'N/A'}"
        )
        hover_texts.append(hover)

        # 标记等价节点
        is_equivalent = nid in highlighted_ids
        marker_symbols.append("diamond" if is_equivalent else "circle")
        marker_line_colors.append("#e74c3c" if is_equivalent else "#333")

    if not xs:
        return fig

    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=colors,
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(
                title="Reward",
                x=1.02,
                len=0.6,
            ),
            line=dict(width=1.5, color=marker_line_colors),
            symbol=marker_symbols,
        ),
        text=texts,
        textposition="bottom center",
        textfont=dict(size=10, color="#333"),
        hovertext=hover_texts,
        hoverinfo="text",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            align="left",
        ),
        showlegend=False,
    ))

    # ── 图例说明（等价节点标记） ──
    if equivalent_groups:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=1.02, y=0.35,
            text=(
                "<b>图例</b><br>"
                "● 普通节点<br>"
                "◇ <span style='color:#e74c3c'>语义等价节点</span><br>"
                f"<span style='color:#e74c3c'>检测到 {len(equivalent_groups)} 组等价</span>"
            ),
            showarrow=False,
            font=dict(size=11),
            align="left",
            bordercolor="#ccc",
            borderwidth=1,
            borderpad=4,
            bgcolor="#f9f9f9",
        )

    # ── 边标签（用 fig.add_annotation） ──
    for ann in edge_annotations:
        fig.add_annotation(**ann)

    # ── 布局 ──
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=18),
        ),
        showlegend=False,
        hovermode="closest",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False,
            range=[-0.05, 1.05],
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False,
            range=[-0.05, 1.05],
        ),
        height=max(500, len(nodes_data) * 55),
        width=min(1400, max(800, len(nodes_data) * 40)),
        margin=dict(l=20, r=80, t=60, b=30),
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif"),
    )

    return fig


def build_evolution_table(
    path_nodes: list[dict],
    title: str = "表达式演化路径",
) -> go.Figure:
    """
    生成沿某条搜索路径的表达式演化表格。

    path_nodes: 从根到叶的节点列表（已排序）
    """
    if not _HAS_PLOTLY:
        raise ImportError("需要安装 plotly: pip install plotly")

    headers = ["深度", "节点 ID", "表达式", "Reward", "改进维度", "改进说明"]
    rows = []
    for i, n in enumerate(path_nodes):
        dim = DIMENSION_LABELS.get(n.get("target_dimension", ""), n.get("target_dimension", ""))
        suggestion = _truncate(n.get("refinement_suggestion", ""), 60)
        rows.append([
            str(n.get("depth", i)),
            n["id"],
            _short_expr(n["expression"], 40),
            f"{n.get('reward', 0):.4f}",
            dim,
            suggestion,
        ])

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=headers,
            fill_color="#2c3e50",
            font=dict(color="white", size=12),
            align="left",
            height=30,
        ),
        cells=dict(
            values=list(zip(*rows)) if rows else [[]] * len(headers),
            fill_color=["#f9f9f9", "white"],
            font=dict(size=11),
            align="left",
            height=25,
        ),
    )])

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        height=60 + len(rows) * 30,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════
#  合并可视化（树 + 雷达 + 表格）
# ═════════════════════════════════════════════════════════════════════════

def _collect_root_to_leaf_paths(
    tree_dict: dict,
) -> list[list[str]]:
    """收集所有从根到叶的路径（节点 ID 序列）。"""
    paths = []

    def dfs(node, current_path):
        children = node.get("children", [])
        if not children:
            paths.append(current_path + [node["node_id"]])
        else:
            for child in children:
                dfs(child, current_path + [node["node_id"]])

    dfs(tree_dict, [])
    return paths


def create_full_report(
    tree_dict: dict,
    output_path: str | Path = "mcts_search_tree.html",
    title: str = "MCTS 因子表达式搜索树",
) -> str:
    """
    生成完整的 HTML 报告，包含：
    - 交互式树形图（含路径连线、等价节点标记）
    - 每个节点的 5 维雷达图（通过下拉菜单切换 / 全部显示）
    - 表达式演化路径表格

    返回保存的文件路径。
    """
    if not _HAS_PLOTLY:
        raise ImportError("需要安装 plotly: pip install plotly")

    nodes_data, edges_data = flatten_tree(tree_dict)
    positions = compute_tree_layout(nodes_data, edges_data)
    equiv_groups = find_equivalent_groups(nodes_data)

    # 构建主树图
    tree_fig = build_tree_figure(
        nodes_data, edges_data, positions, equiv_groups, title=title,
    )

    # 构建每个节点的雷达图（作为子图网格）
    n_nodes = len(nodes_data)
    n_cols = 4
    n_rows = math.ceil(n_nodes / n_cols)

    if n_nodes > 0:
        radar_titles = [f"{n['id']}" for n in nodes_data]
        radar_fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=radar_titles,
            specs=[[{"type": "polar"}] * n_cols for _ in range(n_rows)],
            horizontal_spacing=0.02,
            vertical_spacing=0.08,
        )

        for idx, n in enumerate(nodes_data):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            dim_scores = n.get("dimension_scores", {})
            dims = DEFAULT_DIMENSIONS.copy()
            labels = [DIMENSION_LABELS.get(d, d) for d in dims]
            vals = [dim_scores.get(d, 0.0) for d in dims]
            vals_closed = vals + [vals[0]]
            labels_closed = labels + [labels[0]]

            radar_fig.add_trace(
                go.Scatterpolar(
                    r=vals_closed,
                    theta=labels_closed,
                    fill="toself",
                    fillcolor="rgba(39, 174, 96, 0.25)",
                    line=dict(color="#27ae60", width=1.5),
                    name=n["id"],
                    hoverinfo="skip",
                ),
                row=row, col=col,
            )

        radar_fig.update_layout(
            title=dict(text="各节点 5 维评分雷达图", x=0.5, font=dict(size=16)),
            showlegend=False,
            height=max(300, n_rows * 320),
            margin=dict(l=30, r=30, t=60, b=20),
            font=dict(family="Arial, sans-serif"),
        )

        # 更新每个子图的坐标轴范围
        for i in range(1, n_rows * n_cols + 1):
            radar_fig.update_polars(
                radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=8)),
                angularaxis=dict(tickfont=dict(size=9)),
                row=(i - 1) // n_cols + 1,
                col=(i - 1) % n_cols + 1,
            )
    else:
        radar_fig = None

    # 构建演化路径表格（取前 5 条最长路径）
    paths = _collect_root_to_leaf_paths(tree_dict)
    paths.sort(key=len, reverse=True)
    top_paths = paths[:5]

    path_tables = []
    for p_idx, path_ids in enumerate(top_paths):
        id_to_node = {n["id"]: n for n in nodes_data}
        path_nodes = [id_to_node[nid] for nid in path_ids if nid in id_to_node]
        if path_nodes:
            table_fig = build_evolution_table(
                path_nodes,
                title=f"演化路径 {p_idx + 1}",
            )
            path_tables.append(table_fig)

    # ── 合并为 HTML ──
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html_parts = [tree_fig.to_html(full_html=False, include_plotlyjs=True)]
    if radar_fig is not None:
        html_parts.append("<br><hr><br>")
        html_parts.append(radar_fig.to_html(full_html=False, include_plotlyjs=False))
    for tf in path_tables:
        html_parts.append("<br><hr><br>")
        html_parts.append(tf.to_html(full_html=False, include_plotlyjs=False))

    # 添加等价节点摘要
    if equiv_groups:
        html_parts.append("<br><hr><br>")
        html_parts.append('<div style="font-family:Arial;padding:10px;">')
        html_parts.append('<h3>⚠️ 语义等价节点检测</h3>')
        html_parts.append(
            f"<p>检测到 <b>{len(equiv_groups)}</b> 组语义等价的表达式 "
            f"（共涉及 <b>{sum(len(g) for g in equiv_groups)}</b> 个节点）。"
            f"这些节点的语法不同但计算语义相同，可能浪费搜索预算。</p>"
        )
        for g_idx, group in enumerate(equiv_groups):
            html_parts.append(f"<h4>等价组 {g_idx + 1}</h4><ul>")
            for nid in group:
                node = next((n for n in nodes_data if n["id"] == nid), None)
                if node:
                    html_parts.append(
                        f"<li><b>{nid}</b>: <code>{node['expression']}</code></li>"
                    )
            html_parts.append("</ul>")
        html_parts.append("</div>")

    # 组装完整 HTML
    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ margin: 0; padding: 20px; font-family: Arial, sans-serif; background: #fff; }}
  h1, h2, h3 {{ color: #2c3e50; }}
  hr {{ border: none; border-top: 1px solid #ddd; }}
</style>
</head>
<body>
<h1 style="text-align:center;">{title}</h1>
<p style="text-align:center;color:#666;font-size:14px;">
  节点颜色: 绿高 Reward / 红低 Reward &nbsp;|&nbsp;
  菱形 ◇ = 语义等价节点 &nbsp;|&nbsp;
  边标签 = 改进维度
</p>
""" + "\n".join(html_parts) + """
</body>
</html>"""

    output_path.write_text(full_html, encoding="utf-8")
    return str(output_path)


# ═════════════════════════════════════════════════════════════════════════
#  高级 API
# ═════════════════════════════════════════════════════════════════════════

class SearchTreeVisualizer:
    """MCTS 搜索树可视化器，封装了从序列化到渲染的全流程。"""

    def __init__(self, tree_dict: dict):
        self.tree_dict = tree_dict
        self.nodes_data, self.edges_data = flatten_tree(tree_dict)
        self.positions = compute_tree_layout(self.nodes_data, self.edges_data)
        self.equivalent_groups = find_equivalent_groups(self.nodes_data)

    @classmethod
    def from_mcts_method(cls, method) -> "SearchTreeVisualizer":
        """从 AlphaJungleMCTSMethod 实例创建可视化器。"""
        if not hasattr(method, "_search_tree_root") or method._search_tree_root is None:
            raise ValueError("MCTS 方法尚未执行搜索，或 _search_tree_root 为空。")
        return cls(method._search_tree_root.to_dict())

    @classmethod
    def from_json(cls, path: str | Path) -> "SearchTreeVisualizer":
        """从 JSON 文件加载树数据。"""
        with open(path, "r", encoding="utf-8") as f:
            tree_dict = json.load(f)
        return cls(tree_dict)

    def save_html(self, path: str | Path = "mcts_search_tree.html", title: str | None = None) -> str:
        """生成完整 HTML 报告。"""
        t = title or f"MCTS 搜索树（{len(self.nodes_data)} 节点）"
        return create_full_report(self.tree_dict, path, title=t)

    def save_json(self, path: str | Path = "mcts_tree.json") -> str:
        """将树数据保存为 JSON。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.tree_dict, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        return str(path)

    def get_stats(self) -> dict:
        """返回搜索树的统计信息。"""
        n_nodes = len(self.nodes_data)
        n_edges = len(self.edges_data)
        max_depth = max((n.get("depth", 0) for n in self.nodes_data), default=0)
        n_equiv = sum(len(g) for g in self.equivalent_groups)

        all_rewards = [n.get("reward", 0) or 0 for n in self.nodes_data]
        avg_reward = np.mean(all_rewards) if all_rewards else 0

        return {
            "total_nodes": n_nodes,
            "total_edges": n_edges,
            "max_depth": max_depth,
            "equivalent_groups": len(self.equivalent_groups),
            "equivalent_nodes": n_equiv,
            "avg_reward": float(avg_reward),
            "best_reward": float(max(all_rewards)) if all_rewards else 0,
        }


# ═════════════════════════════════════════════════════════════════════════
#  CLI 入口
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MCTS 搜索树可视化工具（推荐使用编程方式调用 SearchTreeVisualizer）",
    )
    parser.add_argument("--load", type=str, required=True, help="已保存的 JSON 树文件路径")
    parser.add_argument("--output", type=str, default="mcts_search_tree.html",
                        help="输出 HTML 路径")
    parser.add_argument("--stats", action="store_true", help="仅打印统计信息，不生成 HTML")

    args = parser.parse_args()

    if not _HAS_PLOTLY:
        print("错误: 需要安装 plotly: pip install plotly", file=sys.stderr)
        sys.exit(1)

    viz = SearchTreeVisualizer.from_json(args.load)

    stats = viz.get_stats()
    print(f"总节点数: {stats['total_nodes']}")
    print(f"最大深度: {stats['max_depth']}")
    print(f"语义等价组: {stats['equivalent_groups']} 组 ({stats['equivalent_nodes']} 节点)")
    print(f"平均 Reward: {stats['avg_reward']:.4f}")
    print(f"最佳 Reward: {stats['best_reward']:.4f}")

    if args.stats:
        return

    html_path = viz.save_html(args.output)
    print(f"HTML 报告已保存: {html_path}")


if __name__ == "__main__":
    main()
