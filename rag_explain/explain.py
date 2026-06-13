"""
explain.py — Top-N 因子 RAG 可解释性分析主脚本

流程：
1. 读取因子 metrics CSV，按 mcts_reward（或指定列）排序取 Top-N
2. 对每个因子：表达式翻译 → 构造检索 query → Chroma 检索知识片段
3. 组装 prompt，调用 DeepSeek（复用 llm_client.query）生成解释
4. 输出 Markdown 报告 report.md

用法：
    python rag_explain/explain.py \
        --metrics ../results_adaptive_uct_linear/metrics/cycle_0001_*.csv \
        --topn 10 \
        --out report.md

无 DEEPSEEK_API_KEY 时仍可生成报告（跳过 LLM 解释，保留翻译+检索结果）。
"""

import os
import sys
import glob
import argparse
from pathlib import Path

import pandas as pd

# 允许从项目根目录 import llm_client
PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from factor_semantics import describe_expression, detect_factor_families
from retriever import retrieve

# 排序候选列（优先级从高到低，取第一个存在的）
RANK_COLUMNS = ["mcts_reward", "ic_ir", "ic_mean"]

# 报告中展示的指标列 → 中文名
METRIC_LABELS = {
    "ic_mean": "IC均值",
    "ic_ir": "IC_IR",
    "ict": "IC t值",
    "long_excret": "多头年化超额",
    "long_sharpe": "多头Sharpe",
    "long_ir": "多头IR",
    "ls_sharpe": "多空Sharpe",
    "mcts_reward": "MCTS综合得分",
}


def load_topn(metrics_glob: str, topn: int) -> pd.DataFrame:
    """读取（可能多个）CSV，合并去重，按打分排序取 Top-N。"""
    files = sorted(glob.glob(metrics_glob))
    if not files:
        raise SystemExit(f"未匹配到 metrics 文件：{metrics_glob}")

    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    if "expression" not in df.columns:
        raise SystemExit("CSV 缺少 expression 列，无法分析。")

    # 同表达式去重，保留打分最高的
    rank_col = next((c for c in RANK_COLUMNS if c in df.columns), None)
    if rank_col:
        df = df.sort_values(rank_col, ascending=False)
    df = df.drop_duplicates(subset="expression", keep="first")
    return df.head(topn).reset_index(drop=True)


def build_query(expression: str, translation: str, families: list[str]) -> str:
    """构造用于知识库检索的查询文本。"""
    return f"{'、'.join(families)}因子。{translation} 表达式：{expression}"


def build_prompt(row: pd.Series, translation: str, families: list[str], hits: list[dict]) -> str:
    """组装发送给 LLM 的提示词。"""
    metrics_lines = []
    for col, label in METRIC_LABELS.items():
        if col in row and pd.notna(row[col]):
            try:
                metrics_lines.append(f"- {label}: {float(row[col]):.4f}")
            except (ValueError, TypeError):
                pass
    metrics_str = "\n".join(metrics_lines) if metrics_lines else "（无指标数据）"

    kb_blocks = []
    for i, h in enumerate(hits, 1):
        kb_blocks.append(
            f"[知识{i}｜来源:{h['source']}｜{h['title']}]\n{h['text']}"
        )
    kb_str = "\n\n".join(kb_blocks) if kb_blocks else "（未检索到相关知识）"

    return f"""你是一名量化研究员，请基于下方提供的【专业量价知识库片段】，对一个由算法挖掘出的选股因子做严谨、可追溯的可解释性分析。

# 待分析因子
表达式：
{row['expression']}

结构翻译：{translation}
启发式因子族判定：{'、'.join(families)}

# 该因子的真实回测指标（务必结合这些数据，不要臆造）
{metrics_str}

# 检索到的专业知识库片段（你的解释必须依据这些内容，并标注引用了哪条知识）
{kb_str}

# 输出要求
请用中文，分以下小节输出（每节简洁、专业）：
1. **经济学逻辑**：该因子捕捉了什么市场行为？为什么可能有效？（引用知识片段）
2. **所属因子族**：结合知识库判断它属于哪类经典因子，与哪个经典因子（如 WorldQuant AlphaXX）最相似，异同点是什么。
3. **有效性归因**：结合真实 IC/IR/Sharpe 指标，说明其选股能力的强弱与稳定性。
4. **潜在风险**：换手、过拟合、失效情形等。
要求：客观，不夸大；凡引用知识库内容请注明 [知识N]。"""


def analyze(df: pd.DataFrame, top_k: int, use_llm: bool) -> list[dict]:
    """对每个因子执行 翻译→检索→（可选）LLM 解释。"""
    results = []
    for idx, row in df.iterrows():
        expr = str(row["expression"])
        translation = describe_expression(expr)
        families = detect_factor_families(expr)
        query = build_query(expr, translation, families)

        print(f"[{idx + 1}/{len(df)}] 检索: {expr[:60]}...")
        hits = retrieve(query, top_k=top_k)

        explanation = ""
        if use_llm:
            prompt = build_prompt(row, translation, families, hits)
            try:
                from llm_client import query as llm_query
                explanation, _ = llm_query(prompt)
            except Exception as e:
                explanation = f"（LLM 调用失败，已跳过：{e}）"

        results.append(
            {
                "row": row,
                "translation": translation,
                "families": families,
                "hits": hits,
                "explanation": explanation,
            }
        )
    return results


def render_report(results: list[dict], use_llm: bool) -> str:
    """生成 Markdown 报告。"""
    lines = [
        "# Top 因子 RAG 可解释性分析报告",
        "",
        "> 本报告对挖掘出的高分因子，结合本地量价知识库（WorldQuant Alpha101 / "
        "算子语义 / 因子族范式）检索最相似知识，并由 LLM 生成可追溯的解释。",
        "",
        f"共分析 {len(results)} 个因子。",
        "",
    ]

    for i, r in enumerate(results, 1):
        row = r["row"]
        alpha_id = row.get("alpha_id", f"factor_{i}")
        lines.append(f"## {i}. {alpha_id}")
        lines.append("")
        lines.append("**因子表达式**")
        lines.append("```")
        lines.append(str(row["expression"]))
        lines.append("```")
        lines.append("")
        lines.append(f"**结构翻译**：{r['translation']}")
        lines.append("")
        lines.append(f"**启发式因子族**：{'、'.join(r['families'])}")
        lines.append("")

        # 指标卡
        metric_cells = []
        for col, label in METRIC_LABELS.items():
            if col in row and pd.notna(row[col]):
                try:
                    metric_cells.append(f"{label}={float(row[col]):.4f}")
                except (ValueError, TypeError):
                    pass
        if metric_cells:
            lines.append("**回测指标**：" + " ｜ ".join(metric_cells))
            lines.append("")

        # 检索命中
        lines.append("**检索命中的知识（可追溯来源）**")
        lines.append("")
        for j, h in enumerate(r["hits"], 1):
            snippet = h["text"].replace("\n", " ")
            if len(snippet) > 180:
                snippet = snippet[:180] + "…"
            lines.append(
                f"{j}. `[{h['source']} | {h['title']}]` (距离={h['distance']:.3f}) {snippet}"
            )
        lines.append("")

        if use_llm:
            lines.append("**LLM 解释**")
            lines.append("")
            lines.append(r["explanation"] or "（无）")
            lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Top-N 因子 RAG 可解释性分析")
    parser.add_argument(
        "--metrics",
        default=str(PROJ_ROOT / "results_adaptive_uct_linear" / "metrics" / "cycle_0001_*final_metrics.csv"),
        help="metrics CSV 路径（支持通配符）",
    )
    parser.add_argument("--topn", type=int, default=10, help="分析前 N 个因子")
    parser.add_argument("--top_k", type=int, default=4, help="每个因子检索的知识片段数")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "report.md"))
    parser.add_argument("--no-llm", action="store_true", help="跳过 LLM 解释，仅做翻译+检索")
    args = parser.parse_args()

    try:
        from config import DEEPSEEK_API_KEY
    except Exception:
        DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

    has_key = bool(DEEPSEEK_API_KEY)
    use_llm = not args.no_llm and has_key
    if not args.no_llm and not has_key:
        print("⚠️ 未检测到 DEEPSEEK_API_KEY，将跳过 LLM 解释（仅输出翻译+检索）。")

    df = load_topn(args.metrics, args.topn)
    print(f"已载入 Top-{len(df)} 因子。\n")

    results = analyze(df, top_k=args.top_k, use_llm=use_llm)
    report = render_report(results, use_llm=use_llm)

    Path(args.out).write_text(report, encoding="utf-8")
    print(f"\n报告已生成：{args.out}")


if __name__ == "__main__":
    main()
