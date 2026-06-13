"""
explain.py — Top-N 因子 RAG 可解释性分析主脚本

流程：
1. 读取因子 metrics CSV，按 long_excret（或备选列）排序取 Top-N
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
RANK_COLUMNS = ["long_excret", "ic_ir", "ic_mean"]
RANK_LABELS = {
    "long_excret": "多头年化超额",
    "ic_ir": "IC_IR",
    "ic_mean": "IC均值",
}

# 报告中展示的指标列 → 中文名
METRIC_LABELS = {
    "ic_mean": "IC均值",
    "ic_ir": "IC_IR",
    "ict": "IC t值",
    "long_excret": "多头年化超额",
    "long_sharpe": "多头Sharpe",
    "long_ir": "多头IR",
    "ls_sharpe": "多空Sharpe",
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
    top_df = df.head(topn).reset_index(drop=True)
    top_df.attrs["rank_col"] = rank_col or ""
    return top_df


def factor_direction_note(row: pd.Series) -> str:
    """生成 IC 方向提示，避免 LLM 把负 IC 与多头收益关系说反。"""
    if "ic_mean" not in row or pd.isna(row["ic_mean"]):
        return "IC方向：CSV 未提供 ic_mean，不能判断原始因子值与未来收益的方向关系。"

    ic_mean = float(row["ic_mean"])
    if ic_mean > 0:
        return "IC方向：ic_mean 为正，表示样本内原始因子值越高，未来收益倾向越高。"
    if ic_mean < 0:
        return "IC方向：ic_mean 为负，表示样本内原始因子值越高，未来收益倾向越低；若多头指标为正，通常意味着多头端可能来自低因子值一侧，需以评估器分组方向为准。"
    return "IC方向：ic_mean 接近 0，不能据此判断原始因子值与未来收益存在稳定方向关系。"


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
    direction_note = factor_direction_note(row)

    kb_blocks = []
    for i, h in enumerate(hits, 1):
        kb_blocks.append(
            f"[知识{i}｜来源:{h['source']}｜{h['title']}]\n{h['text']}"
        )
    kb_str = "\n\n".join(kb_blocks) if kb_blocks else "（未检索到相关知识）"

    return f"""你是一名严谨的量化研究员。请**仅依据**下方提供的【检索到的知识库片段】和【真实回测指标】，对一个由算法挖掘出的选股因子做可追溯的可解释性分析。

# 待分析因子
表达式：
{row['expression']}

结构翻译：{translation}
启发式因子族判定（仅供参考，可被知识库推翻）：{'、'.join(families)}

# 该因子的真实回测指标（这些是样本内、未扣交易成本的结果）
{metrics_str}

{direction_note}

# 检索到的专业知识库片段（你唯一可引用的外部依据）
{kb_str}

# 铁律（违反即视为错误输出）
1. **严禁臆造**：不得提及任何未在上方知识库片段中出现的经典因子编号或公式（例如不得凭记忆写出某个 WorldQuant Alpha 的公式）。只有当某个 Alpha 的**公式确实出现在上方片段里**时，才可引用它，且必须照抄片段中的公式，不得改写。
2. **无对应就明说**：若检索片段中没有与本因子可对照的经典因子，直接写"知识库中无直接对应的经典因子"，不要勉强攀附。
3. **引用标注**：凡使用知识库内容，必须标注 [知识N]（N 为上方片段编号）。
4. **指标口径统一**：|IC|≥0.03 为有一定选股能力；|IC_IR|>0.5 为优秀、0.3~0.5 为中等、<0.3 为偏弱。所有 Sharpe/IC_IR 均为样本内、未扣成本，解读时必须点明这一点，不得渲染为"实盘必然如此"。
5. **方向一致**：解释 IC 时必须区分"原始因子值高低"与"多头组合收益"。IC 为负时，不得写成"因子值越高未来收益越高"；若要说明多头收益，只能说"多头端可能来自低因子值一侧，需以评估器分组方向为准"。
6. **证据边界**：没有分项贡献、消融实验或分组收益时，不得断言某一项"主导"因子表现，只能使用"可能、倾向于、需要进一步验证"等保守表述。
7. **客观克制**：不夸大；避免使用"极强、极其、惊人、几乎不可能、必然、可靠识别"等绝对化措辞。若指标与逻辑冲突（如号称强势却 IC 为负），如实指出矛盾，不要强行自圆其说。

# 输出要求（中文，分小节，简洁专业）
1. **经济学逻辑**：拆解表达式各部分捕捉的市场行为及可能有效的原因（引用 [知识N]）。
2. **所属因子族**：依据知识库判断属于哪类因子；仅在公式确实出现在片段中时才做经典因子对照，否则按铁律 2 处理。
3. **有效性归因**：结合真实 IC/IC_IR/Sharpe（按铁律 4 和 5 的口径与样本内/未扣成本前提）说明选股能力强弱、方向关系与稳定性，语气保持审慎。
4. **潜在风险**：换手、过拟合、失效情形等。"""


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
    rank_col = ""
    if results:
        rank_col = results[0].get("rank_col", "")
    rank_label = RANK_LABELS.get(rank_col, rank_col) if rank_col else "未指定排序列"

    lines = [
        "# Top 因子 RAG 可解释性分析报告",
        "",
        "> 本报告对挖掘出的高分因子，结合本地量价知识库（WorldQuant Alpha101 / "
        "算子语义 / 因子族范式）检索最相似知识，并由 LLM 生成可追溯的解释。",
        "",
        f"> Top-N 排序口径：按 `{rank_col or 'N/A'}`（{rank_label}）从高到低排序；若该列缺失，按代码中的备选列顺延。",
        "",
        "> 指标说明：所有回测指标均为样本内、未扣交易成本结果；IC 方向代表原始因子值与未来收益的相关方向，不等同于实盘结论。",
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
        lines.append(f"**IC方向提示**：{factor_direction_note(row)}")
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
    rank_col = df.attrs.get("rank_col", "")
    print(f"已载入 Top-{len(df)} 因子。\n")

    results = analyze(df, top_k=args.top_k, use_llm=use_llm)
    for result in results:
        result["rank_col"] = rank_col
    report = render_report(results, use_llm=use_llm)

    Path(args.out).write_text(report, encoding="utf-8")
    print(f"\n报告已生成：{args.out}")


if __name__ == "__main__":
    main()
