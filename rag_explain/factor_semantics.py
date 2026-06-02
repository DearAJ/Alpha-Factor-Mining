"""
factor_semantics.py — 因子表达式语义翻译

职责：
1. OP_SEMANTICS  算子 → 自然语言含义字典（与 operators.py 对齐）
2. describe_expression()  把 DSL 表达式翻译为可读的中文描述
3. detect_factor_families()  根据表达式结构推断所属因子族

无外部依赖，可独立调用。
"""

import re

# 算子语义字典（key 为算子名，value 为简短中文含义）
# 与 operators.py 中 Operators 类的方法名对齐
OP_SEMANTICS = {
    "Mean": "N日滚动均值（价格中枢/平均量能）",
    "Std": "N日滚动标准差（波动率）",
    "Var": "N日滚动方差（波动率）",
    "Skew": "N日滚动偏度（分布非对称性）",
    "Kurt": "N日滚动峰度（厚尾/极端事件）",
    "Mad": "N日平均绝对偏离（稳健波动度量）",
    "Med": "N日滚动中位数",
    "Sum": "N日滚动求和",
    "Max": "N日滚动最大值",
    "Min": "N日滚动最小值",
    "Quantile": "N日滚动分位数",
    "Count": "N日非空计数",
    "Delta": "N日变化量（动量/涨跌）",
    "Ref": "N日前的取值（时间平移）",
    "Slope": "对时间滚动回归的斜率（趋势强度）",
    "Rsquare": "对时间滚动回归的R方（趋势质量）",
    "Resi": "对时间滚动回归的残差（短期背离）",
    "EMA": "指数加权移动平均（近期更敏感）",
    "WMA": "加权移动平均",
    "Rank": "排名归一化（横截面相对位置）",
    "Corr": "两变量N日滚动相关性（量价关系）",
    "Cov": "两变量N日滚动协方差",
    "Abs": "绝对值",
    "Sign": "符号函数",
    "Log": "自然对数（压缩右偏分布）",
    "Power": "幂运算",
    "Greater": "取较大值",
    "Less": "取较小值",
    "IdxMax": "N日最大值位置",
    "IdxMin": "N日最小值位置",
}

# 字段含义
FIELD_SEMANTICS = {
    "close": "收盘价",
    "open": "开盘价",
    "high": "最高价",
    "low": "最低价",
    "volume": "成交量",
    "amount": "成交额",
    "vwap": "成交量加权均价",
}


def used_operators(expression: str) -> list[str]:
    """返回表达式中用到的算子名（按出现顺序去重）。"""
    found = []
    for op in OP_SEMANTICS:
        if re.search(r"\b" + re.escape(op) + r"\s*\(", expression):
            if op not in found:
                found.append(op)
    return found


def used_fields(expression: str) -> list[str]:
    """返回表达式中用到的量价字段。"""
    found = []
    for f in FIELD_SEMANTICS:
        if re.search(r"\b" + re.escape(f) + r"\b", expression):
            found.append(f)
    return found


def describe_expression(expression: str) -> str:
    """
    把因子表达式翻译为一句中文摘要，列出涉及的算子与字段，并标注方向。
    用于构造检索 query 与 LLM prompt。
    """
    ops = used_operators(expression)
    fields = used_fields(expression)

    op_parts = [f"{op}（{OP_SEMANTICS[op]}）" for op in ops]
    field_parts = [f"{f}（{FIELD_SEMANTICS[f]}）" for f in fields]

    direction = ""
    # 末尾 * -1 或 -1 * 开头 → 反向信号
    if re.search(r"\*\s*-1\s*$", expression.strip()) or re.search(r"^\s*-1\s*\*", expression.strip()):
        direction = "整体取负（反向信号，多为反转/风险规避逻辑）。"

    desc = "该因子使用的算子：" + "，".join(op_parts) + "。"
    if field_parts:
        desc += "涉及量价字段：" + "，".join(field_parts) + "。"
    if direction:
        desc += direction
    return desc


# 因子族识别规则：(族名, 判定函数)
def detect_factor_families(expression: str) -> list[str]:
    """
    根据表达式结构启发式推断所属因子族，返回族名列表（可能多个）。
    """
    expr = expression
    families = []

    has_neg = bool(re.search(r"\*\s*-1", expr) or re.search(r"-1\s*\*", expr))

    # 量价背离：含 Corr 且表达式同时涉及价格与量字段
    if re.search(r"Corr\s*\(", expr):
        price_in = any(re.search(rf"\b{p}\b", expr) for p in ["close", "open", "high", "low", "vwap"])
        vol_in = any(re.search(rf"\b{v}\b", expr) for v in ["volume", "amount"])
        if price_in and vol_in:
            families.append("量价背离/量价相关")
        else:
            families.append("相关性结构")

    # 波动率
    if re.search(r"Std\s*\(\s*(close|return|vwap)", expr) or re.search(r"\bMad\s*\(", expr):
        families.append("波动率/低波动")

    # 动量 vs 反转：Delta(close) 主导
    if re.search(r"Delta\s*\(\s*(close|vwap)", expr):
        families.append("反转" if has_neg else "动量")

    # 流动性/换手
    if re.search(r"(Mean|Rank|Sum)\s*\(\s*(volume|amount)", expr):
        families.append("流动性/换手率")

    # 趋势质量
    if re.search(r"(Slope|Rsquare|Resi)\s*\(", expr):
        families.append("趋势质量")

    if not families:
        families.append("综合量价")
    # 去重保持顺序
    seen = set()
    return [x for x in families if not (x in seen or seen.add(x))]


if __name__ == "__main__":
    import sys

    demo = sys.argv[1] if len(sys.argv) > 1 else (
        "Rank(Corr(Delta(close, 5), Delta(volume, 5), 10) * "
        "(1 - Abs(Delta(close, 1) / close)) * (1 - Rank(Mean(volume, 20)) / 2) * "
        "(1 - Rank(Std(close, 10)))) * -1"
    )
    print("表达式:", demo)
    print("翻译:", describe_expression(demo))
    print("因子族:", detect_factor_families(demo))
