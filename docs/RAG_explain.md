# Top-N 因子 RAG 可解释性分析

对挖掘出的高分因子，结合**本地量价知识库**（WorldQuant Alpha101 / 算子语义 / 因子族范式）
做基于 RAG 的可解释性分析：检索每个因子最相似的专业知识，再由 LLM 生成**可追溯**的解释报告。

## 语料库

1. WorldQuant Alpha101：101 个经典量价 alpha 的公式与含义，来源于公开论文 *101 Formulaic Alphas*（Kakushadze, 2016）。每个 alpha 一段，标注用到的量价结构，例如动量、反转、量价背离、波动率等。
2. 算子语义说明：把本项目 `operators.py` 中的算子（如 `Corr`、`Delta`、`Rank`、`Std`、`Mean`、`Ts_Regression`）整理成金融含义知识条目，例如 `Corr(price, volume)` 衡量量价同步性。
3. 经典因子范式：整理动量、反转、波动率、流动性、量价背离、换手率等因子族的逻辑与适用场景。

## 流程

```text
knowledge/*.md ──切分──► 本地 embedding ──► Chroma 向量库
                                              │
Top-N 因子 CSV ──表达式翻译+指标──► 检索 top-k ──► LLM(DeepSeek) ──► report.md
```

1. Query 构造（`factor_semantics.py` + `explain.py`）：对每个因子，把表达式翻译成自然语言描述，并拼接成检索 query。

   示例：

   ```text
   Rank(Corr(Delta(close,5),Delta(volume,5),10))*-1
   → 5日价格变化与5日成交量变化的10日滚动相关性的时序滚动排名，取负 → 量价背离反转信号
   ```

   注：本系统 `Rank(x, N)` 为时序滚动排名（`df.rolling(N).rank()`），即每只股票在自身历史窗口内排名，并非横截面排名。

2. 检索（`retriever.py`）：使用 Chroma query 取 `top_k=3~5` 个最相似知识片段。

3. LLM 生成（`explain.py` → `llm_client.query()`）：prompt 模板包含因子表达式、自然语言翻译、真实指标、检索到的知识片段，以及解释要求。为避免臆造，prompt 设有铁律：只能引用检索片段中出现的经典因子公式，无对应时须明说"知识库中无直接对应"，并统一指标口径、声明结果为样本内未扣成本。

4. 报告产出：生成 Markdown 报告。每个因子一节，包含表达式、结构翻译、指标卡、检索命中的知识来源和 LLM 解释。

## 文件

| 文件 | 职责 |
|------|------|
| `knowledge/*.md` | 知识库语料（算子语义 / 因子族 / Alpha101 精选） |
| `factor_semantics.py` | DSL 表达式 → 中文翻译 + 因子族识别（无依赖） |
| `retriever.py` | 本地 embedding + Chroma 检索封装 |
| `build_kb.py` | 切分语料 → 写入向量库 |
| `explain.py` | 主脚本：读 Top-N → 检索 → LLM → 出报告 |

## 安装

在项目根目录执行：

```bash
pip install -r rag_explain/requirements.txt
```

本地 embedding 默认在 CPU 上运行；如需指定 GPU，可设置：

```bash
export RAG_EMBED_DEVICE=cuda:0
```

## 使用

```bash
cd rag_explain

# 1. 构建知识库向量库（只需一次，语料更新后重跑）
python build_kb.py

# 2. 冒烟测试检索
python retriever.py "量价背离反转因子"

# 3. 生成 Top-10 可解释性报告
export DEEPSEEK_API_KEY=你的key  # 也可在项目根目录 config.py 中配置
python explain.py \
     --metrics "../results_adaptive_uct_linear/metrics/cycle_*_final_metrics.csv" \
    --topn 10 \
    --out report.md
```

如果不想调用 LLM，只生成表达式翻译和本地检索结果，可加 `--no-llm`。

注意：`--metrics` 支持通配符，但建议用引号包起来，避免 shell 提前展开成多个参数。

### explain.py 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--metrics` | `cycle_0001_*final_metrics.csv` | metrics CSV 路径，支持通配符，可匹配多文件合并去重 |
| `--topn` | `10` | 取打分前 N 个因子，排序优先级为 `mcts_reward` → `ic_ir` → `ic_mean` |
| `--top_k` | 4 | 每个因子检索的知识片段数 |
| `--out` | `report.md` | 报告输出路径 |
| `--no-llm` | 关 | 仅翻译+检索，不调用 LLM |

## 常见命令

分析所有 cycle 的 final metrics：

```bash
python explain.py \
     --metrics "../results_adaptive_uct_linear/metrics/cycle_*_final_metrics.csv" \
     --topn 10 \
     --out report.md
```

只分析 `cycle_0001`：

```bash
python explain.py \
     --metrics "../results_adaptive_uct_linear/metrics/cycle_0001_*final_metrics.csv" \
     --topn 10 \
     --out report.md
```


