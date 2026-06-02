# Top-N 因子 RAG 可解释性分析

对挖掘出的高分因子，结合**本地量价知识库**（WorldQuant Alpha101 / 算子语义 / 因子族范式）
做基于 RAG 的可解释性分析：检索每个因子最相似的专业知识，再由 LLM 生成**可追溯**的解释报告。

## 流程

```
knowledge/*.md ──切分──► 本地 embedding ──► Chroma 向量库
                                              │
Top-N 因子 CSV ──表达式翻译+指标──► 检索 top-k ──► LLM(DeepSeek) ──► report.md
```

## 文件

| 文件 | 职责 |
|------|------|
| `knowledge/*.md` | 知识库语料（算子语义 / 因子族 / Alpha101 精选） |
| `factor_semantics.py` | DSL 表达式 → 中文翻译 + 因子族识别（无依赖） |
| `retriever.py` | 本地 embedding + Chroma 检索封装 |
| `build_kb.py` | 切分语料 → 写入向量库 |
| `explain.py` | 主脚本：读 Top-N → 检索 → LLM → 出报告 |

## 安装

```bash
pip install -r rag_explain/requirements.txt
```

首次运行 `build_kb.py` 会自动下载中文 embedding 模型 `BAAI/bge-small-zh-v1.5`（约 100MB）。
下载完成后可设 `HF_HUB_OFFLINE=1` 离线运行。

## 使用

```bash
cd rag_explain

# 1. 构建知识库向量库（只需一次，语料更新后重跑）
python build_kb.py

# 2. 冒烟测试检索
python retriever.py "量价背离反转因子"

# 3. 生成 Top-10 可解释性报告
export DEEPSEEK_API_KEY=你的key        # 不设则跳过 LLM，仅输出翻译+检索
python explain.py \
    --metrics ../results_adaptive_uct_linear/metrics/cycle_0001_*final_metrics.csv \
    --topn 10 \
    --out report.md
```

### explain.py 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--metrics` | cycle_0001 | metrics CSV 路径（支持通配符，可匹配多文件合并去重） |
| `--topn` | 10 | 取打分（mcts_reward→ic_ir→ic_mean）前 N 个因子 |
| `--top_k` | 4 | 每个因子检索的知识片段数 |
| `--out` | report.md | 报告输出路径 |
| `--no-llm` | 关 | 仅翻译+检索，不调用 LLM |

## 扩展：投喂真实研报

把整理好的研报文本（md，按 `## ` 二级标题分段）放入 `knowledge/` 后重跑 `build_kb.py` 即可。
若是 PDF，可先转为 md/txt（后续可加 `pypdf` 解析脚本）。
