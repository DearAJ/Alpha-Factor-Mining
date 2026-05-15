# MCTS 搜索树可视化方案

## 一、背景与动机

### 1.1 问题描述

AlphaJungleMCTS 使用蒙特卡洛树搜索（MCTS）在因子表达式空间中搜索高质量 alpha 因子。在搜索过程中：

1. **根节点**（初始表达式）通过 LLM 生成一个种子因子
2. 每步沿 5 个维度（effectiveness、diversity、stability、turnover、overfit）中评分最低的方向调用 LLM 生成改进版本
3. 持续 24 步，形成一棵搜索树

但此前整个搜索过程是**黑箱**——只有日志里的文本输出，无法直观看到：
- 表达式从根节点到叶节点的逐步演变过程
- 每个节点在 5 个维度上的评分分布
- 搜索树的结构（哪些分支深入了，哪些被放弃了）
- LLM 经常生成的语义等价表达式（`A+B` 和 `B+A` 被视为不同节点）无法被察觉

### 1.2 PPT 汇报需求

在项目汇报的场景下，需要向评审者展示：
- **树形图**：表达式如何一步步演变（根 → 改进 → 再改进）
- **节点详情**：每个表达式长什么样、评分如何
- **搜索效率**：是否存在冗余搜索（语义等价节点）

### 1.3 设计目标

| 目标 | 优先级 | 说明 |
|------|--------|------|
| 交互式可视化 | P0 | 可悬停查看详情，可缩放，适合 PPT 截图或分享 HTML |
| 语义等价检测 | P0 | 识别 A+B vs B+A 类冗余，量化搜索浪费 |
| 5 维评分展示 | P0 | 雷达图直观对比各节点维度优劣 |
| 演化路径追踪 | P1 | 表格形式展示某条路径的逐步改进过程 |
| 独立可分享 | P0 | 单个 HTML 文件，不依赖服务器 |


---

## 二、设计思路

### 2.1 架构选型

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| Plotly 交互式 HTML | 自包含、可悬停、支持雷达图 | 需安装 plotly | **采用** |
| Matplotlib 静态图 | 无需额外安装 | 无交互、布局难看 | 放弃 |
| D3.js 网页 | 最灵活 | 需前端开发、无法嵌入 Python | 放弃 |
| Graphviz | 树布局好 | 节点内容展示有限 | 放弃 |

### 2.2 数据流设计

```
MCTSSearchNode (mining_methods.py)
     │
     │ to_dict()
     ▼
纯 Python dict（JSON 可序列化）
     │
     ▼
SearchTreeVisualizer (tree_viz.py)
     │
     ├─ flatten_tree()      DFS 拍平为节点/边列表
     ├─ compute_tree_layout()  自顶向下布局坐标
     ├─ expression_signature() AST 归一化签名
     ├─ find_equivalent_groups() 语义等价检测
     ├─ build_tree_figure()     Plotly 树形图
     ├─ build_evolution_table() 演化路径表格
     └─ create_full_report()   合并为 HTML
```

---

## 三、详细设计

### 3.1 涉及文件

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `mining_methods.py` | 新增 45 行 | `MCTSSearchNode.to_dict()` + 保存 `_search_tree_root` |
| `tree_viz.py` | **新建** | 可视化核心模块，~810 行 |
| `run.py` | 新增 14 行 | 可视化集成 block（可选执行，不影响非 MCTS 方法）|
| `viz_output/` | 新建目录 | 输出目录 |

### 3.2 mining_methods.py 改动

#### 3.2.1 新增 `MCTSSearchNode.to_dict()` 方法

将整个搜索树递归序列化为纯 Python dict，跳过 `factor_values`（大 DataFrame）以避免序列化问题：

```python
def to_dict(self) -> dict:
    def _convert(v):
        if isinstance(v, (float, np.floating)):
            return None if np.isnan(v) or np.isinf(v) else float(v)
        if isinstance(v, np.integer):
            return int(v)
        return v
    # ... 递归序列化 candidate、dimension_scores、children ...
    return {
        "node_id": self.node_id,
        "depth": self.depth,
        "reward": _convert(self.reward),
        "candidate": { ... },        # expression, metrics 等
        "dimension_scores": { ... },  # 5 维评分
        "parent_id": ...,
        "children": [child.to_dict() for child in self.children],
    }
```

#### 3.2.2 保存搜索树根节点

在 `mine_in_sample()` 搜索循环结束后、排名筛选前，保存根节点：

```python
self._search_tree_root = root
```

### 3.3 tree_viz.py 模块结构

```
tree_viz.py
├── NumpyEncoder                    JSON 序列化（处理 np.nan/np.inf）
├── expression_signature(expr)      AST 规范签名（交换律归一化）
├── find_equivalent_groups(nodes)   分组语义等价节点
├── flatten_tree(tree_dict)         DFS 拍平为节点/边列表
├── compute_tree_layout(nodes, edges) 自顶向下树布局
├── build_tree_figure(nodes, edges, pos, equiv)  Plotly 树形图
├── build_evolution_table(path)     Plotly 演化路径表格
├── create_full_report(tree_dict)   合并 HTML
└── SearchTreeVisualizer
    ├── from_mcts_method(method)    ← 唯一推荐入口
    ├── from_json(path)
    ├── save_html(path)
    ├── save_json(path)
    └── get_stats()
```

### 3.4 语义等价检测原理

使用 Python 的 `ast` 模块解析表达式语法树，对交换律操作符（`+` 和 `*`）的操作数排序后生成规范签名：

```
Rank(A) + Mean(B)  →  ("BinOp", "Add", ("Call", "rank", ...), ("Call", "mean", ...))
Mean(B) + Rank(A)  →  ("BinOp", "Add", ("Call", "rank", ...), ("Call", "mean", ...))  ← 排序后相同
```

相同签名的节点被标记为语义等价，在图中用菱形 ◇ 高亮。

### 3.5 树布局算法

自顶向下递归分配坐标：

```
1. 根节点居中于 (0.5, 1.0)
2. 每层下移 y_spacing=0.13
3. 水平空间按子树叶节点数比例分配
   ┌──────────────────────────┐
   │         root             │  y=1.0
   │      ┌──┴──┐             │
   │      A      B            │  y=0.87
   │    ┌─┴─┐  ┌┴┐           │
   │    C   D  E F            │  y=0.74
   └──────────────────────────┘
```

### 3.6 run.py 集成

在 `run_single_cycle()` 的 `mine_in_sample()` 调用后加入可选 block：

```python
if hasattr(self.mining_method, "_search_tree_root"):
    from tree_viz import SearchTreeVisualizer
    viz = SearchTreeVisualizer.from_mcts_method(self.mining_method)
    viz.save_html("viz_output/mcts_search_tree.html")
    viz.save_json("viz_output/mcts_tree.json")
```

- 仅 `alpha_jungle_mcts` 方法有 `_search_tree_root`，`factor_mad` / `alpha_agent` 自动跳过
- 异常不会影响主流程（`try/except` 包裹）
- 未安装 `plotly` 时跳过，不报错

---

## 四、效果展示

### 4.1 树形图

| 元素 | 展示内容 |
|------|---------|
| 节点 | 表达式摘要 + Reward 值 |
| 节点颜色 | RdYlGn 色阶（绿 = 高 Reward，红 = 低 Reward） |
| 节点大小 | 根节点最大，随深度递减 |
| 边 | 父子节点连线 |
| 边标签 | 改进维度（如 "↑ 有效性"、"↑ 多样性"） |
| 菱形节点 | ◇ 标记语义等价表达式 |

**悬停（hover）详情**：
```
node_001
深度: 1

表达式: Rank(Delta(close, 5)) * Rank(Std(close, 10))

综合 Reward: 0.8200
Q-Value: 0.8200
访问次数: 5

目标维度: effectiveness
改进建议: 加入波动率过滤以提高 IC

5 维评分:
  有效性 (effectiveness): 0.880
  多样性 (diversity): 0.420
  稳定性 (stability): 0.720
  换手率 (turnover): 0.650
  过拟合 (overfit): 0.520

IC(5) mean: 0.045
IC IR: 0.700
```

### 4.2 雷达图网格

每个节点一个 5 维雷达图，4 列网格排列，轴范围统一 [0, 1]，方便横向对比各节点的维度优劣势。

### 4.3 演化路径表格

自动提取根→叶的前 5 条最长路径，以表格展示每步的表达式变化：

| 深度 | 节点 ID | 表达式 | Reward | 改进维度 | 改进说明 |
|------|---------|--------|--------|---------|---------|
| 0 | root | Rank(Delta(close, 5)) | 0.72 | — | 初始种子 |
| 1 | n001 | Rank(Delta(close,5)) * Rank(Std(close,10)) | 0.85 | 有效性 | 加入波动率过滤 |
| 2 | n002 | ... * Rank(Volume,5) | 0.90 | 稳定性 | 加入成交量确认 |

### 4.4 语义等价检测报告

页面底部列出所有等价表达式组，量化搜索浪费：

```
⚠️ 检测到 2 组语义等价的表达式（共涉及 4 个节点）

等价组 1:
  - node_004: Rank(A) + Mean(B)
  - node_012: Mean(B) + Rank(A)

等价组 2:
  - node_007: Rank(close) * Std(volume, 10)
  - node_019: Std(volume, 10) * Rank(close)
```

### 4.5 输出文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `viz_output/mcts_search_tree.html` | ~3.6 MB | 自包含交互式 HTML（含 Plotly.js） |
| `viz_output/mcts_tree.json` | ~2 KB | 树数据 JSON（可回读、分享） |

---

## 五、如何使用

### 5.1 使用流程

> **前置条件：需要安装 plotly**
> ```bash
> pip install plotly
> ```
> 如果没装，可视化 block 会被 `try/except` 捕获跳过，不影响因子挖掘主流程，但不会输出 HTML 报告。

**其他什么都不用做 — `run.py` 已预置集成代码。**

跑 MCTS 时（`--method_config` 指向 `alpha_jungle_mcts.yaml`），可视化会自动触发并输出到 `viz_output/`：

```bash
python run.py --method_config ./method_config/alpha_jungle_mcts.yaml
```

跑完后 `viz_output/` 目录下会出现：
- `mcts_search_tree.html` — 浏览器直接打开的交互式报告
- `mcts_tree.json` — 树数据（可选，用于后续分析）

> 如果跑的是 `factor_mad` 或 `alpha_agent` 等其他方法，可视化 block 通过 `hasattr` 判断 `_search_tree_root` 不存在，自动跳过，**完全不影响你的流程**。



### 5.2 手动调用可视化（三行代码）

```python
from tree_viz import SearchTreeVisualizer

viz = SearchTreeVisualizer.from_mcts_method(self.mining_method)
viz.save_html("viz_output/tree.html")
```

适合场景：
- 想看一下加了可解释性字段后的 hover 效果
- 想确认多 reward 体系下的节点颜色分布
- 只想在特定轮次生成报告，不想每次都输出


### 5.3 注意事项

| 事项 | 说明 |
|------|------|
| **不要删除 `tree_viz.py`** | 删掉后 `run.py` 的 `from tree_viz import SearchTreeVisualizer` 会报 `ModuleNotFoundError`，但被 `try/except` 捕获，不影响主流程 |
| **不要删除 `viz_output/` 目录** | `save_html/save_json` 会自动创建，删了也没关系 |
| **不要改 `mining_methods.py` 中的 `to_dict()`** | 其他组员都不会用到这个方法，改了可能影响 JSON 输出格式 |
| HTML 文件过大 | 24 节点约 3.6 MB（含 Plotly.js 库），如果觉得大可删掉 `include_plotlyjs=True` 改为 CDN 加载 |


---

## 六、总结

本改进为 AlphaJungleMCTS 搜索过程提供了完整的可视化支持，核心成果：

1. **搜索过程透明化**：从黑箱到直观的交互式树形图
2. **语义等价检测**：自动识别 `A+B` vs `B+A` 类冗余，量化搜索浪费
3. **5 维评分可视化**：雷达图网格 + hover 详情，快速定位瓶颈维度
4. **演化路径追踪**：表格展示表达式从根到叶的每步改进


总计新增代码：`tree_viz.py` ~810 行 + `mining_methods.py` 45 行 + `run.py` 14 行。
