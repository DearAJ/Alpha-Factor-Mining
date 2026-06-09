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
| `tree_viz.py` | **新建** | 可视化核心模块，~870 行（含静态树、动态展开动画、节点交替标签） |
| `tree_compare.py` | **新建** | 跨轮次对比分析模块，~210 行 |
| `run.py` | 新增 26 行 | 可视化集成 block + 逐轮保存历史树 + 跑完后自动输出对比报告 |
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
├── _assign_expand_steps(nodes, edges) BFS 分配动画步序号
├── build_tree_figure(nodes, edges, pos, equiv)  Plotly 树形图（静态）
├── build_animated_tree_figure(...)     Plotly 树形图（动态展开动画）
├── build_evolution_table(path)     Plotly 演化路径表格
├── create_full_report(tree_dict)   合并 HTML
└── SearchTreeVisualizer
    ├── from_mcts_method(method)    ← 唯一推荐入口
    ├── from_json(path)
    ├── save_html(path)             静态报告（树 + 雷达图 + 表格）
    ├── save_animated_html(path)    动态展开动画（树逐步生长）
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
    # 保存当前轮次的历史树（用于跨轮次对比）
    cycle_tag = f"cycle_{cycle_count:04d}"
    viz.save_html(f"viz_output/{cycle_tag}_tree.html")
    viz.save_json(f"viz_output/{cycle_tag}_tree.json")
    # 同时覆盖保存最新树（方便快速查看）
    viz.save_html("viz_output/mcts_search_tree.html")
    viz.save_json("viz_output/mcts_tree.json")
```

- 仅 `alpha_jungle_mcts` 方法有 `_search_tree_root`，`factor_mad` / `alpha_agent` 自动跳过
- 异常不会影响主流程（`try/except` 包裹）
- 未安装 `plotly` 时跳过，不报错
- 多轮运行结束后，自动调用 `tree_compare.py` 生成跨轮次对比报告

### 3.7 tree_compare.py 跨轮次对比设计

#### 3.7.1 动机

单棵树可视化可以展示"一次搜索的内部结构"，但无法回答以下问题：

- **趋势判断**：第 5 轮找到的因子比第 1 轮更好吗？Reward 在提升还是波动？
- **搜索效率**：随着轮次增加，搜索树是扩大还是收敛？语义等价节点在增多还是减少？
- **因子演化**：最佳因子的表达式和维度评分在轮次间如何变化？

#### 3.7.2 数据流

```
run.py 多轮运行
     │
     │ 每轮保存 cycle_{编号}_tree.json
     ▼
viz_output/
  ├── cycle_0001_tree.json
  ├── cycle_0002_tree.json
  ├── cycle_0003_tree.json    ← 加载后分析对比
  └── ...
     │
     ▼
tree_compare.py
  ├─ load_cycle_trees()       批量加载并排序
  ├─ build_metrics_trend()     2×2 趋势面板
  ├─ build_best_factor_table() 最佳因子对比表
  ├─ build_dimension_radar_overlay() 5维雷达叠加
  └─ create_comparison_report() 合并为 HTML
     │
     ▼
viz_output/comparison_report.html
```

#### 3.7.3 对比维度

| 图表 | 对比内容 | 说明 |
|------|---------|------|
| Best Reward 折线 | 每轮最佳因子 Rewad | 判断因子质量是否逐轮提升 |
| Avg Reward 折线 | 每轮节点平均 Reward | 判断搜索整体质量趋势 |
| 节点数柱状 | 每轮搜索树规模 | 判断搜索是否在扩大/收敛 |
| 等价节点柱状 | 每轮语义等价节点数 | 判断搜索浪费是否在减少 |
| 5维雷达叠加 | 每轮最佳因子评分分布 | 判断搜索焦点是否偏移 |
| 最佳因子对比表 | 表达式 + 指标 | 每轮最终选出了什么 |

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

### 4.5 动态展开动画（新增）

相比静态树的"一次性展示最终状态"，动态动画展示的是**搜索树逐步生长的过程**，更接近 MCTS 正在搜索的观感。

**交互方式**：打开 HTML 后，点击右上角的 ▶ Play 按钮，树从根节点开始逐帧扩展，每 0.6 秒新增一个节点。

| 元素 | 说明 |
|------|------|
| ▶ Play / ⏸ Pause | 开始 / 暂停动画 |
| ⟲ Reset | 回到第 1 步（仅根节点） |
| 步进滑块 | 拖拽跳转到任意搜索步 |
| 红色边框节点 | 当前帧新展开的节点 |
| 节点标签 | Reward + 改进维度（完整表达式见 hover） |

**搜索顺序**：按 BFS 展开，同层子节点按 Reward 从高到低排列，模拟 MCTS 优先扩展高价值分支的行为。

```
Step 1:        Step 5:               Step 23:
● root         ● root                ● root
               ├── ● A               ├── ● A
               ├── ● C               │    ├── ● A1
               └── ● B               │    │    ├── ● A1a
                                      │    │    └── ● A1b
                                      │    ├── ● A2
                                      │    └── ● A3
                                      ├── ● C
                                      │    ├── ● C1
                                      │    └── ● C2
                                      └── ● B
                                           ├── ● B1
                                           └── ● B2
```

### 4.6 输出文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `viz_output/mcts_search_tree.html` | ~3.6 MB | 静态交互式报告（树 + 雷达图 + 演化表格 + 等价检测） |
| `viz_output/mcts_tree_animated.html` | ~3.8 MB | 动态展开动画（树逐步生长，▶ Play 播放） |
| `viz_output/mcts_tree.json` | ~2 KB | 树数据 JSON（覆盖式，始终为最新一轮） |
| `viz_output/cycle_0001_tree.json` | ~2 KB | 第 1 轮历史树 JSON（用于跨轮次对比） |
| `viz_output/cycle_0002_tree.json` | ~2 KB | 第 2 轮历史树 JSON（用于跨轮次对比） |
| `viz_output/comparison_report.html` | ~3.5 MB | 跨轮次对比报告（趋势图 + 对比表 + 雷达叠加） |

### 4.7 跨轮次对比报告

多轮运行结束后自动生成，包含三大模块：

**模块一：指标趋势面板（2×2）**

展示 Best Reward、Avg Reward、节点数、语义等价节点数在轮次间的变化趋势，快速判断搜索是否在进步。

```
Best Reward            Avg Reward
 1.0 ───●───●───     0.9 ───●───●───
 0.9    ●             0.8    ●
    1   2   3   4         1   2   3   4

节点数                 语义等价节点
 10 ┌──┐               4 ┌──┐
  5 ┼──┼──┐            2 ┼──┼──┐
    1  2  3  4           1  2  3  4
```

**模块二：5 维评分雷达叠加**

将各轮最佳因子的 dimension_scores 叠加在同一张雷达图上，直接对比搜索焦点偏移。

**模块三：最佳因子对比表**

| 轮次 | 最佳表达式 | Reward | 有效性 | 多样性 | 稳定性 | 换手率 | 过拟合 | IC IR |
|------|-----------|--------|--------|--------|--------|--------|--------|-------|
| 1 | Rank(Delta(close, 5)) | 0.72 | 0.65 | 0.40 | 0.55 | 0.70 | 0.60 | 0.58 |
| 2 | Rank(...) * Rank(...) | 0.85 | 0.88 | 0.42 | 0.72 | 0.65 | 0.52 | 0.70 |

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
# 单轮运行（输出最新树可视化）
python run.py --method_config ./method_config/alpha_jungle_mcts.yaml

# 多轮运行（输出每轮历史树 + 跑完后自动生成跨轮次对比报告）
python run.py --method_config ./method_config/alpha_jungle_mcts.yaml --max_cycles 5
```

跑完后 `viz_output/` 目录下会出现：
- `mcts_search_tree.html` — 浏览器直接打开的交互式报告（最新一轮）
- `mcts_tree.json` — 树数据 JSON（最新一轮）
- `cycle_0001_tree.json` ~ `cycle_0005_tree.json` — 各轮历史树 JSON
- `comparison_report.html` — 跨轮次对比报告（仅 `--max_cycles >= 2` 时生成）

> 如果跑的是 `factor_mad` 或 `alpha_agent` 等其他方法，可视化 block 通过 `hasattr` 判断 `_search_tree_root` 不存在，自动跳过，**完全不影响你的流程**。



### 5.2 手动调用可视化（三行代码）

```python
from tree_viz import SearchTreeVisualizer

viz = SearchTreeVisualizer.from_mcts_method(self.mining_method)
viz.save_html("viz_output/tree.html")                  # 静态报告
viz.save_animated_html("viz_output/tree_anim.html")    # 动态展开动画
```

适合场景：
- 想看一下加了可解释性字段后的 hover 效果
- 想确认多 reward 体系下的节点颜色分布
- 只想在特定轮次生成报告，不想每次都输出


### 5.3 手动生成跨轮次对比报告

已有历史树 JSON 时，可独立运行 `tree_compare.py`：

```bash
# 指定 glob 模式加载所有轮次树，输出对比报告
python tree_compare.py --load "viz_output/cycle_*_tree.json"

# 指定输出路径
python tree_compare.py --load "viz_output/cycle_*_tree.json" --output viz_output/my_comparison.html
```

### 5.4 注意事项

| 事项 | 说明 |
|------|------|
| **不要删除 `tree_viz.py`** | 删掉后 `run.py` 的 `from tree_viz import SearchTreeVisualizer` 会报 `ModuleNotFoundError`，但被 `try/except` 捕获，不影响主流程 |
| **不要删除 `tree_compare.py`** | 同理，跨轮次对比会跳过报错，不影响主流程 |
| **不要删除 `viz_output/` 目录** | `save_html/save_json` 会自动创建，删了也没关系 |
| **不要改 `mining_methods.py` 中的 `to_dict()`** | 其他组员都不会用到这个方法，改了可能影响 JSON 输出格式 |
| HTML 文件过大 | 单树约 3.6 MB（含 Plotly.js 库），对比报告约 3.5 MB |
| 对比报告至少需要 2 轮 | `--max_cycles 1` 时不会生成跨轮次对比 |


---

## 六、总结

本改进为 AlphaJungleMCTS 搜索过程提供了完整的可视化支持，核心成果：

1. **搜索过程透明化**：从黑箱到直观的交互式树形图
2. **语义等价检测**：自动识别 `A+B` vs `B+A` 类冗余，量化搜索浪费
3. **5 维评分可视化**：雷达图网格 + hover 详情，快速定位瓶颈维度
4. **演化路径追踪**：表格展示表达式从根到叶的每步改进
5. **动态展开动画**：▶ Play 按钮展示搜索树逐步生长过程
6. **跨轮次对比分析**：多轮运行后自动生成对比报告，追踪因子质量趋势


总计新增代码：`tree_viz.py` ~870 行 + `tree_compare.py` ~210 行 + `mining_methods.py` 45 行 + `run.py` 26 行。
