# AlphaJungleMCTS 多维 Reward 引导搜索改进方案

## 一、背景与动机

### 1.1 原有实现

AlphaJungleMCTS 当前已经在候选因子评估阶段计算了 5 个维度的反馈：

- `effectiveness`：因子预测有效性，主要来自 IC、ICIR、多头收益
- `diversity`：结构差异度，主要来自 AST 相似度
- `stability`：稳定性，主要来自 ICIR、Sharpe 的相对分位
- `turnover`：换手代理得分，越低换手越优
- `overfit`：复杂度、稳定性、差异度与父节点相似度的综合约束

但这些维度在现有搜索中很快被压缩成一个标量：

```python
dimension_scores = {
    "effectiveness": effectiveness_score,
    "diversity": diversity_score,
    "stability": stability_score,
    "turnover": turnover_score,
    "overfit": overfit_score,
}
active_dimensions = list(self.params.get("target_dimensions", self.DEFAULT_DIMENSIONS))
reward = float(np.mean([dimension_scores.get(dim, 0.0) for dim in active_dimensions]))
```

也就是说，5 个维度最终只是简单平均为 `mcts_reward`。

节点选择仍然是标量 UCT：

```python
def _select_node(self, nodes: list[MCTSSearchNode],
                 step: int = 0, total_steps: int = 1) -> MCTSSearchNode | None:
    max_depth = int(self.params.get("max_tree_depth", 4))
    eligible = [node for node in nodes if node.depth < max_depth]
    if not eligible:
        return None

    total_visits = sum(node.visits for node in eligible) + 1
    exploration_c = self._compute_adaptive_uct_c(step, total_steps)

    def _uct(node: MCTSSearchNode) -> float:
        explore = exploration_c * sqrt(log(total_visits + 1.0) / max(1.0, node.visits))
        virtual_bonus = 1.0 / (1.0 + len(node.children))
        return node.q_value + explore + 0.05 * virtual_bonus

    return max(eligible, key=_uct)
```

回传也只保留单个标量 `q_value`：

```python
def _backpropagate(self, node: MCTSSearchNode):
    current = node
    reward = node.reward
    while current.parent is not None:
        parent = current.parent
        parent.visits += 1
        parent.q_value = max(parent.q_value, reward)
        current = parent
```

因此，当前实现虽然有多维反馈，但搜索策略本质上仍是：

```text
UCT(v) = scalar_Q(v) + C(t) * visit_uncertainty(v) + child_bonus(v)
```

### 1.2 当前完整流程实验结果与问题诊断

本次运行日志显示，当前实际配置为：

| 配置项 | 值 |
|--------|-----|
| 搜索预算（search_budget） | 24 步/轮 |
| 总轮数（max_cycles） | 20 |
| UCT 探索系数 | 固定 `uct_c=0.70` |
| 自适应 UCT | `adaptive_uct: false` |
| 训练期 | 2010-01-01 ~ 2019-12-31 |
| 验证期（样本外） | 2020-01-01 ~ 2025-04-30 |
| IC 筛选阈值 | \|ic_5_mean\| > 0.02，\|ic_20_mean\| > 0.04 |
| 最大因子相关性 | 0.70 |

#### 1.2.1 逐轮结果总览

| Cycle | 训练期筛选 | 样本外验证 | 最终入库 | 状态 |
|-------|-----------|-----------|---------|------|
| 1 | 0 通过 | — | 0 | 失败 |
| 2 | 0 通过 | — | 0 | 失败 |
| 3 | 12/20 通过 | 11 | **11** | 成功 |
| 4 | 3/20 通过 | 3 | **3** | 成功 |
| 5 | 6/20 通过 | 6 | **6** | 成功 |
| 6 | 根节点生成失败 | — | 0 | 失败 |
| 7 | 12/19 通过 | 8 | **8** | 成功 |
| 8 | 1/20 通过 | 相关性过滤失败 | 0 | 失败 |
| 9 | 0 通过 | — | 0 | 失败 |
| 10 | 6/19 通过 | 4 | **4** | 成功 |
| 11 | 6/19 通过 | 5 | **5** | 成功 |
| 12 | 9/20 通过 | 9 | **9** | 成功 |
| 13 | 根节点生成失败 | — | 0 | 失败 |
| 14 | 13/20 通过 | 10 | **10** | 成功 |
| 15–19 | 根节点生成失败 | — | 0 | 失败 |
| 20 | LLM 返回异常 | — | 0 | 失败 |

**整体表现：**

| 指标 | 数值 |
|------|------|
| 成功轮次 | 8/20 |
| 入库因子总数 | 56 |
| 平均 \|ic_mean\| | 0.0545 |
| 平均 \|ic_5_mean\| | 0.0464 |
| 平均 \|ic_20_mean\| | 0.0642 |
| 平均 Long Sharpe | 0.5056 |
| 最佳 Long Sharpe | 1.6328 |
| 最佳多头超额收益 | 22.04% |

#### 1.2.2 入库因子的维度表现

| 维度 | 平均值 | 中位数 | 现象 |
|------|-------:|-------:|------|
| `effectiveness_score` | 0.5762 | 0.6136 | 有区分度，但经常被上限截断 |
| `diversity_score` | 0.5293 | 0.5241 | 区分度较强，是前期搜索重要信号 |
| `stability_score` | 0.9410 | 1.0000 | 明显饱和，简单平均时信息量偏低 |
| `turnover_score` | 0.5595 | 0.5830 | 中等区分度 |
| `overfit_score` | 0.4777 | 0.4885 | 偏低，说明复杂度/同质化仍有压力 |

#### 1.2.3 因子结构同质化

本次 56 个入库因子中：

| 字段或结构 | 出现次数 |
|------------|---------:|
| 使用 `close` | 56/56 |
| 使用 `volume` | 55/56 |
| 使用 `amount` | 30/56 |
| 使用 `open/high/low/vwap` | 0/56 |
| 出现 `Delta(close)` 与 `Delta(volume)` 乘积或相关结构 | 48/56 |
| 出现 `Corr(Delta(close), Delta(volume))` | 23/56 |

这说明当前搜索已经明显集中到“收盘价变化 × 成交量变化”的量价共振模板。虽然 MMR 和 AST 相似度能过滤一部分重复表达式，但 MCTS 搜索过程本身仍在反复扩展同类结构。

#### 1.2.4 问题根因：多维反馈没有进入搜索决策

| 环节 | 当前实现 | 问题 |
|------|----------|------|
| 候选打分 | 计算 5 个维度 | 有多维反馈 |
| reward 合成 | 简单平均 5 个维度 | 不区分搜索阶段，不区分维度信息量 |
| 节点选择 | 使用单个 `q_value` | 多维优势被压扁 |
| 回传 | `q_value = max(q_value, reward)` | 只保留标量最大值，容易被偶然高分带偏 |
| target dimension | 按低分维度抽样 | 没有前中后期目标偏好 |
| 最终排序 | 仍使用标量 reward / IC | 搜索期多样性与入库期有效性没有分层 |

核心矛盾是：

> 因子搜索前期需要多样性来打开表达式空间，后期需要有效性和稳定性来收敛到可入库因子；但当前实现把所有阶段都变成同一套等权平均 reward。

### 1.3 改进思路

引入“阶段化多维 Reward 引导搜索”：

1. **前期（探索结构）**：提高 `diversity` 和 `overfit` 权重，鼓励跳出同质模板。
2. **中期（筛选机制）**：平衡 `effectiveness`、`diversity`、`stability`。
3. **后期（收敛入库）**：提高 `effectiveness` 和 `stability` 权重，集中预算开发高预测力区域。

改进后的节点选择公式：

\[
\text{Score}(v, t)
=
\sum_d w_d(t) \cdot Q_d(v)
+ C(t) \cdot \sqrt{\frac{\ln N_{\text{parent}}}{N_v}}
+ B_{\text{child}}(v)
+ B_{\text{novelty}}(v, t)
- P_{\text{risk}}(v)
\]

其中：

- \(d\)：5 个 reward 维度
- \(w_d(t)\)：随搜索进度变化的维度权重
- \(Q_d(v)\)：节点在该维度上的历史价值
- \(C(t)\)：探索系数，只负责访问不确定性，不再承担“探索方向”的职责
- \(B_{\text{novelty}}\)：前期鼓励低相似度、低覆盖维度的结构
- \(P_{\text{risk}}\)：对过高复杂度、过高相似度、低有效值比例的惩罚

### 1.4 学术依据

该方案对应 MCTS / 多目标优化中的几个经典方向：

1. **Multi-objective MCTS**：多目标任务不应过早标量化，至少应保留目标向量供选择阶段动态加权。
2. **Progressive Bias**：搜索前期可引入启发式偏置，随访问次数增加逐渐让真实 reward 主导。
3. **Curriculum Search**：复杂搜索空间中，先扩结构空间，再强化高质量区域，通常比从一开始就追单一目标更稳定。
4. **Pareto Search**：多目标场景中，短期低 effectiveness 但高 diversity 的节点可能是后续高质量分支入口，不应被简单平均过早淘汰。

## 二、详细设计

### 2.1 涉及文件

| 文件 | 修改内容 |
|------|---------|
| `mining_methods.py` | 扩展 `MCTSSearchNode`，改造 `_select_node()`、`_backpropagate()`、`_sample_target_dimension()` |
| `method_config/alpha_jungle_mcts.yaml` | 新增多维 reward 调度配置 |
| `method_config/alpha_jungle_mcts_cosine.yaml` | 同步新增多维 reward 配置，便于和 cosine C 衰减组合实验 |
| `docs/多维Reward引导MCTS改进方案.md` | 记录设计、实验与参数建议 |

### 2.2 新增配置参数

在 `alpha_jungle_mcts.yaml` 的 `params` 中新增：

```yaml
  # ---- 多维 Reward 引导搜索 ----
  multi_reward_uct: true
  reward_schedule:
    early:
      progress_end: 0.35
      weights:
        effectiveness: 0.10
        diversity: 0.35
        stability: 0.15
        turnover: 0.15
        overfit: 0.25
      dimension_temperature: 0.70
    middle:
      progress_end: 0.70
      weights:
        effectiveness: 0.30
        diversity: 0.25
        stability: 0.20
        turnover: 0.125
        overfit: 0.125
      dimension_temperature: 0.45
    late:
      progress_end: 1.00
      weights:
        effectiveness: 0.45
        diversity: 0.05
        stability: 0.25
        turnover: 0.15
        overfit: 0.10
      dimension_temperature: 0.25

  q_mean_weight: 0.30
  q_max_weight: 0.70
  novelty_bonus_weight: 0.08
  risk_penalty_weight: 0.10
```

**向后兼容性**：如果 `multi_reward_uct: false` 或缺失，则继续使用原来的标量 `q_value + C(t)` 选择逻辑。

### 2.3 阶段化权重设计

| 阶段 | step 范围 | 主要目标 | 权重倾向 |
|------|----------|----------|----------|
| early | 0%–35% | 打开搜索空间，避免模板坍缩 | 高 `diversity`、高 `overfit` |
| middle | 35%–70% | 保留多样性，同时开始追有效性 | 平衡 5 维 |
| late | 70%–100% | 集中开发可入库区域 | 高 `effectiveness`、高 `stability` |

权重变化示意：

```text
weight
0.45 |                         effectiveness ****
0.35 | diversity ****
0.30 |              effectiveness **
0.25 | overfit ***  diversity **
0.20 |              stability **
0.15 | turnover *   stability *      turnover *
0.05 |                         diversity *
     +----------+----------+-----------> progress
        early      middle      late
```

### 2.4 代码修改详情

#### 2.4.1 扩展 `MCTSSearchNode`

新增每个节点的多维历史价值统计：

```python
@dataclass
class MCTSSearchNode:
    ...
    dimension_value_sum: dict[str, float] = field(default_factory=dict)
    dimension_value_max: dict[str, float] = field(default_factory=dict)
    dimension_visit_count: int = 0
```

初始化时把自身 `dimension_scores` 写入统计：

```python
def _initialize_dimension_stats(self, node: MCTSSearchNode):
    node.dimension_value_sum = dict(node.dimension_scores or {})
    node.dimension_value_max = dict(node.dimension_scores or {})
    node.dimension_visit_count = 1
```

#### 2.4.2 新增阶段权重方法

在 `AlphaJungleMCTSMethod` 中新增：

```python
def _get_reward_phase(self, step: int, total_steps: int) -> tuple[str, dict[str, float], float]:
    progress = min(step / max(total_steps, 1), 1.0)
    schedule = self.params.get("reward_schedule") or {}

    for phase_name in ["early", "middle", "late"]:
        phase = schedule.get(phase_name, {})
        if progress <= float(phase.get("progress_end", 1.0)):
            weights = phase.get("weights", {})
            temperature = float(phase.get("dimension_temperature", self.params.get("dimension_temperature", 0.45)))
            return phase_name, self._normalize_dimension_weights(weights), temperature

    late = schedule.get("late", {})
    return "late", self._normalize_dimension_weights(late.get("weights", {})), float(
        late.get("dimension_temperature", self.params.get("dimension_temperature", 0.45))
    )
```

权重归一化：

```python
def _normalize_dimension_weights(self, weights: dict[str, float]) -> dict[str, float]:
    dimensions = list(self.params.get("target_dimensions", self.DEFAULT_DIMENSIONS))
    if not weights:
        return {dim: 1.0 / len(dimensions) for dim in dimensions}
    raw = {dim: max(0.0, float(weights.get(dim, 0.0))) for dim in dimensions}
    total = sum(raw.values())
    if total <= 0:
        return {dim: 1.0 / len(dimensions) for dim in dimensions}
    return {dim: value / total for dim, value in raw.items()}
```

#### 2.4.3 新增节点多维价值计算

```python
def _node_dimension_q(self, node: MCTSSearchNode, dim: str) -> float:
    current = safe_float(node.dimension_scores.get(dim), 0.0)
    if node.dimension_visit_count <= 0:
        return current

    mean_value = safe_float(node.dimension_value_sum.get(dim), 0.0) / max(1, node.dimension_visit_count)
    max_value = safe_float(node.dimension_value_max.get(dim), current)

    mean_weight = float(self.params.get("q_mean_weight", 0.30))
    max_weight = float(self.params.get("q_max_weight", 0.70))
    return max_weight * max_value + mean_weight * mean_value
```

#### 2.4.4 修改 `_select_node()`

多维 UCT 版本：

```python
def _select_node(
    self,
    nodes: list[MCTSSearchNode],
    step: int = 0,
    total_steps: int = 1,
) -> MCTSSearchNode | None:
    max_depth = int(self.params.get("max_tree_depth", 4))
    eligible = [node for node in nodes if node.depth < max_depth]
    if not eligible:
        return None

    if not self.params.get("multi_reward_uct", False):
        return self._select_node_scalar_uct(eligible, step, total_steps)

    phase_name, weights, _ = self._get_reward_phase(step, total_steps)
    total_visits = sum(node.visits for node in eligible) + 1
    exploration_c = self._compute_adaptive_uct_c(step, total_steps)

    def _score(node: MCTSSearchNode) -> float:
        weighted_q = sum(
            weights.get(dim, 0.0) * self._node_dimension_q(node, dim)
            for dim in weights
        )
        uncertainty = exploration_c * sqrt(log(total_visits + 1.0) / max(1.0, node.visits))
        child_bonus = 0.05 / (1.0 + len(node.children))
        novelty_bonus = self._compute_novelty_bonus(node, weights)
        risk_penalty = self._compute_search_risk_penalty(node)
        return weighted_q + uncertainty + child_bonus + novelty_bonus - risk_penalty

    selected = max(eligible, key=_score)
    self.system.logger.debug(
        f"MCTS multi-reward phase={phase_name}, weights={weights}, "
        f"selected={selected.node_id}, score={_score(selected):.4f}"
    )
    return selected
```

#### 2.4.5 修改 target dimension 抽样

从“只看低分维度”改成“阶段权重 × 低分压力”：

```python
def _sample_target_dimension(
    self,
    dimension_scores: dict[str, float],
    step: int = 0,
    total_steps: int = 1,
) -> str:
    dimensions = list(self.params.get("target_dimensions", self.DEFAULT_DIMENSIONS))
    _, weights, temperature = self._get_reward_phase(step, total_steps)

    pressure = []
    for dim in dimensions:
        score = safe_float(dimension_scores.get(dim), 0.5)
        score = max(0.0, min(1.0, score))
        pressure.append(weights.get(dim, 0.0) * (1.0 - score))

    chosen_idx = softmax_choice_index(pressure, temperature=temperature)
    return dimensions[chosen_idx]
```

主循环调用改为：

```python
target_dimension = self._sample_target_dimension(
    parent.dimension_scores,
    step=step,
    total_steps=search_budget,
)
```

#### 2.4.6 修改 `_backpropagate()`

回传 child 的 5 维向量，而不是只回传标量 reward：

```python
def _backpropagate(self, node: MCTSSearchNode):
    current = node
    reward = node.reward
    dimension_scores = node.dimension_scores or {}

    while current.parent is not None:
        parent = current.parent
        parent.visits += 1
        parent.q_value = max(parent.q_value, reward)
        parent.dimension_visit_count += 1

        for dim, value in dimension_scores.items():
            value = safe_float(value, 0.0)
            parent.dimension_value_sum[dim] = parent.dimension_value_sum.get(dim, 0.0) + value
            parent.dimension_value_max[dim] = max(
                parent.dimension_value_max.get(dim, value),
                value,
            )

        current = parent
```

#### 2.4.7 最终排序与搜索排序分离

搜索阶段使用阶段权重，最终候选排序使用入库导向权重：

```yaml
  final_reward_weights:
    effectiveness: 0.50
    diversity: 0.10
    stability: 0.25
    turnover: 0.10
    overfit: 0.05
```

最终 `ranked_nodes` 从：

```python
ranked_nodes = sorted(nodes[1:] or nodes, key=lambda node: node.reward, reverse=True)
```

改为：

```python
ranked_nodes = sorted(
    nodes[1:] or nodes,
    key=lambda node: self._weighted_dimension_score(
        node.dimension_scores,
        self.params.get("final_reward_weights"),
    ),
    reverse=True,
)
```

这样搜索前期可以大胆探索，但最终仍偏向能过样本外验证的节点。

## 三、完整修改汇总

### 3.1 `method_config/alpha_jungle_mcts.yaml`

新增：

```yaml
  multi_reward_uct: true
  reward_schedule:
    early:
      progress_end: 0.35
      weights:
        effectiveness: 0.10
        diversity: 0.35
        stability: 0.15
        turnover: 0.15
        overfit: 0.25
      dimension_temperature: 0.70
    middle:
      progress_end: 0.70
      weights:
        effectiveness: 0.30
        diversity: 0.25
        stability: 0.20
        turnover: 0.125
        overfit: 0.125
      dimension_temperature: 0.45
    late:
      progress_end: 1.00
      weights:
        effectiveness: 0.45
        diversity: 0.05
        stability: 0.25
        turnover: 0.15
        overfit: 0.10
      dimension_temperature: 0.25

  q_mean_weight: 0.30
  q_max_weight: 0.70
  novelty_bonus_weight: 0.08
  risk_penalty_weight: 0.10
  final_reward_weights:
    effectiveness: 0.50
    diversity: 0.10
    stability: 0.25
    turnover: 0.10
    overfit: 0.05
```

### 3.2 `mining_methods.py`

**修改点一**：`MCTSSearchNode` 新增多维统计字段。

**修改点二**：新增 `_get_reward_phase()`、`_normalize_dimension_weights()`、`_node_dimension_q()`、`_weighted_dimension_score()`。

**修改点三**：`_select_node()` 支持 `multi_reward_uct`，关闭时回退到原标量 UCT。

**修改点四**：`_sample_target_dimension()` 接收 `step` 和 `total_steps`，按阶段权重抽样。

**修改点五**：`_backpropagate()` 回传 5 维向量。

**修改点六**：最终候选排序使用 `final_reward_weights`。

### 3.3 日志输出

新增建议日志：

```text
INFO  - 多维 Reward UCT 已启用：early/middle/late schedule
DEBUG - step=7/24 phase=early weights={...} selected=node_003 target=diversity score=0.8123
DEBUG - backprop node=node_008 dims={effectiveness=0.62, diversity=0.74, ...}
```

## 四、效果验证方案

### 4.1 对照实验设计

| 实验组 | 配置 | 目的 |
|-------|------|------|
| Baseline | `multi_reward_uct: false`, `adaptive_uct: false`, `uct_c=0.70` | 当前标量 UCT |
| Adaptive C | `multi_reward_uct: false`, `adaptive_uct: true`, `C: 1.0→0.5` | 只验证 C 衰减 |
| MultiReward v1 | `multi_reward_uct: true`, `adaptive_uct: false` | 只验证多维 reward 调度 |
| MultiReward v2 | `multi_reward_uct: true`, `adaptive_uct: true`, `C: 1.0→0.5` | 多维 reward + 自适应 C |
| MultiReward v3 | v2 + 更高 late effectiveness 权重 | 验证后期收敛力度 |

### 4.2 重点观察指标

| 指标 | 期望变化 |
|------|----------|
| 成功率 | 高于当前 8/20 |
| 入库因子总数 | 不低于当前 56，或略低但质量提升 |
| 平均 \|IC\| | 高于 0.0545 |
| 平均 Long Sharpe | 高于 0.5056 |
| 最佳 Long Sharpe | 接近或超过 1.6328 |
| 平均 `diversity_score` | 高于当前 0.5293 |
| `max_ast_similarity` 中位数 | 低于当前 0.4759 |
| 字段覆盖 | 出现更多 `open/high/low/vwap` 或非单一 close-volume 模板 |
| 根节点失败率 | 通过容错单独降低，不应由 reward 方案恶化 |

### 4.3 运行命令

```bash
# Baseline
python run.py --method_config method_config/alpha_jungle_mcts.yaml --max_cycles 20

# Cosine C + multi reward
python run.py --method_config method_config/alpha_jungle_mcts_cosine.yaml --max_cycles 20
```

### 4.4 日志验证

期望日志出现：

```text
多维 Reward UCT 已启用
phase=early
phase=middle
phase=late
target_dimension=diversity
target_dimension=effectiveness
```

并且 early 阶段 `diversity/overfit` target 明显更多，late 阶段 `effectiveness/stability` target 明显更多。

## 五、参数敏感性分析

### 5.1 关键参数建议

| 参数 | 推荐值 | 推荐范围 | 说明 |
|------|-------|---------|------|
| `early.progress_end` | 0.35 | [0.25, 0.45] | 前期结构探索占比 |
| `middle.progress_end` | 0.70 | [0.60, 0.80] | 中期平衡搜索占比 |
| `early.diversity` | 0.35 | [0.25, 0.45] | 过低无法跳出同质模板，过高会牺牲 IC |
| `late.effectiveness` | 0.45 | [0.35, 0.60] | 后期入库导向 |
| `q_max_weight` | 0.70 | [0.50, 0.80] | 保留高潜力分支 |
| `q_mean_weight` | 0.30 | [0.20, 0.50] | 抑制偶然高分 |
| `novelty_bonus_weight` | 0.08 | [0.03, 0.12] | 前期结构新颖性奖励 |
| `risk_penalty_weight` | 0.10 | [0.05, 0.20] | 抑制复杂/过拟合节点 |

### 5.2 预期现象

| 现象 | 解释 | 处理 |
|------|------|------|
| 因子数量上升但 Sharpe 下降 | 前期探索变强，低质量边界因子增多 | 提高 `final_reward_weights.effectiveness` 或样本外阈值 |
| 因子数量下降但平均质量上升 | late 阶段收敛更强 | 可接受，观察最佳因子和组合效果 |
| diversity 提升但 IC 下降 | early 权重过度偏多样性 | 降低 `early.diversity`，提高 `middle.effectiveness` |
| 搜索仍集中 close-volume | diversity 只看 AST 不够 | 后续加入 field/operator diversity |
| stability 长期饱和 | 当前 stability 打分信息量不足 | 改为分位数校准或滚动标准化 |

## 六、后续增强方向

### 6.1 Reward 校准

当前 `stability_score` 明显饱和，简单加权会放大低信息量维度。后续可加入：

- 每轮候选内分位数归一化
- 历史 rolling mean/std 标准化
- 对长期饱和维度降低边际贡献

### 6.2 多样性拆分

当前 `diversity_score` 主要来自 AST 相似度。建议拆成：

- `ast_diversity`
- `field_diversity`
- `operator_diversity`
- `value_corr_diversity`

这样才能真正避免所有因子集中在 `close-volume-amount` 模板。

### 6.3 Pareto Frontier 候选池

对 5 维 reward 维护 Pareto frontier：只要节点没有被其他节点在所有维度上支配，就保留为候选。阶段权重只用于从 frontier 中选择扩展节点，而不是过早淘汰某个维度突出的节点。

## 七、总结

本方案的核心不是继续调整 UCT 中的 `C`，而是把已经计算出来的 5 维 reward 真正接入 MCTS 的选择、抽样、回传和最终排序。

预期收益：

1. **前期更能跳出同质结构**：降低 close-volume 模板反复扩展的问题。
2. **中期保留多条机制路径**：避免单个偶然高分节点支配整棵树。
3. **后期更聚焦可入库质量**：提高 effectiveness 和 stability 对最终搜索预算的控制力。
4. **与自适应 C 兼容**：`C(t)` 继续控制访问不确定性，多维 reward 控制搜索方向。
5. **向后兼容**：`multi_reward_uct: false` 即退回现有标量 UCT 行为。

如果只做一版 v1，建议优先实现：

1. 阶段化 reward 权重；
2. 五维向量回传；
3. target dimension 阶段化抽样；
4. final ranking 与 search reward 分离。

这 4 点能以较小代码改动，让 AlphaJungleMCTS 从“计算多维、使用标量”升级为真正的“多维 reward 引导搜索”。
