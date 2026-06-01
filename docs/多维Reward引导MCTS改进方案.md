# AlphaJungleMCTS 多维 Reward 引导搜索改进方案

## 一、背景与问题诊断

### 1.1 Baseline 中多维反馈的使用方式

AlphaJungleMCTS 在候选因子评估阶段已经计算了 5 个维度的反馈：

- `effectiveness`：因子预测有效性，主要来自 IC、ICIR、多头收益
- `diversity`：结构差异度，主要来自 AST 相似度
- `stability`：稳定性，主要来自 ICIR、Sharpe 的相对分位
- `turnover`：换手代理得分，越低换手越优
- `overfit`：复杂度、稳定性、差异度与父节点相似度的综合约束

但在 baseline 中，这些维度会被简单平均成单一 `mcts_reward`：

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

随后节点选择仍然使用标量 UCT：

```text
UCT(v) = q_value(v) + C * visit_bonus(v) + child_bonus(v)
```

回传也只更新单一 `q_value`：

```python
parent.q_value = max(parent.q_value, reward)
```

因此 baseline 虽然计算了多维反馈，但多维信息主要用于记录、简单平均和 target dimension 抽样，没有真正进入 UCT 的多目标选择与回传。

### 1.2 Baseline 实验暴露的问题

固定 C baseline 的 20 轮实验中，最终入库 56 个因子，成功轮次为 8/20。整体表现可用，但存在几个明显问题：

| 问题 | 表现 | 影响 |
|------|------|------|
| 多维信息被压缩 | 5 维 reward 简单平均为 `mcts_reward` | 不同阶段无法偏向不同搜索目标 |
| 搜索阶段不分层 | 前期、后期都使用同一标量 reward | 前期探索和后期收敛目标混在一起 |
| 回传信息不足 | 父节点只记录标量 `q_value` | 无法知道节点在哪些维度上长期表现好 |
| 最终排序粗糙 | 候选排序仍主要依赖标量 reward | 搜索期多样性和入库期有效性没有分离 |
| 结构同质化 | 56 个因子全部使用 `close`，55 个使用 `volume` | 搜索集中在 close-volume 模板 |

Baseline 入库因子的关键统计：

| 指标 | 数值 |
|------|------|
| 成功轮次 | 8/20 |
| 入库因子总数 | 56 |
| 平均 \|ic_mean\| | 0.0545 |
| 平均 Long Sharpe | 0.5056 |
| 最佳 Long Sharpe | 1.6328 |
| 最佳多头超额收益 | 22.04% |

结构分布也显示出集中现象：

| 字段或结构 | 出现次数 |
|------------|---------:|
| 使用 `close` | 56/56 |
| 使用 `volume` | 55/56 |
| 使用 `amount` | 30/56 |
| 使用 `open/high/low/vwap` | 0/56 |
| 出现 `Corr(Delta(close), Delta(volume))` | 23/56 |

核心矛盾是：

> 因子搜索前期需要多样性打开表达式空间，后期需要有效性和稳定性收敛到可入库因子；但 baseline 把所有阶段都压成同一套标量 reward。

## 二、v1 改进目标

v1 只做最小可用的多维 reward 搜索改造，不引入新的验证逻辑、不修改算子、不修改训练期或样本外筛选。

目标如下：

1. **让多维 reward 进入 UCT 选择**：节点选择不再只看标量 `q_value`。
2. **让搜索阶段有明确偏好**：early 偏探索，middle 平衡，late 偏入库质量。
3. **让回传保留维度信息**：父节点记录每个维度的历史均值和最大值。
4. **区分搜索排序和最终排序**：搜索期可以鼓励多样性，最终排序更偏入库质量。
5. **保持最小化和向后兼容**：关闭 `multi_reward_uct` 时，回退到原标量 UCT。

本方案不修改：

- `method_config/alpha_jungle_mcts.yaml`
- `evaluator.py`
- `factor_mining.py`
- 算子实现
- 训练期和样本外验证逻辑
- MMR 筛选逻辑

## 三、v1 设计

### 3.1 多维 UCT 公式

v1 的节点选择公式为：

\[
\text{Score}(v,t)
=
\sum_d w_d(t) \cdot Q_d(v)
+ C(t) \cdot \sqrt{\frac{\ln N}{N_v}}
+ 0.05 \cdot \frac{1}{1 + |\text{children}(v)|}
\]

其中：

- \(d\)：5 个 reward 维度
- \(w_d(t)\)：随搜索阶段变化的维度权重
- \(Q_d(v)\)：节点在维度 \(d\) 上的历史价值
- \(C(t)\)：自适应探索系数，只负责访问不确定性
- child bonus：保留原实现对子节点较少节点的轻微扩展偏置

设计上把职责拆开：

| 组件 | 职责 |
|------|------|
| `C(t)` | 控制访问不确定性 |
| 多维 reward 权重 | 控制当前阶段的搜索方向 |
| `final_reward_weights` | 控制最终候选排序 |

### 3.2 阶段化权重

v1 将搜索过程分为 early / middle / late 三个阶段：

| 阶段 | 进度范围 | 主要目标 | 权重倾向 |
|------|----------|----------|----------|
| early | 0%–35% | 打开结构空间，避免过早收敛 | 高 `diversity`、高 `overfit` |
| middle | 35%–70% | 保留多样性，同时开始追有效性 | 五维相对均衡 |
| late | 70%–100% | 集中开发可入库区域 | 高 `effectiveness`、高 `stability` |

配置只加在 `method_config/alpha_jungle_mcts_cosine.yaml` 中：

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
final_reward_weights:
  effectiveness: 0.50
  diversity: 0.10
  stability: 0.25
  turnover: 0.10
```

`method_config/alpha_jungle_mcts.yaml` 保持固定 C / 标量 UCT baseline，不加入多维 reward 配置。

### 3.3 节点维度统计

`MCTSSearchNode` 新增每个节点的 5 维历史统计：

```python
dimension_value_sum: dict[str, float] = field(default_factory=dict)
dimension_value_max: dict[str, float] = field(default_factory=dict)
dimension_visit_count: int = 0
```

节点初始化时，如果已有 `dimension_scores`，就写入初始统计：

```python
if not self.dimension_value_sum:
    self.dimension_value_sum = {
        dim: safe_float(value, 0.0)
        for dim, value in (self.dimension_scores or {}).items()
    }
if not self.dimension_value_max:
    self.dimension_value_max = dict(self.dimension_value_sum)
if not self.dimension_visit_count and self.dimension_scores:
    self.dimension_visit_count = 1
```

节点在某一维度上的 \(Q_d\) 由历史均值和历史最大值共同决定：

```python
mean_value = dimension_value_sum[dim] / dimension_visit_count
max_value = dimension_value_max[dim]
q_d = (q_max_weight * max_value + q_mean_weight * mean_value) / total_weight
```

默认：

```yaml
q_mean_weight: 0.30
q_max_weight: 0.70
```

这样既保留高潜力分支，也避免完全被单次偶然高分控制。

### 3.4 节点选择

`_select_node()` 在 `multi_reward_uct: true` 时使用阶段权重计算 exploit：

```python
if use_multi_reward:
    exploit = sum(
        reward_weights.get(dim, 0.0) * self._node_dimension_q(node, dim)
        for dim in reward_weights
    )
else:
    exploit = node.q_value
```

探索项仍然沿用自适应 UCT：

```python
explore = exploration_c * sqrt(log(total_visits + 1.0) / max(1.0, node.visits))
virtual_bonus = 1.0 / (1.0 + len(node.children))
score = exploit + explore + 0.05 * virtual_bonus
```

也就是说，v1 中：

- `C(t)` 继续控制访问不确定性；
- 多维 reward 权重控制搜索方向；
- 若关闭 `multi_reward_uct`，仍使用原来的标量 `q_value`。

### 3.5 target dimension 抽样

Baseline 中 target dimension 主要按“低分维度压力”抽样。v1 改为：

```text
抽样压力 = 阶段权重 × 当前维度低分压力
```

实现含义：

- early 阶段更容易抽到 `diversity`、`overfit`
- middle 阶段五维相对均衡
- late 阶段更容易抽到 `effectiveness`、`stability`

这会影响 LLM refinement prompt 的改写方向，使搜索过程和阶段目标一致。

### 3.6 回传

v1 保留原标量 `q_value` 回传，同时额外回传 5 维向量：

```python
parent.visits += 1
parent.q_value = max(parent.q_value, reward)
parent.dimension_visit_count += 1

for dim, value in dimension_scores.items():
    value = safe_float(value, 0.0)
    parent.dimension_value_sum[dim] = parent.dimension_value_sum.get(dim, 0.0) + value
    parent.dimension_value_max[dim] = max(parent.dimension_value_max.get(dim, value), value)
```

这样父节点不只知道“这个分支是否整体好”，也知道“这个分支在哪些维度上好”。

### 3.7 最终排序

搜索阶段使用阶段化权重，最终输出候选时使用入库导向权重：

```yaml
final_reward_weights:
  effectiveness: 0.50
  diversity: 0.10
  stability: 0.25
  turnover: 0.10
```

排序逻辑：

```python
ranked_nodes = sorted(
    rank_pool,
    key=lambda node: self._weighted_dimension_score(
        node.dimension_scores,
        final_reward_weights,
    ),
    reverse=True,
)
```

如果没有配置 `final_reward_weights`，则沿用原来的 `node.reward` 排序。

## 四、最小化原则

v1 的改动范围保持在 MCTS 内部和 cosine 实验配置：

| 模块 | v1 改动 |
|------|---------|
| `mining_methods.py` | 多维统计、多维 UCT、阶段化 target 抽样、五维回传、最终加权排序 |
| `method_config/alpha_jungle_mcts_cosine.yaml` | 启用 `multi_reward_uct`，新增阶段权重和最终权重 |
| `docs/多维Reward引导MCTS改进方案.md` | 记录方案 |

保持不变：

- baseline 配置 `method_config/alpha_jungle_mcts.yaml`
- 数据加载
- 因子计算
- 指标计算
- 训练期筛选
- 样本外验证
- MMR
- 结果保存格式

因此，v1 的实验对照关系比较清晰：

```text
baseline: 固定 C + 标量 UCT
v1: cosine 动态 C + 多维 reward UCT
```

## 五、验证方案

### 5.1 静态检查

```bash
python -m py_compile mining_methods.py
python -c "import yaml; yaml.safe_load(open('method_config/alpha_jungle_mcts_cosine.yaml', encoding='utf-8'))"
```

### 5.2 Synthetic 检查

构造少量本地节点，检查：

- `MCTSSearchNode` 初始化后存在 `dimension_value_sum`、`dimension_value_max`、`dimension_visit_count`
- `multi_reward_uct: true` 时 `_select_node()` 可以计算多维加权分数
- `multi_reward_uct: false` 或缺失时仍回退标量 UCT
- `_sample_target_dimension()` 会受阶段权重影响
- `_backpropagate()` 会更新父节点的 5 维统计
- 配置 `final_reward_weights` 时最终排序使用多维加权分数

### 5.3 小规模运行

```bash
python run.py --method_config method_config/alpha_jungle_mcts_cosine.yaml --max_cycles 1
```

检查日志中：

```text
加载挖掘方法配置：alpha_jungle_mcts_cosine.yaml
自适应 UCT 已启用：C_max=1.0, C_min=0.5, decay=cosine
训练期 筛选完成
MMR 筛选
样本外指标 筛选完成
```

最终 metrics 中应保留：

- `target_dimension`
- `effectiveness_score`
- `diversity_score`
- `stability_score`
- `turnover_score`
- `overfit_score`
- `mcts_reward`

## 六、实验观察重点

v1 的实验重点不是引入更多复杂约束，而是确认“多维反馈进入搜索决策”是否有效。

重点观察：

| 指标 | 关注点 |
|------|--------|
| 入库因子数 | 是否高于 baseline 的 56 |
| 训练期通过率 | 搜索是否更容易触达有效候选 |
| 样本外通过率 | 多维搜索是否破坏泛化 |
| 平均 \|IC\| | 整体方向性是否增强 |
| 平均多头超额收益 | 收益质量是否改善 |
| 平均 Long Sharpe | 稳定收益是否改善 |
| 最佳因子表现 | 是否牺牲 top 因子能力 |
| `target_dimension` 分布 | 阶段化目标是否改变搜索方向 |
| 表达式结构分布 | 是否仍集中在 close-volume 模板 |

## 七、总结

v1 的核心改动是把已经计算出来的 5 维 reward 从“记录和简单平均”提升为“搜索决策信号”。

具体来说：

1. 搜索选择从标量 `q_value` 改为阶段化多维 \(Q_d\) 加权。
2. target dimension 抽样从单纯低分压力改为阶段权重 × 低分压力。
3. 回传从单一 reward 扩展为 5 维统计。
4. 最终排序从搜索期 reward 分离为入库导向权重。
5. baseline 配置和验证流程保持不变，保证对照实验清晰。

这是一版最小化多维 reward 改进：不改变外部验证流程，不引入额外复杂约束，只让 MCTS 真正使用已有的多维反馈。
