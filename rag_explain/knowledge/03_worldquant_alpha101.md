# WorldQuant 101 Formulaic Alphas（精选）

来源：Zura Kakushadze, "101 Formulaic Alphas" (2016), arXiv:1601.00991（公开论文）。
以下为代表性公式化 alpha 的精选，标注其量价结构与逻辑，用于对照解释挖掘出的因子。
原文使用 returns、close、open、high、low、volume、vwap、adv（平均成交额）等字段，
rank 表示横截面排名，ts_/Ts_ 表示时序算子，correlation 即滚动相关。

## Alpha#1 — 波动率反转
`(rank(Ts_ArgMax(SignedPower((returns<0?stddev(returns,20):close),2),5))-0.5)`
逻辑：在亏损期用波动率、盈利期用价格平方，取近 5 日最大值位置排名。
属于波动率 + 反转的组合，捕捉极端波动后的反转。

## Alpha#2 — 量价变动背离
`(-1*correlation(rank(delta(log(volume),2)),rank((close-open)/open),6))`
逻辑：成交量对数 2 日变化的排名 与 当日涨跌幅排名 的 6 日相关性，取负。
典型**量价背离**因子：量价同步上升时给低分。属于反转/背离族。

## Alpha#3 — 开盘价与成交量背离
`(-1*correlation(rank(open),rank(volume),10))`
逻辑：开盘价排名与成交量排名的 10 日相关性取负。量价背离族，
开盘走高但放量者未来收益低。

## Alpha#4 — 低价反转
`(-1*Ts_Rank(rank(low),9))`
逻辑：最低价排名的 9 日时序排名取负。短期低点反转。

## Alpha#6 — 开盘量价相关
`(-1*correlation(open,volume,10))`
逻辑：开盘价与成交量的 10 日相关取负，量价背离族的简洁形式。

## Alpha#12 — 量价短反转
`(sign(delta(volume,1))*(-1*delta(close,1)))`
逻辑：成交量日变化方向 × 价格日变化取负。放量下跌/缩量上涨给正分，
量价配合的 1 日反转。

## Alpha#13 — 量价协方差反转
`(-1*rank(covariance(rank(close),rank(volume),5)))`
逻辑：收盘排名与成交量排名的 5 日协方差排名取负，量价背离族。

## Alpha#21 — 均值与波动的择时
`基于 close 的 8 日均值、2 日均值 与 8 日标准差 的比较，叠加成交量/adv20 条件`
逻辑：短均线 vs 长均线 ± 波动率带，结合量能确认，趋势/反转择时。

## Alpha#41 — 价格几何中枢偏离
`(((high*low)^0.5)-vwap)`
逻辑：最高最低价几何平均与成交量加权均价之差，刻画日内价格相对 vwap 的位置。

## Alpha#101 — 日内动量
`((close-open)/((high-low)+0.001))`
逻辑：日内涨跌幅相对于日内振幅，刻画当日买盘力量强弱，最简动量。

## 共性提炼
- 大量 alpha 以 `correlation(price, volume)` 为核心并取负 → **量价背离**是 WorldQuant
  因子库的主线之一。
- `rank` 几乎无处不在 → 横截面排名是稳健化的标准操作。
- 短窗口（2–10 日）+ 取负 → **短期反转**在公式化 alpha 中占主导。
- 波动率（stddev）常作为状态切换或风险项出现。
