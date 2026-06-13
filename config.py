"""
config.py — 系统配置（股票量价因子挖掘）

所有可调参数集中在此文件，避免散落在各模块。
API Key 优先从环境变量读取，其次使用下方默认值。
"""

import os
from pathlib import Path

# ============================================================
# DeepSeek API 配置
# ============================================================
DEEPSEEK_API_KEY  = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL    = "deepseek-chat"

BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# 股票量价因子挖掘系统配置
# ============================================================
STOCK_CONFIG = {
    # ---- 路径 ----
    "data_path":          str(BASE_DIR / "量价data2000"),        # 量价数据目录（.pqt 格式）
    "metrics_save_path":  str(BASE_DIR / "results" / "metrics"), # 因子表现 CSV 保存路径
    "factors_save_path":  str(BASE_DIR / "results" / "factors"), # 因子值 parquet 保存路径
    "log_path":           str(BASE_DIR / "logs"),
    "checkpoint_path":    str(BASE_DIR / "checkpoints"),
    "method_config":      str(BASE_DIR / "method_config" / "alpha_jungle_mcts.yaml"),
    "validation_config":  str(BASE_DIR / "validation_config" / "full_domain.yaml"),

    # ---- 数据范围 ----
    "start_date": "20100101",   # 训练期开始
    "end_date":   "20191231",   # 训练期结束

    # ---- 因子生成 ----
    "factors_per_cycle":    20,                    # 每轮目标因子数
    "window":               [5, 10, 20, 30, 60],   # 允许的时间窗口
    "correlation_threshold": 0.7,                  # 与已有因子的相关性上限

    # ---- MMR 多样性筛选 ----
    "mmr_lambda":    0.9,   # 越大越偏重 IC，越小越偏重多样性
    "mmr_threshold": 1.0,   # 相关性高于此值的因子直接跳过（1.0=不过滤）

    # ---- 相关性矩阵权重 ----
    "cs_weight": 0.5,   # 截面相关性权重
    "ts_weight": 0.5,   # 时序相关性权重

    # ---- 样本外验证 ----
    "validation_start_date":    "20200101",
    "validation_end_date":      "20250430",
    "validation_min_ic_abs":    0.04,   # |IC| 最低要求
    "validation_min_long_ret":  0.045,  # 多头年化超额收益最低要求
    "validation_max_correlation": 0.7,  # 与已有因子的最大相关性

    # ---- 系统运行 ----
    "n_jobs":         10,   # LLM 并行调用线程数
    "cycle_interval": 0,   # 两轮之间等待时间（秒），0 = 立即开始下一轮
    "error_wait":     3,    # 出错后等待时间（秒）
    "max_cycles":     20,   # 最大轮次，None = 不限
    "max_hours":      8.0,  # 最长运行时间（小时），None = 不限

    # ---- 可选：成分股过滤 ----
    # 取消注释后，IC 和分层收益只计算指数成分股内的股票
    # "poolsel_path": "poolsel/zz800.pqt",  # 可选: hs300 / zz500 / zz800
}
