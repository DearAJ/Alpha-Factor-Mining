"""
run.py — 自动化因子挖掘系统主入口（股票量价版）

使用方式：
    python run.py                    # 使用 config.py 中的默认配置运行
    python run.py --max_cycles 5    # 最多运行 5 轮
    python run.py --max_hours 2.0   # 最多运行 2 小时

当前流程：
  1. 样本内挖掘  — 由 method_config 指定具体方法
  2. MMR 筛选   — 在候选因子中按 IC/多样性选出代表因子
  3. 样本外验证 — 在 2020-2025 数据上评估，满足阈值才入库
"""

import os
import sys
import time
import pickle
import logging
import datetime
import argparse
import traceback
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from config import STOCK_CONFIG
from dataloader import dataloader
from operators import ExpressionCalculator, Operators
from correlations import cal_correlation_matrix_comprehensive
from evaluator import cal_alpha
from metrics_hooks import compute_extra_metrics, load_data_deps, passes_filter
from mining_methods import build_mining_method
from factor_mining import (
    calculate_factors_performance,
    is_low_correlated_with_fixed_factors,
    mmr_selection,
)


class AutomatedFactorSystem:
    """
    自动化量价因子挖掘系统（股票 A 股版）。

    核心循环：样本内挖掘 → 样本外验证 → 入库
    """

    def __init__(self, config: dict):
        self.config = config
        self.setup_logging()
        self.setup_directories()
        self.load_method_config()
        self.load_validation_config()
        self.load_data()
        self.load_baseline_factors()
        self.mining_method = build_mining_method(self, self.method_profile)
        self.cycle_count = self.load_checkpoint()

    # ----------------------------------------------------------------
    # 初始化
    # ----------------------------------------------------------------

    def setup_logging(self):
        log_dir = Path(self.config["log_path"])
        log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("FactorSystem")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers.clear()

        log_file = log_dir / f"factor_system_{datetime.datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        for h in [file_handler, console_handler]:
            h.setFormatter(formatter)
            self.logger.addHandler(h)

    def setup_directories(self):
        for d in [
            self.config["metrics_save_path"],
            self.config["factors_save_path"],
            self.config["log_path"],
            self.config["checkpoint_path"],
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)
        self.logger.info("目录初始化完成")

    def load_validation_config(self):
        cfg_path = self.config.get("validation_config")
        if not cfg_path:
            self.validation_profile = {
                "name": "inline_default",
                "data_deps": {},
                "metric_hooks": [],
                "metric_hook_params": {},
                "train_filter": [],
                "validation_filter": [],
            }
            self.loaded_metric_deps = {}
            return

        cfg_path = Path(cfg_path).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = (Path(__file__).resolve().parent / cfg_path).resolve()

        with open(cfg_path, "r", encoding="utf-8") as f:
            profile = yaml.safe_load(f) or {}

        profile.setdefault("name", cfg_path.stem)
        profile["metric_hooks"] = profile.get("metric_hooks") or []
        profile["metric_hook_params"] = profile.get("metric_hook_params") or {}
        profile["validation_filter"] = profile.get("validation_filter") or []
        raw_train_filter = profile.get("train_filter")
        if raw_train_filter:
            profile["train_filter"] = raw_train_filter
            profile["train_filter_inherited"] = False
        elif profile["validation_filter"]:
            profile["train_filter"] = copy.deepcopy(profile["validation_filter"])
            profile["train_filter_inherited"] = True
        else:
            profile["train_filter"] = []
            profile["train_filter_inherited"] = False
        profile["poolsel_path"] = profile.get("poolsel_path")

        raw_deps = profile.get("data_deps") or {}
        resolved_deps = {}
        for dep_name, dep_path in raw_deps.items():
            if not dep_path:
                continue
            dep_path = Path(dep_path).expanduser()
            if not dep_path.is_absolute():
                dep_path = (cfg_path.parent / dep_path).resolve()
            resolved_deps[dep_name] = str(dep_path)

        profile["data_deps"] = resolved_deps
        if profile.get("poolsel_path"):
            poolsel_path = Path(profile["poolsel_path"]).expanduser()
            if not poolsel_path.is_absolute():
                poolsel_path = (cfg_path.parent / poolsel_path).resolve()
            profile["poolsel_path"] = str(poolsel_path)
        self.validation_profile = profile
        self.validation_config_path = cfg_path
        self.loaded_metric_deps = load_data_deps(resolved_deps)

        self.logger.info(
            f"加载验证配置：{profile['name']} ({cfg_path})"
        )
        self.logger.info(
            f"验证配置详情：hooks={profile['metric_hooks']}, "
            f"train_filter={len(profile['train_filter'])}, "
            f"validation_filter={len(profile['validation_filter'])}"
        )
        if profile.get("train_filter_inherited"):
            self.logger.info("train_filter 为空，默认继承 validation_filter")

    def load_method_config(self):
        cfg_path = self.config.get("method_config")
        if not cfg_path:
            self.method_profile = {
                "name": "factor_mad",
                "method": "factor_mad",
                "params": {},
            }
            return

        cfg_path = Path(cfg_path).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = (Path(__file__).resolve().parent / cfg_path).resolve()

        with open(cfg_path, "r", encoding="utf-8") as f:
            profile = yaml.safe_load(f) or {}

        profile.setdefault("name", cfg_path.stem)
        profile.setdefault("method", "factor_mad")
        profile["params"] = profile.get("params") or {}
        self.method_profile = profile
        self.method_config_path = cfg_path

        self.logger.info(f"加载挖掘方法配置：{profile['name']} ({cfg_path})")
        self.logger.info(
            f"挖掘方法详情：method={profile['method']}, params={profile['params']}"
        )

    def load_data(self):
        self.logger.info("加载基础数据...")
        self.factor_dfs = dataloader(
            self.config["data_path"],
            start_date=self.config["start_date"],
            end_date=self.config["end_date"],
        )
        self.calculator = ExpressionCalculator(self.factor_dfs)
        self.ops_info = Operators.get_all_method_info()
        self.fields = list(self.factor_dfs.keys())

        self.logger.info(f"数据加载完成：{len(self.factor_dfs)} 个字段，{len(self.ops_info)} 个算子")

    def apply_metric_filters(
        self,
        factors_df: pd.DataFrame,
        factor_values_map: dict,
        filter_conditions: list,
        start_date: str,
        end_date: str,
        phase_label: str,
        runtime_context: dict | None = None,
    ) -> pd.DataFrame:
        if factors_df.empty or not filter_conditions:
            return factors_df

        hook_names = self.validation_profile.get("metric_hooks", [])
        passed_rows = []

        for _, row in factors_df.iterrows():
            metrics = row.to_dict()
            alpha_name = metrics["alpha_id"]

            if hook_names:
                factor_values = factor_values_map.get(alpha_name)
                if factor_values is None:
                    self.logger.warning(f"{phase_label} {alpha_name} 缺少因子值，无法计算扩展指标")
                    continue
                try:
                    hook_deps = dict(self.loaded_metric_deps)
                    hook_deps["poolsel_path"] = self.validation_profile.get("poolsel_path")
                    if runtime_context:
                        hook_deps.update(runtime_context)
                    extra_metrics = compute_extra_metrics(
                        factor_values,
                        hook_names,
                        hook_deps,
                        start_date,
                        end_date,
                        self.validation_profile.get("metric_hook_params"),
                    )
                    metrics.update(extra_metrics)
                except Exception as e:
                    self.logger.warning(f"{phase_label} {alpha_name} 额外指标计算失败：{e}")
                    continue

            passed, reason = passes_filter(metrics, filter_conditions)
            if passed:
                passed_rows.append(metrics)
            else:
                self.logger.info(f"{phase_label} {alpha_name} 未通过筛选：{reason}")

        if not passed_rows:
            self.logger.warning(f"{phase_label} 没有因子通过指标筛选")
            return pd.DataFrame()

        filtered_df = pd.DataFrame(passed_rows)
        self.logger.info(f"{phase_label} 筛选完成：{len(factors_df)} → {len(filtered_df)} 个因子")
        return filtered_df

    def load_baseline_factors(self):
        """预加载已入库的 baseline 因子（用于 MMR 多样性约束）"""
        self.logger.info("预加载 baseline 因子...")
        self.baseline_factors = {}
        self.baseline_factor_records = []

        # 只从当前项目的已保存因子目录加载
        for path in [self.config["factors_save_path"]]:
            try:
                fdict = dataloader(path, self.config["start_date"], self.config["end_date"])
                self.baseline_factors.update(fdict)
                self.logger.info(f"预加载 {path}：{len(fdict)} 个因子")
            except Exception as e:
                self.logger.warning(f"预加载 {path} 失败：{e}")

        metric_records = {}
        metrics_dir = Path(self.config["metrics_save_path"])
        for csv_path in sorted(metrics_dir.glob("*.csv")):
            try:
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
            except Exception as e:
                self.logger.warning(f"读取 baseline 指标文件失败 {csv_path.name}: {e}")
                continue

            if "alpha_id" not in df.columns or "expression" not in df.columns:
                continue

            for _, row in df.iterrows():
                alpha_id = row.get("alpha_id")
                expression = row.get("expression")
                if pd.isna(alpha_id) or pd.isna(expression):
                    continue
                record = row.to_dict()
                explanation = str(record.get("explanation", ""))
                record["explanation"] = explanation.split("=====")[0].strip()
                metric_records[str(alpha_id)] = record

        self.baseline_factor_records = list(metric_records.values())
        self.logger.info(f"共预加载 {len(self.baseline_factor_records)} 条 baseline 因子元数据")
        self.logger.info(f"共预加载 {len(self.baseline_factors)} 个 baseline 因子")

    def load_checkpoint(self) -> int:
        ckpt_file = Path(self.config["checkpoint_path"]) / "system_checkpoint.pkl"
        if ckpt_file.exists():
            try:
                with open(ckpt_file, "rb") as f:
                    ckpt = pickle.load(f)
                n = ckpt.get("cycle_count", 0)
                self.logger.info(f"从检查点恢复，当前轮次：{n}")
                return n
            except Exception as e:
                self.logger.error(f"加载检查点失败：{e}")
        return 0

    def save_checkpoint(self, cycle_count: int):
        ckpt_file = Path(self.config["checkpoint_path"]) / "system_checkpoint.pkl"
        try:
            with open(ckpt_file, "wb") as f:
                pickle.dump({"cycle_count": cycle_count, "timestamp": datetime.datetime.now()}, f)
        except Exception as e:
            self.logger.error(f"保存检查点失败：{e}")

    def alpha_mmr_selection(
        self, factors_df: pd.DataFrame, label: str
    ) -> tuple:
        self.logger.info(f"MMR 筛选（{label}）")

        alpha_list = factors_df["expression"].to_list()
        alpha_name = factors_df["alpha_id"].to_list()
        candidate_factor_values = cal_alpha(alpha_list, alpha_name, self.calculator)

        all_factor_values = {**candidate_factor_values, **self.baseline_factors}
        self.logger.info(
            f"计算相关性矩阵：{len(candidate_factor_values)} 个候选 + "
            f"{len(self.baseline_factors)} 个 baseline = {len(all_factor_values)} 个因子"
        )
        corr_matrix = cal_correlation_matrix_comprehensive(
            all_factor_values,
            cs_weight=self.config["cs_weight"],
            ts_weight=self.config["ts_weight"],
        )

        selected_alphas, selected_values = mmr_selection(
            factors_df,
            candidate_factor_values,
            list(self.baseline_factors.keys()),
            corr_matrix,
            num_to_select=self.config["factors_per_cycle"],
            lambda_param=self.config["mmr_lambda"],
            threshold=self.config["mmr_threshold"],
        )

        selected_df = factors_df[factors_df["alpha_id"].isin(selected_alphas)].copy()
        self.logger.info(
            f"MMR 选择完成（{label}）：{len(factors_df)} → {len(selected_df)} 个因子"
        )
        return selected_df, selected_values

    def validate_factors_out_of_sample(
        self,
        factors_df: pd.DataFrame,
        validation_start: str,
        validation_end: str,
    ) -> tuple:
        """样本外验证：在 validation 期重新计算 IC 并应用阈值过滤"""
        self.logger.info(f"样本外验证（{validation_start} - {validation_end}）")

        # 加载验证期数据
        val_dfs = dataloader(
            self.config["data_path"],
            start_date=validation_start,
            end_date=validation_end,
        )
        val_calculator = ExpressionCalculator(val_dfs)

        alpha_list = factors_df["expression"].to_list()
        alpha_names = factors_df["alpha_id"].to_list()

        val_results = []
        val_factor_values = {}

        for alpha_name, expression in zip(alpha_names, alpha_list):
            try:
                val_fv = val_calculator.calculate(expression)
                perf = calculate_factors_performance(
                    {alpha_name: val_fv},
                    [expression],
                    val_dfs["close"],
                    start_date=validation_start,
                    end_date=validation_end,
                    poolsel_path=self.validation_profile.get("poolsel_path"),
                )[0]
                if perf.get("ic_mean") is None or np.isnan(perf.get("ic_mean", np.nan)):
                    self.logger.warning(f"因子 {alpha_name} 验证期 IC 无效，跳过")
                    continue
                val_factor_values[alpha_name] = val_fv
                val_results.append(perf)
            except Exception as e:
                self.logger.warning(f"因子 {alpha_name} 验证期计算失败：{e}")

        if not val_results:
            self.logger.warning("所有因子验证期计算均失败！")
            return pd.DataFrame(), {}

        val_df = pd.DataFrame(val_results)
        val_df = self.apply_metric_filters(
            val_df,
            val_factor_values,
            self.validation_profile.get("validation_filter", []),
            validation_start,
            validation_end,
            "样本外指标",
            runtime_context={
                "close": val_dfs["close"],
                "poolsel_path": self.validation_profile.get("poolsel_path"),
            },
        )
        if val_df.empty:
            return pd.DataFrame(), {}

        max_corr = self.config.get("validation_max_correlation", 0.7)
        passed = []
        for _, row in val_df.iterrows():
            name = row["alpha_id"]

            # 与已有因子相关性检查
            low_corr = is_low_correlated_with_fixed_factors(
                val_factor_values[name], self.baseline_factors, max_corr
            )
            if not low_corr:
                self.logger.info(f"{name} 未通过相关性筛选")
                continue

            passed.append(name)
            self.logger.info(f"{name} 通过所有验证条件")

        del val_dfs, val_calculator

        if not passed:
            self.logger.warning("没有因子通过样本外验证！")
            return pd.DataFrame(), {}

        # 用完整时间范围（训练+验证）重新计算因子值用于保存
        full_dfs = dataloader(
            self.config["data_path"],
            start_date=self.config["start_date"],
            end_date=validation_end,
        )
        full_calc = ExpressionCalculator(full_dfs)
        final_fv = {}
        validated_df = factors_df[factors_df["alpha_id"].isin(passed)].copy()
        for name in passed:
            expr = validated_df[validated_df["alpha_id"] == name]["expression"].iloc[0]
            final_fv[name] = full_calc.calculate(expr)

        self.logger.info(f"通过验证的因子：{len(validated_df)} 个")
        return validated_df, final_fv

    def save_final_results(
        self,
        final_factors_df: pd.DataFrame,
        final_factor_values: dict,
        cycle_count: int,
        cycle_timestamp: str,
    ) -> dict:
        """保存入库因子：CSV 指标 + parquet 因子值"""
        metrics_file = (
            Path(self.config["metrics_save_path"])
            / f"cycle_{cycle_count:04d}_{cycle_timestamp}_final_metrics.csv"
        )
        final_factors_df.to_csv(metrics_file, index=False, encoding="utf-8-sig")

        factors_dir = Path(self.config["factors_save_path"])
        factors_dir.mkdir(exist_ok=True)
        for alpha_id, fv in final_factor_values.items():
            fv.to_parquet(factors_dir / f"{alpha_id}.pqt")

        # 更新 baseline（本轮入库的因子下轮可用）
        self.baseline_factors.update(final_factor_values)
        record_map = {str(x.get("alpha_id")): x for x in self.baseline_factor_records}
        for _, row in final_factors_df.iterrows():
            record = row.to_dict()
            explanation = str(record.get("explanation", ""))
            record["explanation"] = explanation.split("=====")[0].strip()
            record_map[str(record["alpha_id"])] = record
        self.baseline_factor_records = list(record_map.values())

        self.logger.info(f"结果保存完成：{metrics_file}，{len(final_factors_df)} 个因子入库")
        return {
            "metrics_file": str(metrics_file),
            "factors_dir": str(factors_dir),
            "factor_count": len(final_factors_df),
        }

    # ----------------------------------------------------------------
    # 主循环
    # ----------------------------------------------------------------

    def run_single_cycle(self) -> bool:
        """
        执行一轮因子挖掘循环：
          1. 样本内挖掘
          2. 样本外验证
          3. 入库
        """
        self.cycle_count += 1
        cycle_count = self.cycle_count
        cycle_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger.info(f"===== 开始第 {cycle_count} 轮循环 ({cycle_timestamp}) =====")

        try:
            # 样本内挖掘（由 method_config 指定）
            final_df = self.mining_method.mine_in_sample(cycle_timestamp)

            # ── [可视化] MCTS 搜索树导出（仅 alpha_jungle_mcts 生效）─────────
            # 如需禁用可视化，注释掉下方 block 即可。
            # 注意：本 block 不会影响 factor_mad / alpha_agent 等方法。
            # ─────────────────────────────────────────────────────────────────
            if hasattr(self.mining_method, "_search_tree_root") and self.mining_method._search_tree_root is not None:
                try:
                    from tree_viz import SearchTreeVisualizer

                    viz = SearchTreeVisualizer.from_mcts_method(self.mining_method)
                    viz.save_html("viz_output/mcts_search_tree.html")
                    viz.save_json("viz_output/mcts_tree.json")
                    self.logger.info(f"MCTS 搜索树可视化已保存到 viz_output/")
                except Exception as viz_err:
                    self.logger.warning(f"MCTS 可视化导出失败（不影响主流程）: {viz_err}")
            # ── [可视化结束] ─────────────────────────────────────────────────

            if final_df.empty:
                self.logger.warning("样本内挖掘结果为空，跳过本轮")
                return False

            # 阶段 4：样本外验证
            validated_df, validated_fv = self.validate_factors_out_of_sample(
                final_df,
                validation_start=self.config.get("validation_start_date", "20200101"),
                validation_end=self.config.get("validation_end_date", "20250430"),
            )
            if validated_df.empty:
                self.logger.warning("没有因子通过样本外验证，本轮无结果保存")
                return False

            # 入库
            save_info = self.save_final_results(
                validated_df, validated_fv, cycle_count, cycle_timestamp
            )
            self.logger.info(f"===== 第 {cycle_count} 轮完成，入库 {save_info['factor_count']} 个因子 =====")
            return True

        except Exception as e:
            self.logger.error(f"第 {cycle_count} 轮执行失败：{e}")
            self.logger.error(traceback.format_exc())
            return False

    def run_forever(self):
        """持续运行，直到达到 max_cycles / max_hours 或用户中断"""
        self.logger.info("========== 启动自动化因子挖掘系统 ==========")
        self.logger.info(f"配置：{self.config}")

        cur_count = 0
        start_time = time.time()

        while True:
            # 停止条件检查
            if self.config.get("max_cycles") is not None and cur_count >= self.config["max_cycles"]:
                self.logger.info(f"[STOP] 已达到最大轮次 {self.config['max_cycles']}")
                break

            elapsed_hours = (time.time() - start_time) / 3600
            if self.config.get("max_hours") is not None and elapsed_hours >= self.config["max_hours"]:
                self.logger.info(
                    f"[STOP] 已运行 {elapsed_hours:.1f}h，超过上限 {self.config['max_hours']}h"
                )
                break

            try:
                success = self.run_single_cycle()
                if success:
                    self.save_checkpoint(self.cycle_count)
                    self.logger.info(f"累计完成 {self.cycle_count} 轮循环")
                    interval = self.config.get("cycle_interval", 0)
                    if interval > 0:
                        self.logger.info(f"等待 {interval}s 后开始下一轮...")
                        time.sleep(interval)
                else:
                    wait = self.config.get("error_wait", 3)
                    self.logger.info(f"本轮失败，等待 {wait}s 后重试...")
                    time.sleep(wait)

            except KeyboardInterrupt:
                self.logger.info("收到中断信号，系统停止")
                break
            except Exception as e:
                self.logger.error(f"意外错误：{e}")
                self.logger.error(traceback.format_exc())
                time.sleep(self.config.get("error_wait", 3))

            cur_count += 1


# ================================================================
# =================== 入口 =======================================
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="自动化 A 股量价因子挖掘系统")
    parser.add_argument("--max_cycles", type=int, default=None, help="最大轮次（覆盖 config）")
    parser.add_argument("--max_hours", type=float, default=None, help="最大运行小时数（覆盖 config）")
    parser.add_argument(
        "--method_config",
        type=str,
        default=None,
        help="挖掘方法 YAML 路径（覆盖 config）",
    )
    parser.add_argument(
        "--validation_config",
        type=str,
        default=None,
        help="验证配置 YAML 路径（覆盖 config）",
    )
    args = parser.parse_args()

    config = dict(STOCK_CONFIG)  # 从 config.py 读取默认配置
    if args.max_cycles is not None:
        config["max_cycles"] = args.max_cycles
    if args.max_hours is not None:
        config["max_hours"] = args.max_hours
    if args.method_config is not None:
        config["method_config"] = args.method_config
    if args.validation_config is not None:
        config["validation_config"] = args.validation_config

    system = AutomatedFactorSystem(config)
    system.run_forever()


if __name__ == "__main__":
    main()
