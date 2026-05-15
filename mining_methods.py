"""
mining_methods.py - 可插拔的因子挖掘方法注册区

将“样本内如何挖掘候选因子”从 run.py 中解耦出来。
当前内置方法包括：

- factor_mad:         参考 FactorMAD 的双 Agent 辩论式挖掘流程
- alpha_agent:        参考 AlphaAgent 的多角色协同流程
- alpha_jungle_mcts:  参考 Alpha Jungle 的 MCTS + FSA 搜索流程
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from math import cos, log, pi, sqrt

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from evaluator import cal_alpha
from factor_mining import (
    calculate_factors_performance,
    is_low_correlated_with_fixed_factors,
)
from llm_client import (
    contains_future_function,
    contains_too_many_fields,
    convert_fields_to_lowercase,
    extract_string,
    query,
)
from method_utils import (
    EvaluatedCandidate,
    clipped_ratio,
    compute_complexity_features,
    compute_complexity_score,
    compute_hypothesis_alignment_score,
    compute_turnover_proxy,
    expression_contains_subtree,
    max_similarity_to_records,
    mine_frequent_subtrees,
    percentile_rank_score,
    safe_float,
    softmax_choice_index,
)


METHOD_REGISTRY: dict = {}


def register_method(name: str):
    def decorator(cls):
        METHOD_REGISTRY[name] = cls
        return cls

    return decorator


def build_mining_method(system, method_profile: dict):
    method_name = method_profile.get("method", "factor_mad")
    cls = METHOD_REGISTRY.get(method_name)
    if cls is None:
        raise ValueError(
            f"未知 mining method: '{method_name}'，可选: {list(METHOD_REGISTRY.keys())}"
        )
    return cls(system, method_profile)


class MiningMethodBase:
    def __init__(self, system, method_profile: dict):
        self.system = system
        self.profile = method_profile or {}
        self.params = self.profile.get("params") or {}

    def mine_in_sample(self, cycle_timestamp: str) -> pd.DataFrame:
        raise NotImplementedError


class StructuredLLMMethodBase(MiningMethodBase):
    """给更复杂的论文方法提供共用的表达式校验与候选后处理。"""

    _FULLWIDTH_TRANSLATION = str.maketrans(
        {
            "（": "(",
            "）": ")",
            "，": ",",
            "：": ":",
            "；": ";",
            "＝": "=",
            "＋": "+",
            "－": "-",
            "—": "-",
            "＊": "*",
            "／": "/",
            "＜": "<",
            "＞": ">",
            "！": "!",
            "？": "?",
            "【": "[",
            "】": "]",
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
        }
    )
    _EXPR_OUTPUT_NAMES = (
        "final_signal",
        "factor",
        "alpha",
        "signal",
        "filtered_signal",
        "signal_weighted",
        "factor_raw",
        "raw_signal",
    )

    def _extract_sections(self, text: str, labels: list[str]) -> dict[str, str]:
        sections = {}
        for idx, label in enumerate(labels):
            next_labels = labels[idx + 1 :]
            if next_labels:
                lookahead = "|".join(re.escape(x) for x in next_labels)
                pattern = rf"{re.escape(label)}\s*:\s*(.*?)(?=\n(?:{lookahead})\s*:|\Z)"
            else:
                pattern = rf"{re.escape(label)}\s*:\s*(.*)$"
            match = re.search(pattern, text or "", flags=re.IGNORECASE | re.DOTALL)
            sections[label] = match.group(1).strip() if match else ""
        return sections

    def _sanitize_expression_text(self, expression: str) -> str:
        expression = (expression or "").strip()
        if not expression:
            return ""

        expression = expression.translate(self._FULLWIDTH_TRANSLATION)
        expression = expression.replace("\r\n", "\n").replace("\r", "\n")
        expression = re.sub(r"```[a-zA-Z0-9_+-]*", "", expression)
        expression = expression.replace("```", "")

        statements = []
        for raw_line in expression.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = re.sub(r"^expression\s*:\s*", "", line, flags=re.IGNORECASE)
            line = re.sub(r"//.*$", "", line).strip()
            line = re.sub(r"#.*$", "", line).strip()
            if not line:
                continue
            for chunk in line.split(";"):
                chunk = chunk.strip()
                if chunk:
                    statements.append(chunk)

        if not statements:
            return ""

        if len(statements) == 1 and "=" not in statements[0]:
            return re.sub(r"\s+", " ", statements[0]).strip().strip(";")

        assignments = {}
        assignment_order = []
        terminal_expr = ""

        for statement in statements:
            match = re.match(r"^([A-Za-z_]\w*)\s*=\s*(.+)$", statement)
            if match:
                name, rhs = match.groups()
                assignments[name] = rhs.strip()
                assignment_order.append(name)
            else:
                terminal_expr = statement

        if terminal_expr:
            candidate = terminal_expr
        elif assignments:
            output_name = next(
                (name for name in self._EXPR_OUTPUT_NAMES if name in assignments),
                assignment_order[-1],
            )
            candidate = assignments[output_name]
        else:
            return ""

        reserved = set(self.system.fields) | set(self.system.ops_info.keys()) | {
            "True",
            "False",
            "None",
            "and",
            "or",
            "not",
        }

        def expand(expr: str, stack: tuple[str, ...] = ()) -> str:
            expanded = expr
            for name in sorted(assignments.keys(), key=len, reverse=True):
                if name in reserved or name in stack:
                    continue
                if not re.search(rf"\b{name}\b", expanded):
                    continue
                replacement = expand(assignments[name], stack + (name,))
                expanded = re.sub(rf"\b{name}\b", f"({replacement})", expanded)
            return expanded

        candidate = expand(candidate)
        candidate = re.sub(r"\s+", " ", candidate).strip().strip(";")
        return candidate

    def _sample_reference_examples(self, rng: random.Random, count: int) -> list[dict]:
        records = list(self.system.baseline_factor_records)
        if not records or count <= 0:
            return []
        count = min(count, len(records))
        if self.params.get("prefer_top_baselines", True):
            records = sorted(
                records,
                key=lambda x: abs(safe_float(x.get("ic_mean"), 0.0)),
                reverse=True,
            )
            pool = records[: max(count * 3, count)]
            return rng.sample(pool, count)
        return rng.sample(records, count)

    def _format_reference_examples(self, examples: list[dict]) -> str:
        if not examples:
            return "当前没有可用历史示例，请自主提出结构清晰的新方案。"
        blocks = []
        for idx, rec in enumerate(examples, start=1):
            explanation = str(rec.get("explanation", "")).split("=====")[0].strip()
            blocks.append(
                f"[示例{idx}] alpha_id={rec.get('alpha_id')} | expression={rec.get('expression')} | "
                f"ic_mean={safe_float(rec.get('ic_mean'), np.nan):.4f} | "
                f"long_excret={safe_float(rec.get('long_excret'), np.nan):.4f}\n"
                f"解释: {explanation[:180]}"
            )
        return "\n".join(blocks)

    def _performance_score(self, metrics: dict) -> float:
        ic_score = clipped_ratio(abs(safe_float(metrics.get("ic_mean"), 0.0)), self.params.get("ic_scale", 0.05))
        ic_ir_score = clipped_ratio(max(0.0, safe_float(metrics.get("ic_ir"), 0.0)), self.params.get("ic_ir_scale", 0.5))
        long_score = clipped_ratio(max(0.0, safe_float(metrics.get("long_excret"), 0.0)), self.params.get("long_ret_scale", 0.08))
        return 0.45 * ic_score + 0.25 * ic_ir_score + 0.30 * long_score

    def _evaluate_expression_candidate(
        self,
        alpha_id: str,
        expression: str,
        explanation: str,
        extra: dict | None = None,
        similarity_records: list[dict] | None = None,
        similarity_cap: float | None = None,
        enforce_baseline_diversity: bool | None = None,
    ) -> tuple[EvaluatedCandidate | None, list[str]]:
        errors = []
        expression = (expression or "").strip()
        if not expression:
            return None, ["OUTPUT FORMAT: 未找到有效表达式。"]

        expression = convert_fields_to_lowercase(expression, self.system.fields)
        expression = expression.replace("'", "")

        if contains_future_function(expression):
            errors.append("LOOK-AHEAD BIAS: 检测到未来函数或负窗口。")

        max_operator_count = int(self.params.get("max_operator_count", 12))
        if contains_too_many_fields(expression, list(self.system.ops_info.keys()), max_fields=max_operator_count):
            errors.append("COMPLEXITY: 表达式算子过多，超出当前方法上限。")

        complexity = compute_complexity_features(expression, self.system.fields)
        max_similarity, similar_record = max_similarity_to_records(expression, similarity_records or [])
        if similarity_cap is not None and max_similarity >= similarity_cap:
            errors.append(
                f"ORIGINALITY: 与已有 alpha 的 AST 相似度 {max_similarity:.2f} 过高，"
                f"最相近的是 {similar_record.get('alpha_id') if similar_record else 'unknown'}。"
            )

        factor_values = None
        non_null_ratio = np.nan
        if not errors:
            try:
                factor_values = self.system.calculator.calculate(expression)
                factor_values = factor_values.replace([np.inf, -np.inf], np.nan)
            except Exception as e:
                errors.append(f"EXECUTION: 表达式计算失败 - {e}")

        if factor_values is not None:
            total = max(1, factor_values.shape[0] * factor_values.shape[1])
            non_null_ratio = factor_values.notna().sum().sum() / total
            if non_null_ratio < float(self.params.get("min_non_null_ratio", 0.05)):
                errors.append("OUTPUT NAN: 有效值过少，因子过于稀疏。")

        perf = None
        if factor_values is not None and not errors:
            try:
                perf = calculate_factors_performance(
                    {alpha_id: factor_values},
                    [expression],
                    self.system.factor_dfs["close"],
                    start_date=self.system.config["start_date"],
                    end_date=self.system.config["end_date"],
                    explanation_list=[explanation],
                    poolsel_path=self.system.validation_profile.get("poolsel_path"),
                )[0]
            except Exception as e:
                errors.append(f"EVALUATION: 训练期评估失败 - {e}")

        if perf is not None and np.isnan(safe_float(perf.get("ic_mean"), np.nan)):
            errors.append("EVALUATION: 训练期 IC 无效。")

        should_check_diversity = (
            self.params.get("enforce_baseline_diversity", True)
            if enforce_baseline_diversity is None
            else enforce_baseline_diversity
        )
        if (
            should_check_diversity
            and factor_values is not None
            and not errors
            and self.system.baseline_factors
        ):
            low_corr = is_low_correlated_with_fixed_factors(
                factor_values,
                self.system.baseline_factors,
                self.system.config["correlation_threshold"],
            )
            if not low_corr:
                errors.append("DIVERSITY: 与已有 baseline 因子相关性过高。")

        if errors or perf is None:
            return None, errors

        payload = dict(extra or {})
        payload.update(
            {
                "max_ast_similarity": max_similarity,
                "most_similar_alpha_id": similar_record.get("alpha_id") if similar_record else "",
                "complexity_symbolic_length": complexity["symbolic_length"],
                "complexity_operator_count": complexity["operator_count"],
                "complexity_parameter_count": complexity["parameter_count"],
                "complexity_feature_count": complexity["feature_count"],
                "complexity_depth": complexity["depth"],
                "non_null_ratio": non_null_ratio,
            }
        )

        metrics = {k: perf.get(k) for k in perf.keys()}
        return (
            EvaluatedCandidate(
                alpha_id=alpha_id,
                expression=expression,
                explanation=explanation,
                metrics=metrics,
                extra=payload,
                factor_values=factor_values,
            ),
            [],
        )

    def _build_candidate_df(self, candidates: list[EvaluatedCandidate], phase_label: str) -> pd.DataFrame:
        if not candidates:
            return pd.DataFrame()
        df = pd.DataFrame([candidate.to_record() for candidate in candidates])
        train_filter = self.system.validation_profile.get("train_filter", [])
        if train_filter:
            factor_values = {candidate.alpha_id: candidate.factor_values for candidate in candidates}
            df = self.system.apply_metric_filters(
                df,
                factor_values,
                train_filter,
                self.system.config["start_date"],
                self.system.config["end_date"],
                phase_label,
                runtime_context={
                    "close": self.system.factor_dfs["close"],
                    "poolsel_path": self.system.validation_profile.get("poolsel_path"),
                },
            )
        return df

@register_method("alpha_agent")
class AlphaAgentMethod(StructuredLLMMethodBase):
    """
    AlphaAgent 风格实现。

    保留论文中最适合当前框架的三点：
    1. idea / factor / eval 三角色分工
    2. originality / alignment / complexity 正则
    3. 评估反馈驱动的闭环修正

    仍然沿用当前 DSL 执行层，因此不会破坏 evaluator / validation / save 流程。
    """

    HYPOTHESIS_LABELS = ["observation", "knowledge", "justification", "specification"]

    def mine_in_sample(self, cycle_timestamp: str) -> pd.DataFrame:
        idea_count = int(self.params.get("ideas_per_cycle", self.system.config["factors_per_cycle"]))
        n_jobs = int(self.params.get("n_jobs", self.system.config["n_jobs"]))

        self.system.logger.info(
            f"AlphaAgent 样本内挖掘启动：ideas={idea_count}, "
            f"candidates_per_idea={self.params.get('candidates_per_idea', 1)}, "
            f"max_refine_rounds={self.params.get('max_refine_rounds', 1)}"
        )

        candidates = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
            delayed(self._run_single_idea)(idea_idx, cycle_timestamp)
            for idea_idx in tqdm(range(idea_count), desc="AlphaAgent 挖掘")
        )
        candidates = [candidate for candidate in candidates if candidate is not None]
        if not candidates:
            self.system.logger.warning("AlphaAgent 未生成任何有效候选因子")
            return pd.DataFrame()

        candidates_df = self._build_candidate_df(candidates, "AlphaAgent 训练期")
        if candidates_df.empty:
            return pd.DataFrame()

        if self.params.get("apply_mmr", True):
            final_df, _ = self.system.alpha_mmr_selection(candidates_df, "AlphaAgent")
            return final_df

        rank_metric = self.params.get("rank_metric", "regularized_score")
        top_n = int(self.params.get("top_n", self.system.config["factors_per_cycle"]))
        if rank_metric in candidates_df.columns:
            return candidates_df.sort_values(rank_metric, ascending=False).head(top_n).copy()
        return candidates_df.head(top_n).copy()

    def _run_single_idea(self, idea_idx: int, cycle_timestamp: str) -> EvaluatedCandidate | None:
        rng = random.Random(f"{cycle_timestamp}-alpha-agent-{idea_idx}")
        examples = self._sample_reference_examples(rng, int(self.params.get("example_count", 3)))

        hypothesis = self._generate_hypothesis(idea_idx, examples)
        if not hypothesis:
            return None

        best_candidate = None
        best_payload = None
        candidate_count = max(1, int(self.params.get("candidates_per_idea", 1)))
        for candidate_idx in range(candidate_count):
            candidate, payload = self._generate_factor_candidate(
                hypothesis=hypothesis,
                examples=examples,
                alpha_id=f"alpha-{cycle_timestamp}_AA_i{idea_idx:03d}_c{candidate_idx:02d}",
            )
            if candidate is None:
                continue
            if (
                best_candidate is None
                or safe_float(candidate.extra.get("regularized_score"), -np.inf)
                > safe_float(best_candidate.extra.get("regularized_score"), -np.inf)
            ):
                best_candidate = candidate
                best_payload = payload

        if best_candidate is None:
            return None

        feedback = self._generate_eval_feedback(hypothesis, best_candidate, best_payload or {})
        best_candidate.extra.update(
            {
                "eval_assessment": feedback.get("assessment", ""),
                "eval_improvement": feedback.get("improvement", ""),
            }
        )

        need_refine = (
            safe_float(best_candidate.extra.get("regularized_score"), 0.0)
            < float(self.params.get("min_regularized_score", 0.45))
            or safe_float(best_candidate.extra.get("originality_score"), 1.0)
            < float(self.params.get("min_originality_score", 0.25))
            or safe_float(best_candidate.extra.get("alignment_score"), 1.0)
            < float(self.params.get("min_alignment_score", 0.50))
        )

        if need_refine:
            refined = self._refine_candidate(
                hypothesis=hypothesis,
                base_candidate=best_candidate,
                base_payload=best_payload or {},
                feedback=feedback,
                examples=examples,
                alpha_id=f"alpha-{cycle_timestamp}_AA_i{idea_idx:03d}_refine",
            )
            if (
                refined is not None
                and safe_float(refined.extra.get("regularized_score"), -np.inf)
                >= safe_float(best_candidate.extra.get("regularized_score"), -np.inf)
            ):
                best_candidate = refined

        min_score = float(self.params.get("min_regularized_score", 0.45))
        if safe_float(best_candidate.extra.get("regularized_score"), 0.0) < min_score:
            return None
        return best_candidate

    def _generate_hypothesis(self, idea_idx: int, examples: list[dict]) -> dict | None:
        prompt = self._build_hypothesis_prompt(idea_idx, examples)
        max_rounds = int(self.params.get("idea_correction_rounds", 1))
        current_prompt = prompt

        for _ in range(max_rounds + 1):
            answer, _ = query(current_prompt)
            sections = self._extract_sections(answer, self.HYPOTHESIS_LABELS)
            filled = sum(bool(v) for v in sections.values())
            if filled >= 3:
                sections["full_text"] = "\n".join(
                    f"{label}: {sections[label]}" for label in self.HYPOTHESIS_LABELS if sections.get(label)
                )
                sections["raw_answer"] = answer
                return sections
            current_prompt = f"""
你是 AlphaAgent 的 idea agent。
上一版 hypothesis 不够结构化，请只修复输出格式，不要直接输出因子表达式。

上一版回答：
{answer}

请严格按如下格式重新输出：
observation: 你观察到的量价现象
knowledge: 相关金融直觉或经验规律
justification: 为什么这个现象可能有预测力
specification: 建议重点使用的字段、窗口或实现约束
""".strip()

        return None

    def _generate_factor_candidate(
        self,
        hypothesis: dict,
        examples: list[dict],
        alpha_id: str,
        refinement_feedback: dict | None = None,
        base_candidate: EvaluatedCandidate | None = None,
    ) -> tuple[EvaluatedCandidate | None, dict | None]:
        prompt = self._build_factor_prompt(hypothesis, examples, refinement_feedback, base_candidate)
        max_corrections = int(self.params.get("max_correction_rounds", 2))
        current_prompt = prompt
        payload = None
        last_errors = []

        for attempt_idx in range(max_corrections + 1):
            answer, think = query(current_prompt)
            payload = self._parse_factor_response(answer, think)
            description = payload.get("description") or payload.get("summary") or ""
            explanation = payload.get("explanation") or description
            candidate, errors = self._evaluate_expression_candidate(
                alpha_id=alpha_id,
                expression=payload.get("expression", ""),
                explanation=explanation,
                similarity_records=self.system.baseline_factor_records,
                similarity_cap=self.params.get("originality_similarity_cap"),
                extra={
                    "method_name": "alpha_agent",
                    "factor_description": description,
                    "hypothesis_observation": hypothesis.get("observation", ""),
                    "hypothesis_knowledge": hypothesis.get("knowledge", ""),
                    "hypothesis_justification": hypothesis.get("justification", ""),
                    "hypothesis_specification": hypothesis.get("specification", ""),
                },
            )
            if candidate is not None:
                self._attach_alphaagent_scores(candidate, hypothesis, payload)
                return candidate, payload
            last_errors = errors
            expr_preview = payload.get("expression") or payload.get("expression_raw") or ""
            expr_preview = re.sub(r"\s+", " ", expr_preview)[:220]
            self.system.logger.info(
                f"AlphaAgent candidate rejected: alpha_id={alpha_id}, "
                f"attempt={attempt_idx + 1}/{max_corrections + 1}, "
                f"errors={' | '.join(errors[:3])}, expression={expr_preview}"
            )
            current_prompt = self._build_alphaagent_correction_prompt(prompt, answer, errors)

        if last_errors:
            self.system.logger.warning(
                f"AlphaAgent candidate failed after {max_corrections + 1} attempts: "
                f"alpha_id={alpha_id}, reasons={' | '.join(last_errors[:4])}"
            )
        return None, payload

    def _attach_alphaagent_scores(self, candidate: EvaluatedCandidate, hypothesis: dict, payload: dict):
        complexity = {
            "symbolic_length": candidate.extra.get("complexity_symbolic_length", 0),
            "parameter_count": candidate.extra.get("complexity_parameter_count", 0),
            "feature_count": candidate.extra.get("complexity_feature_count", 0),
            "depth": candidate.extra.get("complexity_depth", 0),
        }
        complexity_score = compute_complexity_score(
            complexity,
            max_symbolic_length=int(self.params.get("max_symbolic_length", 28)),
            max_free_params=int(self.params.get("max_free_params", 6)),
            max_features=int(self.params.get("max_features", 5)),
            max_depth=int(self.params.get("max_depth", 7)),
        )
        alignment_score = compute_hypothesis_alignment_score(
            hypothesis.get("full_text", ""),
            f"{payload.get('description', '')}\n{candidate.explanation}",
            candidate.expression,
            self.system.fields,
        )
        originality_score = max(0.0, 1.0 - safe_float(candidate.extra.get("max_ast_similarity"), 0.0))
        performance_score = self._performance_score(candidate.metrics)

        weights = {
            "performance": float(self.params.get("performance_weight", 0.40)),
            "originality": float(self.params.get("originality_weight", 0.25)),
            "alignment": float(self.params.get("alignment_weight", 0.20)),
            "complexity": float(self.params.get("complexity_weight", 0.15)),
        }
        total_weight = sum(weights.values()) or 1.0
        regularized_score = (
            weights["performance"] * performance_score
            + weights["originality"] * originality_score
            + weights["alignment"] * alignment_score
            + weights["complexity"] * complexity_score
        ) / total_weight

        candidate.extra.update(
            {
                "performance_score": performance_score,
                "originality_score": originality_score,
                "alignment_score": alignment_score,
                "complexity_score": complexity_score,
                "regularized_score": regularized_score,
            }
        )

    def _generate_eval_feedback(self, hypothesis: dict, candidate: EvaluatedCandidate, payload: dict) -> dict:
        prompt = self._build_eval_prompt(hypothesis, candidate, payload)
        answer, _ = query(prompt)
        sections = self._extract_sections(answer, ["assessment", "improvement"])
        if not sections.get("assessment") and not sections.get("improvement"):
            sections["assessment"] = answer.strip()
        return sections

    def _refine_candidate(
        self,
        hypothesis: dict,
        base_candidate: EvaluatedCandidate,
        base_payload: dict,
        feedback: dict,
        examples: list[dict],
        alpha_id: str,
    ) -> EvaluatedCandidate | None:
        max_refine_rounds = int(self.params.get("max_refine_rounds", 1))
        best_candidate = base_candidate
        current_payload = base_payload
        current_feedback = feedback

        for _ in range(max_refine_rounds):
            refined, payload = self._generate_factor_candidate(
                hypothesis=hypothesis,
                examples=examples,
                alpha_id=alpha_id,
                refinement_feedback=current_feedback,
                base_candidate=best_candidate,
            )
            if refined is None:
                break
            refined_feedback = self._generate_eval_feedback(hypothesis, refined, payload or {})
            refined.extra.update(
                {
                    "eval_assessment": refined_feedback.get("assessment", ""),
                    "eval_improvement": refined_feedback.get("improvement", ""),
                }
            )
            if safe_float(refined.extra.get("regularized_score"), -np.inf) >= safe_float(
                best_candidate.extra.get("regularized_score"), -np.inf
            ):
                best_candidate = refined
                current_payload = payload
                current_feedback = refined_feedback
            else:
                break

        return best_candidate

    def _parse_factor_response(self, answer: str, think: str) -> dict:
        sections = self._extract_sections(answer, ["summary", "description", "expression", "explanation"])
        expression_raw = extract_string(answer, "alpha")
        if not isinstance(expression_raw, str) or not expression_raw.strip():
            expression_raw = sections.get("expression", "")
        expression = self._sanitize_expression_text(expression_raw)
        return {
            "summary": sections.get("summary", ""),
            "description": sections.get("description", "") or sections.get("summary", ""),
            "expression": expression.strip(),
            "expression_raw": str(expression_raw).strip(),
            "explanation": sections.get("explanation", "") or think.strip(),
            "raw_answer": answer,
        }

    def _build_hypothesis_prompt(self, idea_idx: int, examples: list[dict]) -> str:
        return f"""
你是 AlphaAgent 的 idea agent，负责提出一个新的市场假设，用来指导后续 factor agent 生成量价因子。

研究目标：
- 在 A 股 OHLCV / VWAP 数据上发掘不容易快速拥挤的量价 alpha。
- 需要兼顾金融直觉、原创性和后续可实现性。

历史参考因子：
{self._format_reference_examples(examples)}

要求：
- 暂时不要写表达式。
- 只提出一个结构清晰、可落到量价 DSL 的市场假设。
- 假设最好能说明观察现象、相关知识、为什么可能有效、以及实现边界。

请严格按如下格式输出：
observation: 你观察到的量价现象
knowledge: 支撑该假设的经验规律或金融直觉
justification: 为什么该现象可能对未来收益有预测力
specification: 建议重点使用的字段、窗口或实现约束
""".strip()

    def _build_factor_prompt(
        self,
        hypothesis: dict,
        examples: list[dict],
        refinement_feedback: dict | None,
        base_candidate: EvaluatedCandidate | None,
    ) -> str:
        refine_block = ""
        if refinement_feedback:
            refine_block = f"""
上一轮 eval agent 的反馈：
assessment: {refinement_feedback.get('assessment', '')}
improvement: {refinement_feedback.get('improvement', '')}

待改进的上一版因子：
expression: {base_candidate.expression if base_candidate else ''}
regularized_score: {safe_float(base_candidate.extra.get('regularized_score'), np.nan) if base_candidate else np.nan:.4f}
""".strip()

        return f"""
你是 AlphaAgent 的 factor agent，需要把 market hypothesis 转成一个新的量价因子表达式。

当前 hypothesis：
observation: {hypothesis.get('observation', '')}
knowledge: {hypothesis.get('knowledge', '')}
justification: {hypothesis.get('justification', '')}
specification: {hypothesis.get('specification', '')}

历史参考因子：
{self._format_reference_examples(examples)}

{refine_block}

可用字段：
{', '.join(self.system.fields)}

可用算子：
{', '.join(sorted(self.system.ops_info.keys()))}

可用窗口：
{self.system.config['window']}

硬性要求：
- 表达式必须可被当前 DSL 直接计算。
- 禁止未来函数和负窗口。
- 最终信号要可跨股票比较，尽量无量纲。
- 不要只把历史因子改一个窗口数字。
- 尽量让描述、经济直觉和表达式真正对齐。
- `expression` 必须是单行最终表达式，不要输出多行程序。
- 禁止代码块、注释、分号、变量赋值语句，如 `x = ...`、`// ...`、```。
- 禁止全角符号，括号和逗号一律使用 ASCII 字符：`() , + - * / < >`。
- 如果你需要中间步骤，请直接在最终表达式里内联展开。
- 请务必真的写成 `expression: <...>` 的形式，尖括号里只放最终表达式本身。

请严格按如下格式输出：
summary: 这一版因子想抓什么机制
description: 用自然语言描述这个因子的实现逻辑
expression: <一个可计算的因子表达式>
explanation: 用通俗中文解释为什么这个实现符合 hypothesis
""".strip()

    def _build_eval_prompt(self, hypothesis: dict, candidate: EvaluatedCandidate, payload: dict) -> str:
        return f"""
你是 AlphaAgent 的 eval agent，需要评估当前因子是否真的符合 hypothesis，并指出下一步改进方向。

hypothesis:
observation: {hypothesis.get('observation', '')}
knowledge: {hypothesis.get('knowledge', '')}
justification: {hypothesis.get('justification', '')}
specification: {hypothesis.get('specification', '')}

candidate:
description: {payload.get('description', '')}
expression: {candidate.expression}
explanation: {candidate.explanation}

backtest summary:
ic_mean={safe_float(candidate.metrics.get('ic_mean'), np.nan):.4f}
ic_ir={safe_float(candidate.metrics.get('ic_ir'), np.nan):.4f}
long_excret={safe_float(candidate.metrics.get('long_excret'), np.nan):.4f}

regularization summary:
originality_score={safe_float(candidate.extra.get('originality_score'), np.nan):.4f}
alignment_score={safe_float(candidate.extra.get('alignment_score'), np.nan):.4f}
complexity_score={safe_float(candidate.extra.get('complexity_score'), np.nan):.4f}
regularized_score={safe_float(candidate.extra.get('regularized_score'), np.nan):.4f}
most_similar_alpha_id={candidate.extra.get('most_similar_alpha_id', '')}

请严格按如下格式输出：
assessment: 从金融逻辑、原创性和稳健性角度评价这版因子
improvement: 下一轮最值得改进的一点，并说明应该怎么改
""".strip()

    def _build_alphaagent_correction_prompt(self, original_prompt: str, last_response: str, errors: list[str]) -> str:
        error_text = "\n".join(f"- {error}" for error in errors)
        return f"""
你是 AlphaAgent 的 correction agent。
上一版 factor agent 回答没有通过系统校验，请根据错误反馈修复，但尽量保留原有 market hypothesis。

原始任务：
{original_prompt}

上一版回答：
{last_response}

错误反馈：
{error_text}

再次强调：
- `expression` 只能有一行。
- 不能有代码块、注释、分号、赋值语句或中间变量。
- 不能写 `a = ...` 再写 `final_signal = ...`。
- 只能直接给最终 DSL 表达式，并放在尖括号 `<...>` 内。
- 全部符号必须是 ASCII，不能使用全角括号、全角逗号、全角冒号。

请继续严格按如下格式输出：
summary: 修复后这一版因子想抓什么机制
description: 修复后的实现逻辑描述
expression: <修复后的因子表达式>
explanation: 用通俗中文解释修复思路
""".strip()


@dataclass
class MCTSSearchNode:
    node_id: str
    candidate: EvaluatedCandidate
    reward: float
    dimension_scores: dict[str, float]
    target_dimension: str = ""
    refinement_suggestion: str = ""
    parent: MCTSSearchNode | None = None
    depth: int = 0
    visits: int = 1
    q_value: float = 0.0
    children: list[MCTSSearchNode] = field(default_factory=list)

    def __post_init__(self):
        if not self.q_value:
            self.q_value = self.reward

    def to_dict(self) -> dict:
        """将 MCTS 节点及其子树递归序列化为纯 Python 字典（可 JSON 序列化）。
        用于可视化导出，会跳过 factor_values 等大型数据结构。"""
        def _convert(v):
            if isinstance(v, (float, np.floating)):
                return None if np.isnan(v) or np.isinf(v) else float(v)
            if isinstance(v, np.integer):
                return int(v)
            return v

        def _candidate_dict(c):
            metrics = {k: _convert(v) for k, v in (c.metrics or {}).items()}
            extra = {}
            for k, v in (c.extra or {}).items():
                if k == "factor_values":
                    continue
                extra[k] = _convert(v) if not isinstance(v, str) else v
            return {
                "alpha_id": c.alpha_id,
                "expression": c.expression,
                "explanation": c.explanation,
                "metrics": metrics,
                "extra": extra,
            }

        return {
            "node_id": self.node_id,
            "depth": self.depth,
            "reward": _convert(self.reward),
            "q_value": _convert(self.q_value),
            "visits": self.visits,
            "target_dimension": self.target_dimension,
            "refinement_suggestion": self.refinement_suggestion,
            "candidate": _candidate_dict(self.candidate),
            "dimension_scores": {k: _convert(v) for k, v in (self.dimension_scores or {}).items()},
            "parent_id": self.parent.node_id if self.parent else None,
            "children": [child.to_dict() for child in self.children],
        }


@register_method("alpha_jungle_mcts")
class AlphaJungleMCTSMethod(StructuredLLMMethodBase):
    """
    Alpha Jungle / MCTS 论文的适配实现。

    保留论文中的：
    1. MCTS 选择 - 扩展 - 回传框架
    2. 多维反馈引导的 refinement
    3. Frequent Subtree Avoidance (FSA)

    同样继续落在当前 DSL 上，方便与现有 evaluator / validation 无缝衔接。
    """

    DEFAULT_DIMENSIONS = ["effectiveness", "diversity", "stability", "turnover", "overfit"]

    def mine_in_sample(self, cycle_timestamp: str) -> pd.DataFrame:
        self.system.logger.info(
            f"AlphaJungleMCTS 样本内挖掘启动：search_budget={self.params.get('search_budget', 24)}, "
            f"uct_c={self.params.get('uct_c', 0.7)}, "
            f"top_forbidden_subtrees={self.params.get('top_forbidden_subtrees', 5)}"
        )

        rng = random.Random(f"{cycle_timestamp}-alpha-jungle")
        repository = self._initial_repository_records()
        root = self._generate_root_node(cycle_timestamp, rng, repository)
        if root is None:
            self.system.logger.warning("AlphaJungleMCTS 根节点生成失败")
            return pd.DataFrame()

        nodes = [root]
        if self._qualifies_for_repository(root):
            repository.append(root.candidate.to_record())

        search_budget = max(1, int(self.params.get("search_budget", self.system.config["factors_per_cycle"] * 3)))

        if self.params.get("adaptive_uct", False):
            self.system.logger.info(
                f"自适应 UCT 已启用：C_max={self.params.get('uct_c_max', 1.2)}, "
                f"C_min={self.params.get('uct_c_min', 0.3)}, "
                f"decay={self.params.get('uct_decay', 'linear')}"
            )

        for step in tqdm(range(1, search_budget), desc="AlphaJungle MCTS"):
            parent = self._select_node(nodes, step=step, total_steps=search_budget)
            if parent is None:
                break

            target_dimension = self._sample_target_dimension(parent.dimension_scores)
            forbidden = mine_frequent_subtrees(
                repository,
                self.system.fields,
                top_k=int(self.params.get("top_forbidden_subtrees", 5)),
                min_size=int(self.params.get("min_forbidden_subtree_size", 3)),
            )
            suggestion = self._generate_refinement_suggestion(parent, target_dimension, forbidden, repository)
            child = self._expand_node(
                parent=parent,
                target_dimension=target_dimension,
                suggestion=suggestion,
                forbidden=forbidden,
                repository=repository,
                cycle_timestamp=cycle_timestamp,
                step=step,
            )
            if child is None:
                continue

            parent.children.append(child)
            nodes.append(child)
            self._backpropagate(child)
            if self._qualifies_for_repository(child):
                repository.append(child.candidate.to_record())

        # 保存搜索树根节点，供 tree_viz.py 可视化导出
        self._search_tree_root = root

        ranked_nodes = sorted(nodes[1:] or nodes, key=lambda node: node.reward, reverse=True)
        top_n = int(self.params.get("top_n", self.system.config["factors_per_cycle"]))
        candidates = [node.candidate for node in ranked_nodes[:top_n]]
        candidates_df = self._build_candidate_df(candidates, "AlphaJungleMCTS 训练期")
        if candidates_df.empty:
            return pd.DataFrame()

        if self.params.get("apply_mmr", True):
            final_df, _ = self.system.alpha_mmr_selection(candidates_df, "AlphaJungleMCTS")
            return final_df

        rank_metric = self.params.get("rank_metric", "mcts_reward")
        if rank_metric in candidates_df.columns:
            return candidates_df.sort_values(rank_metric, ascending=False).head(top_n).copy()
        return candidates_df.head(top_n).copy()

    def _initial_repository_records(self) -> list[dict]:
        records = [dict(record) for record in self.system.baseline_factor_records if record.get("expression")]
        if not records:
            return []
        records = sorted(records, key=lambda x: abs(safe_float(x.get("ic_mean"), 0.0)), reverse=True)
        seed_count = int(self.params.get("repository_seed_count", 20))
        return records[:seed_count]

    def _generate_root_node(
        self,
        cycle_timestamp: str,
        rng: random.Random,
        repository: list[dict],
    ) -> MCTSSearchNode | None:
        examples = self._sample_reference_examples(rng, int(self.params.get("example_count", 3)))
        prompt = self._build_root_prompt(examples, repository)
        candidate = self._query_formula_candidate(
            prompt=prompt,
            alpha_id=f"alpha-{cycle_timestamp}_MCTS_root",
            repository=repository,
            target_dimension="root",
            suggestion="从空白起点生成一个具备解释性的种子 alpha。",
            forbidden=[],
        )
        if candidate is None:
            return None
        return MCTSSearchNode(
            node_id="root",
            candidate=candidate,
            reward=safe_float(candidate.extra.get("mcts_reward"), 0.0),
            dimension_scores=candidate.extra.get("dimension_scores", {}),
            target_dimension="root",
            refinement_suggestion="seed",
            depth=0,
        )

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

    def _compute_adaptive_uct_c(self, step: int, total_steps: int) -> float:
        """根据搜索进度计算自适应 UCT 探索系数。"""
        if not self.params.get("adaptive_uct", False):
            return float(self.params.get("uct_c", 0.7))

        c_max = float(self.params.get("uct_c_max", 1.2))
        c_min = float(self.params.get("uct_c_min", 0.3))
        decay = self.params.get("uct_decay", "linear")

        progress = min(step / max(total_steps, 1), 1.0)

        if decay == "cosine":
            return c_min + (c_max - c_min) * 0.5 * (1 + cos(pi * progress))
        else:  # linear
            return c_max - (c_max - c_min) * progress

    def _sample_target_dimension(self, dimension_scores: dict[str, float]) -> str:
        dimensions = list(self.params.get("target_dimensions", self.DEFAULT_DIMENSIONS))
        scores = [safe_float(dimension_scores.get(dim), 0.5) for dim in dimensions]
        pressure = [1.0 - max(0.0, min(1.0, score)) for score in scores]
        chosen_idx = softmax_choice_index(pressure, temperature=float(self.params.get("dimension_temperature", 0.45)))
        return dimensions[chosen_idx]

    def _generate_refinement_suggestion(
        self,
        parent: MCTSSearchNode,
        target_dimension: str,
        forbidden: list[dict],
        repository: list[dict],
    ) -> dict:
        prompt = self._build_refinement_prompt(parent, target_dimension, forbidden, repository)
        answer, _ = query(prompt)
        sections = self._extract_sections(answer, ["dimension", "suggestion", "reason"])
        if not sections.get("suggestion"):
            sections["suggestion"] = answer.strip()
        if not sections.get("dimension"):
            sections["dimension"] = target_dimension
        return sections

    def _expand_node(
        self,
        parent: MCTSSearchNode,
        target_dimension: str,
        suggestion: dict,
        forbidden: list[dict],
        repository: list[dict],
        cycle_timestamp: str,
        step: int,
    ) -> MCTSSearchNode | None:
        prompt = self._build_formula_prompt(parent, target_dimension, suggestion, forbidden)
        candidate = self._query_formula_candidate(
            prompt=prompt,
            alpha_id=f"alpha-{cycle_timestamp}_MCTS_{step:03d}",
            repository=repository,
            target_dimension=target_dimension,
            suggestion=suggestion.get("suggestion", ""),
            forbidden=forbidden,
            parent=parent,
        )
        if candidate is None:
            return None
        return MCTSSearchNode(
            node_id=f"node_{step:03d}",
            candidate=candidate,
            reward=safe_float(candidate.extra.get("mcts_reward"), 0.0),
            dimension_scores=candidate.extra.get("dimension_scores", {}),
            target_dimension=target_dimension,
            refinement_suggestion=suggestion.get("suggestion", ""),
            parent=parent,
            depth=parent.depth + 1,
        )

    def _query_formula_candidate(
        self,
        prompt: str,
        alpha_id: str,
        repository: list[dict],
        target_dimension: str,
        suggestion: str,
        forbidden: list[dict],
        parent: MCTSSearchNode | None = None,
    ) -> EvaluatedCandidate | None:
        current_prompt = prompt
        max_corrections = int(self.params.get("max_correction_rounds", 2))

        for _ in range(max_corrections + 1):
            answer, think = query(current_prompt)
            payload = self._parse_formula_response(answer, think)
            candidate, errors = self._evaluate_expression_candidate(
                alpha_id=alpha_id,
                expression=payload.get("expression", ""),
                explanation=payload.get("explanation", "") or payload.get("description", ""),
                similarity_records=repository,
                similarity_cap=self.params.get("repository_similarity_cap"),
                extra={
                    "method_name": "alpha_jungle_mcts",
                    "target_dimension": target_dimension,
                    "refinement_suggestion": suggestion,
                    "factor_description": payload.get("description", ""),
                    "mcts_reason": payload.get("reason", ""),
                },
            )
            if candidate is None:
                current_prompt = self._build_mcts_correction_prompt(prompt, answer, errors)
                continue

            hit_forbidden = [
                item["pattern"]
                for item in forbidden
                if expression_contains_subtree(candidate.expression, item["signature"])
            ]
            if hit_forbidden and self.params.get("enforce_forbidden_subtrees", True):
                current_prompt = self._build_mcts_correction_prompt(
                    prompt,
                    answer,
                    [f"FSA: 命中了需要规避的频繁子树 {', '.join(hit_forbidden)}"],
                )
                continue

            self._attach_mcts_scores(candidate, repository, parent, target_dimension, suggestion, forbidden)
            return candidate

        return None

    def _attach_mcts_scores(
        self,
        candidate: EvaluatedCandidate,
        repository: list[dict],
        parent: MCTSSearchNode | None,
        target_dimension: str,
        suggestion: str,
        forbidden: list[dict],
    ):
        diversity_score = max(0.0, 1.0 - safe_float(candidate.extra.get("max_ast_similarity"), 0.0))
        effectiveness_score = self._performance_score(candidate.metrics)

        repo_ic_irs = [max(0.0, safe_float(record.get("ic_ir"), np.nan)) for record in repository]
        stability_score = percentile_rank_score(
            max(0.0, safe_float(candidate.metrics.get("ic_ir"), 0.0)),
            repo_ic_irs,
            higher_better=True,
        )
        if not repository:
            stability_score = 0.6 * clipped_ratio(
                max(0.0, safe_float(candidate.metrics.get("ic_ir"), 0.0)),
                self.params.get("mcts_ic_ir_scale", 0.6),
            ) + 0.4 * clipped_ratio(
                max(0.0, safe_float(candidate.metrics.get("long_sharpe"), 0.0)),
                self.params.get("mcts_sharpe_scale", 1.2),
            )

        turnover_proxy = compute_turnover_proxy(candidate.factor_values)
        repo_turnovers = [safe_float(record.get("turnover_proxy"), np.nan) for record in repository]
        turnover_score = percentile_rank_score(turnover_proxy, repo_turnovers, higher_better=False)
        if not repository or np.isnan(turnover_proxy):
            scale = float(self.params.get("turnover_scale", 1.5))
            turnover_score = 0.5 if np.isnan(turnover_proxy) else max(0.0, 1.0 - min(turnover_proxy / scale, 1.0))

        complexity = {
            "symbolic_length": candidate.extra.get("complexity_symbolic_length", 0),
            "parameter_count": candidate.extra.get("complexity_parameter_count", 0),
            "feature_count": candidate.extra.get("complexity_feature_count", 0),
            "depth": candidate.extra.get("complexity_depth", 0),
        }
        complexity_score = compute_complexity_score(
            complexity,
            max_symbolic_length=int(self.params.get("max_symbolic_length", 28)),
            max_free_params=int(self.params.get("max_free_params", 6)),
            max_features=int(self.params.get("max_features", 5)),
            max_depth=int(self.params.get("max_depth", 7)),
        )
        parent_similarity = 0.0
        if parent is not None:
            parent_similarity, _ = max_similarity_to_records(
                candidate.expression,
                [{"expression": parent.candidate.expression}],
            )
        overfit_score = max(
            0.0,
            min(
                1.0,
                0.35 * complexity_score
                + 0.25 * diversity_score
                + 0.20 * stability_score
                + 0.20 * (1.0 - parent_similarity),
            ),
        )

        dimension_scores = {
            "effectiveness": effectiveness_score,
            "diversity": diversity_score,
            "stability": stability_score,
            "turnover": turnover_score,
            "overfit": overfit_score,
        }
        active_dimensions = list(self.params.get("target_dimensions", self.DEFAULT_DIMENSIONS))
        reward = float(np.mean([dimension_scores.get(dim, 0.0) for dim in active_dimensions]))

        candidate.extra.update(
            {
                "dimension_scores": dimension_scores,
                "mcts_reward": reward,
                "effectiveness_score": effectiveness_score,
                "diversity_score": diversity_score,
                "stability_score": stability_score,
                "turnover_score": turnover_score,
                "turnover_proxy": turnover_proxy,
                "overfit_score": overfit_score,
                "target_dimension": target_dimension,
                "refinement_suggestion": suggestion,
                "forbidden_subtrees": " | ".join(item["pattern"] for item in forbidden),
                "parent_expression": parent.candidate.expression if parent else "",
            }
        )

    def _qualifies_for_repository(self, node: MCTSSearchNode) -> bool:
        return (
            node.reward >= float(self.params.get("effective_reward_threshold", 0.58))
            or abs(safe_float(node.candidate.metrics.get("ic_mean"), 0.0))
            >= float(self.params.get("repo_ic_threshold", 0.015))
        )

    def _backpropagate(self, node: MCTSSearchNode):
        current = node
        reward = node.reward
        while current.parent is not None:
            parent = current.parent
            parent.visits += 1
            parent.q_value = max(parent.q_value, reward)
            current = parent

    def _parse_formula_response(self, answer: str, think: str) -> dict:
        sections = self._extract_sections(answer, ["description", "expression", "explanation", "reason"])
        expression = extract_string(answer, "alpha")
        if not isinstance(expression, str) or not expression.strip():
            expression = sections.get("expression", "")
        return {
            "description": sections.get("description", ""),
            "expression": expression.strip(),
            "explanation": sections.get("explanation", "") or think.strip(),
            "reason": sections.get("reason", ""),
        }

    def _build_root_prompt(self, examples: list[dict], repository: list[dict]) -> str:
        repo_hint = self._format_reference_examples(repository[:3]) if repository else "暂无历史 alpha zoo。"
        return f"""
你正在运行一个 Alpha Jungle / MCTS 因子挖掘流程。
你需要生成一棵搜索树的根节点，也就是一个具备解释性的 seed alpha。

历史有效 alpha 仓库样例：
{repo_hint}

参考示例：
{self._format_reference_examples(examples)}

可用字段：
{', '.join(self.system.fields)}

可用算子：
{', '.join(sorted(self.system.ops_info.keys()))}

可用窗口：
{self.system.config['window']}

要求：
- 表达式必须可由当前 DSL 直接计算。
- 禁止未来函数和负窗口。
- 初始节点优先选择结构清晰、便于后续 refinement 的表达式。

请严格按如下格式输出：
description: 这个 seed alpha 在抓什么量价机制
expression: <一个可计算的因子表达式>
explanation: 用通俗中文解释该 seed 的直觉
reason: 为什么它适合作为搜索树根节点
""".strip()

    def _build_refinement_prompt(
        self,
        parent: MCTSSearchNode,
        target_dimension: str,
        forbidden: list[dict],
        repository: list[dict],
    ) -> str:
        siblings = []
        if parent.parent is not None:
            siblings = [node for node in parent.parent.children if node.node_id != parent.node_id]
        sibling_block = "\n".join(
            f"- expr={node.candidate.expression} | reward={node.reward:.4f} | target={node.target_dimension}"
            for node in siblings[:3]
        ) or "无"
        child_block = "\n".join(
            f"- expr={node.candidate.expression} | reward={node.reward:.4f} | target={node.target_dimension}"
            for node in parent.children[:3]
        ) or "无"
        forbidden_block = ", ".join(item["pattern"] for item in forbidden) or "当前无高频禁用子树"
        repo_block = self._format_reference_examples(repository[:3]) if repository else "暂无有效仓库样例。"
        score_block = ", ".join(f"{k}={v:.3f}" for k, v in parent.dimension_scores.items())

        return f"""
你是 Alpha Jungle 的 refinement planner。
当前树节点需要沿着低分维度继续扩展。

当前节点：
expression: {parent.candidate.expression}
description: {parent.candidate.extra.get('factor_description', '')}
reward: {parent.reward:.4f}
dimension_scores: {score_block}

目标维度：
{target_dimension}

同层兄弟节点：
{sibling_block}

已有子节点：
{child_block}

有效 alpha 仓库样例：
{repo_block}

需要尽量规避的频繁子树：
{forbidden_block}

请严格按如下格式输出：
dimension: 你正在改进的维度
suggestion: 一个面向该维度的具体 refinement 建议
reason: 为什么这个建议有望提升该维度，同时避免同质化
""".strip()

    def _build_formula_prompt(
        self,
        parent: MCTSSearchNode,
        target_dimension: str,
        suggestion: dict,
        forbidden: list[dict],
    ) -> str:
        forbidden_block = ", ".join(item["pattern"] for item in forbidden) or "当前无"
        return f"""
你是 Alpha Jungle 的 formula generator。
你需要把 refinement suggestion 转换成一个新的 DSL 因子表达式。

父节点：
description: {parent.candidate.extra.get('factor_description', '')}
expression: {parent.candidate.expression}
reward: {parent.reward:.4f}

当前要重点改进的维度：
{target_dimension}

refinement suggestion：
{suggestion.get('suggestion', '')}

reason：
{suggestion.get('reason', '')}

需要尽量规避的频繁子树：
{forbidden_block}

可用字段：
{', '.join(self.system.fields)}

可用算子：
{', '.join(sorted(self.system.ops_info.keys()))}

可用窗口：
{self.system.config['window']}

要求：
- 输出的新表达式必须可被 DSL 直接计算。
- 禁止未来函数和负窗口。
- 不要只改一个窗口数字。
- 优先围绕目标维度做实质性改动。

请严格按如下格式输出：
description: 新表达式在抓什么机制
expression: <新的因子表达式>
explanation: 这次 refinement 做了什么改动
reason: 为什么它有望改善 {target_dimension}
""".strip()

    def _build_mcts_correction_prompt(self, original_prompt: str, last_response: str, errors: list[str]) -> str:
        error_text = "\n".join(f"- {error}" for error in errors)
        return f"""
你是 Alpha Jungle 的 correction agent。
上一版输出没有通过验证，请根据错误反馈修复表达式，但保持原来的 refinement 方向。

原始任务：
{original_prompt}

上一版回答：
{last_response}

错误反馈：
{error_text}

请继续严格按如下格式输出：
description: 修复后的机制描述
expression: <修复后的因子表达式>
explanation: 修复思路
reason: 为什么修复后更符合当前目标维度
""".strip()


@dataclass
class DebateCandidate:
    alpha_id: str
    expression: str
    explanation: str
    summary: str = ""
    critique: str = ""
    ic_mean: float = np.nan
    ic_ir: float = np.nan
    long_excret: float = np.nan
    long_sharpe: float = np.nan
    long_ir: float = np.nan
    long_excmdd: float = np.nan
    ls_ret: float = np.nan
    ls_sharpe: float = np.nan
    ls_mdd: float = np.nan

    def to_record(self) -> dict:
        explanation = self.explanation.strip()
        if self.summary or self.critique:
            explanation = (
                f"summary: {self.summary}\n"
                f"critique: {self.critique}\n"
                f"explanation: {explanation}"
            ).strip()
        return {
            "alpha_id": self.alpha_id,
            "expression": self.expression,
            "explanation": explanation,
            "ic_mean": self.ic_mean,
            "ic_ir": self.ic_ir,
            "long_excret": self.long_excret,
            "long_sharpe": self.long_sharpe,
            "long_ir": self.long_ir,
            "long_excmdd": self.long_excmdd,
            "ls_ret": self.ls_ret,
            "ls_sharpe": self.ls_sharpe,
            "ls_mdd": self.ls_mdd,
        }


@register_method("factor_mad")
class FactorMADMethod(MiningMethodBase):
    """
    FactorMAD 风格方法的适配实现。

    说明：
    - 保留论文中的“多 Agent 辩论 + 校正 + 评估 + 入库”思想。
    - 为了与当前执行层统一，因子实现层仍使用现有 DSL 表达式，而不是自由 Python 代码。
    - 这样既能复现“方法结构”，也不会破坏现有 evaluator / validation / save 流程。
    """

    DEFAULT_PERSPECTIVES = {
        "A": "更偏向趋势延续、流动性冲击、参与度确认，优先寻找稳健、单调、实现简洁的信号。",
        "B": "更偏向反转、波动状态切换、稳健归一化和去噪，优先寻找和已有因子不完全同质的新机制。",
    }

    def mine_in_sample(self, cycle_timestamp: str) -> pd.DataFrame:
        chain_count = int(self.params.get("chains_per_cycle", self.system.config["factors_per_cycle"]))
        n_jobs = int(self.params.get("n_jobs", self.system.config["n_jobs"]))
        rows_per_chain = max(1, int(self.params.get("keep_candidates_per_chain", 1)))

        self.system.logger.info(
            f"FactorMAD 样本内挖掘启动：chains={chain_count}, rounds={self.params.get('debate_rounds', 4)}, "
            f"corrections={self.params.get('max_correction_rounds', 2)}"
        )

        results = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
            delayed(self._run_single_chain)(chain_idx, cycle_timestamp, rows_per_chain)
            for chain_idx in tqdm(range(chain_count), desc="FactorMAD 挖掘")
        )

        flat_rows = []
        for chain_rows in results:
            flat_rows.extend(chain_rows or [])

        if not flat_rows:
            self.system.logger.warning("FactorMAD 未生成任何有效候选因子")
            return pd.DataFrame()

        candidates_df = pd.DataFrame(flat_rows)
        train_filter = self.system.validation_profile.get("train_filter", [])
        if train_filter and not candidates_df.empty:
            factor_values = cal_alpha(
                candidates_df["expression"].to_list(),
                candidates_df["alpha_id"].to_list(),
                self.system.calculator,
            )
            candidates_df = self.system.apply_metric_filters(
                candidates_df,
                factor_values,
                train_filter,
                self.system.config["start_date"],
                self.system.config["end_date"],
                "FactorMAD 训练期",
                runtime_context={
                    "close": self.system.factor_dfs["close"],
                    "poolsel_path": self.system.validation_profile.get("poolsel_path"),
                },
            )

        if candidates_df.empty:
            return pd.DataFrame()

        if self.params.get("apply_mmr", True):
            final_df, _ = self.system.alpha_mmr_selection(candidates_df, "FactorMAD")
            return final_df

        rank_metric = self.params.get("rank_metric", "ic_mean")
        top_n = int(self.params.get("top_n", self.system.config["factors_per_cycle"]))
        if rank_metric in candidates_df.columns:
            return (
                candidates_df.assign(_rank_value=candidates_df[rank_metric].abs())
                .sort_values("_rank_value", ascending=False)
                .drop(columns="_rank_value")
                .head(top_n)
                .copy()
            )
        return candidates_df.head(top_n).copy()

    def _run_single_chain(self, chain_idx: int, cycle_timestamp: str, rows_per_chain: int) -> list[dict]:
        rng = random.Random(f"{cycle_timestamp}-{chain_idx}")
        examples_a, examples_b = self._sample_example_sets(rng)
        perspective_a = self._build_perspective("A", examples_a)
        perspective_b = self._build_perspective("B", examples_b)

        history = []
        accepted = []

        seed = self._generate_seed(
            chain_idx=chain_idx,
            cycle_timestamp=cycle_timestamp,
            rng=rng,
            examples=examples_a,
            perspective=perspective_a,
        )
        if seed is not None:
            history.append(seed)
            if seed.alpha_id.startswith(f"alpha-{cycle_timestamp}_MAD_seed_"):
                accepted.append(seed)

        debate_rounds = int(self.params.get("debate_rounds", 4))
        for round_idx in range(1, debate_rounds + 1):
            agent_name = "A" if round_idx % 2 == 1 else "B"
            examples = examples_a if agent_name == "A" else examples_b
            perspective = perspective_a if agent_name == "A" else perspective_b
            prev = history[-1] if history else None
            candidate = self._generate_debate_candidate(
                chain_idx=chain_idx,
                round_idx=round_idx,
                cycle_timestamp=cycle_timestamp,
                rng=rng,
                agent_name=agent_name,
                examples=examples,
                perspective=perspective,
                previous_candidate=prev,
                history=history,
            )
            if candidate is None:
                continue
            history.append(candidate)
            accepted.append(candidate)

        if not accepted:
            return []

        accepted.sort(
            key=lambda x: abs(x.ic_mean) if pd.notna(x.ic_mean) else -np.inf,
            reverse=True,
        )
        return [x.to_record() for x in accepted[:rows_per_chain]]

    def _sample_example_sets(self, rng: random.Random) -> tuple[list[dict], list[dict]]:
        records = list(self.system.baseline_factor_records)
        k = int(self.params.get("init_example_count", 2))
        if not records:
            return [], []

        k = min(k, len(records))
        examples_a = rng.sample(records, k)
        remaining = [r for r in records if r["alpha_id"] not in {x["alpha_id"] for x in examples_a}]
        if len(remaining) >= k:
            examples_b = rng.sample(remaining, k)
        else:
            examples_b = rng.sample(records, k)
        return examples_a, examples_b

    def _build_perspective(self, agent_name: str, examples: list[dict]) -> str:
        perspective = self.DEFAULT_PERSPECTIVES[agent_name]
        if not examples:
            return perspective

        motifs = []
        example_expr = " ".join(str(x.get("expression", "")) for x in examples)
        if "Rank(" in example_expr:
            motifs.append("偏好 Rank 收口")
        if "Quantile(" in example_expr:
            motifs.append("偏好 Quantile 收口")
        if "Corr(" in example_expr:
            motifs.append("允许使用相关性结构，但不要回到老模板")
        if "Delta(" in example_expr:
            motifs.append("关注变化强度而非静态水平")
        if "Mean(" in example_expr:
            motifs.append("倾向使用滚动均值做归一化")

        if motifs:
            perspective += "\n示例中常见有效做法：" + "；".join(sorted(set(motifs))) + "。"
        return perspective

    def _generate_seed(
        self,
        chain_idx: int,
        cycle_timestamp: str,
        rng: random.Random,
        examples: list[dict],
        perspective: str,
    ) -> DebateCandidate | None:
        p_seed = float(self.params.get("seed_from_baseline_prob", 0.5))
        if self.system.baseline_factor_records and rng.random() < p_seed:
            record = rng.choice(self.system.baseline_factor_records)
            return DebateCandidate(
                alpha_id=str(record["alpha_id"]),
                expression=str(record["expression"]),
                explanation=str(record.get("explanation", "")).split("=====")[0].strip(),
                summary="历史表现较好的已有因子，作为本轮辩论的种子。",
                critique="后续轮次需要在保持有效性的同时进一步提高多样性。",
                ic_mean=float(record.get("ic_mean", np.nan)),
                ic_ir=float(record.get("ic_ir", np.nan)),
                long_excret=float(record.get("long_excret", np.nan)),
                long_sharpe=float(record.get("long_sharpe", np.nan)),
                long_ir=float(record.get("long_ir", np.nan)),
                long_excmdd=float(record.get("long_excmdd", np.nan)),
                ls_ret=float(record.get("ls_ret", np.nan)),
                ls_sharpe=float(record.get("ls_sharpe", np.nan)),
                ls_mdd=float(record.get("ls_mdd", np.nan)),
            )

        prompt = self._build_seed_prompt(chain_idx, perspective, examples)
        alpha_id = f"alpha-{cycle_timestamp}_MAD_seed_{chain_idx:03d}"
        return self._query_and_validate(
            prompt=prompt,
            alpha_id=alpha_id,
            correction_hint="请修复并返回一个可计算、无未来函数、可跨股票比较的表达式。",
        )

    def _generate_debate_candidate(
        self,
        chain_idx: int,
        round_idx: int,
        cycle_timestamp: str,
        rng: random.Random,
        agent_name: str,
        examples: list[dict],
        perspective: str,
        previous_candidate: DebateCandidate | None,
        history: list[DebateCandidate],
    ) -> DebateCandidate | None:
        prompt = self._build_debate_prompt(
            agent_name=agent_name,
            perspective=perspective,
            examples=examples,
            previous_candidate=previous_candidate,
            history=history,
        )
        alpha_id = f"alpha-{cycle_timestamp}_MAD_c{chain_idx:03d}_r{round_idx:02d}_{agent_name}"
        return self._query_and_validate(
            prompt=prompt,
            alpha_id=alpha_id,
            correction_hint="请根据批评意见和报错修复当前候选，使其与已有因子更不相似但仍保持经济含义。",
        )

    def _query_and_validate(
        self,
        prompt: str,
        alpha_id: str,
        correction_hint: str,
    ) -> DebateCandidate | None:
        max_corrections = int(self.params.get("max_correction_rounds", 2))
        current_prompt = prompt
        last_response = ""

        for _ in range(max_corrections + 1):
            answer, think = query(current_prompt)
            last_response = answer
            parsed = self._parse_response(answer, think)
            candidate, errors = self._validate_candidate(alpha_id, parsed)
            if not errors:
                return candidate
            current_prompt = self._build_correction_prompt(
                original_prompt=prompt,
                last_response=last_response,
                errors=errors,
                correction_hint=correction_hint,
            )
        return None

    def _parse_response(self, answer: str, think: str) -> dict:
        def _extract_section(label: str) -> str:
            pattern = rf"{label}\s*:\s*(.*?)(?=\n(?:summary|critique|expression|explanation)\s*:|\Z)"
            match = re.search(pattern, answer, flags=re.IGNORECASE | re.DOTALL)
            return match.group(1).strip() if match else ""

        expression = extract_string(answer, "alpha")
        explanation = _extract_section("explanation") or think.strip()
        return {
            "summary": _extract_section("summary"),
            "critique": _extract_section("critique"),
            "expression": expression if isinstance(expression, str) else "",
            "explanation": explanation,
            "raw_answer": answer,
        }

    def _validate_candidate(self, alpha_id: str, parsed: dict) -> tuple[DebateCandidate | None, list[str]]:
        errors = []
        expression = (parsed.get("expression") or "").strip()
        if not expression:
            errors.append("OUTPUT FORMAT: 未找到用 <> 包裹的 expression。")
            return None, errors

        expression = convert_fields_to_lowercase(expression, self.system.fields)
        expression = expression.replace("'", "")

        if contains_future_function(expression):
            errors.append("LOOK-AHEAD BIAS: 检测到负窗口或未来函数。")
        if contains_too_many_fields(expression, list(self.system.ops_info.keys()), max_fields=12):
            errors.append("COMPLEX: 表达式过长，算子过多。")

        factor_values = None
        if not errors:
            try:
                factor_values = self.system.calculator.calculate(expression)
                factor_values = factor_values.replace([np.inf, -np.inf], np.nan)
            except Exception as e:
                errors.append(f"EXECUTION: 表达式计算失败 - {e}")

        if factor_values is not None:
            non_null_ratio = factor_values.notna().sum().sum() / max(1, factor_values.shape[0] * factor_values.shape[1])
            if non_null_ratio < float(self.params.get("min_non_null_ratio", 0.05)):
                errors.append("OUTPUT NAN: 有效值过少，因子过于稀疏。")

        perf = None
        if factor_values is not None and not errors:
            try:
                perf = calculate_factors_performance(
                    {alpha_id: factor_values},
                    [expression],
                    self.system.factor_dfs["close"],
                    start_date=self.system.config["start_date"],
                    end_date=self.system.config["end_date"],
                    explanation_list=[parsed.get("explanation", "")],
                    poolsel_path=self.system.validation_profile.get("poolsel_path"),
                )[0]
                if perf.get("ic_mean") is None or np.isnan(perf.get("ic_mean", np.nan)):
                    errors.append("EVALUATION: 因子训练期 IC 无效。")
            except Exception as e:
                errors.append(f"EVALUATION: 训练期评估失败 - {e}")

        if factor_values is not None and not errors and self.params.get("enforce_baseline_diversity", True):
            low_corr = is_low_correlated_with_fixed_factors(
                factor_values,
                self.system.baseline_factors,
                self.system.config["correlation_threshold"],
            )
            if not low_corr:
                errors.append("DIVERSITY: 与已有 baseline 因子相关性过高，请更换机制或收口方式。")

        if errors or perf is None:
            return None, errors

        candidate = DebateCandidate(
            alpha_id=alpha_id,
            expression=expression,
            explanation=parsed.get("explanation", ""),
            summary=parsed.get("summary", ""),
            critique=parsed.get("critique", ""),
            ic_mean=float(perf.get("ic_mean", np.nan)),
            ic_ir=float(perf.get("ic_ir", np.nan)),
            long_excret=float(perf.get("long_excret", np.nan)),
            long_sharpe=float(perf.get("long_sharpe", np.nan)),
            long_ir=float(perf.get("long_ir", np.nan)),
            long_excmdd=float(perf.get("long_excmdd", np.nan)),
            ls_ret=float(perf.get("ls_ret", np.nan)),
            ls_sharpe=float(perf.get("ls_sharpe", np.nan)),
            ls_mdd=float(perf.get("ls_mdd", np.nan)),
        )
        return candidate, []

    def _build_seed_prompt(self, chain_idx: int, perspective: str, examples: list[dict]) -> str:
        return f"""
你正在运行一个 FactorMAD 因子挖掘流程。
你是负责提出种子因子的 Agent A。

你的视角：
{perspective}

历史参考因子：
{self._format_examples(examples)}

当前任务：
- 直接提出一个新的 A 股量价因子表达式，作为本轮辩论的 seed。
- 表达式必须基于当前 DSL，可被直接计算。
- 最终信号必须可跨股票比较，尽量保持无量纲。

可用字段：
{', '.join(self.system.fields)}

可用算子：
{', '.join(sorted(self.system.ops_info.keys()))}

可用窗口：
{self.system.config['window']}

硬性要求：
- 禁止未来函数，任何负窗口都不允许。
- 优先保持表达式简洁，避免不必要的深层嵌套。
- 不要直接复刻历史参考因子，必须在机制或收口方式上有变化。
- 尽量让因子和已有库保持差异化。

请严格按如下格式输出：
summary: 你打算捕捉什么市场机制
critique: 当前 seed 因子可能的脆弱点
expression: <一个可计算的因子表达式>
explanation: 用通俗中文简要解释这个因子的原理
""".strip()

    def _build_debate_prompt(
        self,
        agent_name: str,
        perspective: str,
        examples: list[dict],
        previous_candidate: DebateCandidate | None,
        history: list[DebateCandidate],
    ) -> str:
        prev_block = self._format_candidate(previous_candidate) if previous_candidate else "无"
        history_block = self._format_history(history)
        return f"""
你正在运行一个 FactorMAD 因子挖掘流程。
你是 Agent {agent_name}，需要基于对方上一轮结果进行总结、批评和改进。

你的视角：
{perspective}

历史参考因子：
{self._format_examples(examples)}

最近几轮辩论记录：
{history_block}

上一轮候选：
{prev_block}

当前任务：
1. 先总结对方因子的核心机制。
2. 再指出它在稳健性、可解释性、多样性或实现层面的短板。
3. 最后给出一个改进后的新表达式。

可用字段：
{', '.join(self.system.fields)}

可用算子：
{', '.join(sorted(self.system.ops_info.keys()))}

可用窗口：
{self.system.config['window']}

硬性要求：
- 表达式必须能直接被当前 DSL 计算。
- 禁止未来函数，禁止负窗口。
- 最终信号应可跨股票比较，尽量无量纲。
- 避免和上一轮因子只改一个窗口数字；至少在机制、归一化或收口方式上做实质变化。
- 目标是在保持经济含义的同时，提高预测力并减少和已有因子的同质化。

请严格按如下格式输出：
summary: 对上一轮候选的机制总结
critique: 对上一轮候选的主要批评
expression: <改进后的因子表达式>
explanation: 用通俗中文解释你做了什么改进以及为什么更好
""".strip()

    def _build_correction_prompt(
        self,
        original_prompt: str,
        last_response: str,
        errors: list[str],
        correction_hint: str,
    ) -> str:
        error_text = "\n".join(f"- {e}" for e in errors)
        return f"""
你是 FactorMAD 的 correction agent。
上一版回答没有通过校验，请根据错误反馈修复，但尽量保留原有经济直觉。

原始任务：
{original_prompt}

上一版回答：
{last_response}

错误反馈：
{error_text}

修复要求：
- {correction_hint}
- 保持表达式可计算、无未来函数、复杂度适中。
- 如原思路本身有问题，可以换一种更稳健的归一化或收口方式。

请继续严格按如下格式输出：
summary: 修复后方案的机制总结
critique: 原方案的问题和修复思路
expression: <修复后的因子表达式>
explanation: 用通俗中文解释修复后的方案
""".strip()

    def _format_examples(self, examples: list[dict]) -> str:
        if not examples:
            return "当前没有可用历史示例，请根据你的视角自主提出新想法。"

        blocks = []
        for idx, rec in enumerate(examples, start=1):
            explanation = str(rec.get("explanation", "")).split("=====")[0].strip()
            blocks.append(
                f"[示例{idx}] expression={rec.get('expression')} | "
                f"ic_mean={rec.get('ic_mean', np.nan):.4f} | "
                f"ic_ir={rec.get('ic_ir', np.nan):.4f} | "
                f"long_excret={rec.get('long_excret', np.nan):.4f}\n"
                f"解释: {explanation[:220]}"
            )
        return "\n".join(blocks)

    def _format_candidate(self, candidate: DebateCandidate | None) -> str:
        if candidate is None:
            return "无"
        return (
            f"alpha_id={candidate.alpha_id}\n"
            f"expression={candidate.expression}\n"
            f"ic_mean={candidate.ic_mean:.4f}, ic_ir={candidate.ic_ir:.4f}, "
            f"long_excret={candidate.long_excret:.4f}\n"
            f"summary={candidate.summary[:180]}\n"
            f"critique={candidate.critique[:180]}\n"
            f"explanation={candidate.explanation[:220]}"
        )

    def _format_history(self, history: list[DebateCandidate]) -> str:
        if not history:
            return "暂无历史。"
        keep_n = int(self.params.get("history_window", 3))
        blocks = []
        for candidate in history[-keep_n:]:
            blocks.append(self._format_candidate(candidate))
        return "\n\n".join(blocks)
