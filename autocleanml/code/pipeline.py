from dataclasses import dataclass
from time import perf_counter
from typing import Any

from pyspark.sql import DataFrame

from .evaluation import DataQualityEvaluator
from .opex import build_opex_metrics
from .profiler import DataProfiler
from .repair import RepairEngine, RepairPolicy


@dataclass
class AutoCleanMLResult:
    raw_profile: dict[str, Any]
    cleaned_profile: dict[str, Any]
    evaluation: dict[str, Any]
    opex_metrics: dict[str, Any]
    repair_actions: list[dict[str, Any]]
    cleaned_df: DataFrame
    repair_policy: RepairPolicy = None


class AutoCleanML:
    """
    End-to-end AutoCleanML workflow.

    Runs distributed profiling, applies rule-driven repairs, then profiles the
    cleaned output so experiments can compare before/after data quality.
    """

    def __init__(
        self,
        profiler: DataProfiler | None = None,
        repair_engine: RepairEngine | None = None,
        repair_policy: RepairPolicy | None = None,
        evaluator: DataQualityEvaluator | None = None,
    ):
        self.profiler = profiler or DataProfiler()
        self.repair_engine = repair_engine or RepairEngine(repair_policy)
        self.evaluator = evaluator or DataQualityEvaluator()

    def run(
        self,
        df: DataFrame,
        key_columns: list[str] | None = None,
        reference_schema: dict | None = None,
        label_col: str | None = None,
    ) -> AutoCleanMLResult:
        total_start = perf_counter()

        raw_profile_start = perf_counter()
        raw_profile = self.profiler.profile(
            df=df,
            key_columns=key_columns,
            reference_schema=reference_schema,
            label_col=label_col,
        )
        raw_profile_time = perf_counter() - raw_profile_start

        repair_start = perf_counter()
        repair_result = self.repair_engine.repair(
            df=df,
            profile_report=raw_profile,
            reference_schema=reference_schema,
            key_columns=key_columns,
            label_col=label_col,
        )
        repair_time = perf_counter() - repair_start

        cleaned_profile_start = perf_counter()
        cleaned_profile = self.profiler.profile(
            df=repair_result.cleaned_df,
            key_columns=key_columns,
            reference_schema=reference_schema,
            label_col=label_col,
        )
        cleaned_profile_time = perf_counter() - cleaned_profile_start

        evaluation_start = perf_counter()
        evaluation = self.evaluator.evaluate(
            raw_profile=raw_profile,
            cleaned_profile=cleaned_profile,
            repair_actions=repair_result.actions,
        )
        evaluation_time = perf_counter() - evaluation_start
        total_time = perf_counter() - total_start
        opex_metrics = build_opex_metrics(
            raw_profile=raw_profile,
            cleaned_profile=cleaned_profile,
            evaluation=evaluation.metrics,
            raw_profile_time_seconds=raw_profile_time,
            repair_time_seconds=repair_time,
            cleaned_profile_time_seconds=cleaned_profile_time,
            evaluation_time_seconds=evaluation_time,
            total_time_seconds=total_time,
        )

        return AutoCleanMLResult(
            raw_profile=raw_profile,
            cleaned_profile=cleaned_profile,
            evaluation=evaluation.metrics,
            opex_metrics=opex_metrics,
            repair_actions=repair_result.actions,
            cleaned_df=repair_result.cleaned_df,
            repair_policy=self.repair_engine.policy,
        )
