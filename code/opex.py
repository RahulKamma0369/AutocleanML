from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class OPEXMetrics:
    """
    Operational metrics for one AutoCleanML cleaning cycle.

    These metrics are intended to support the thesis OPEX argument: how much
    work the automated cleaning step performs, and how much runtime it costs.
    """

    raw_profile_time_seconds: float
    repair_time_seconds: float
    cleaned_profile_time_seconds: float
    evaluation_time_seconds: float
    total_time_seconds: float
    input_row_count: int
    output_row_count: int
    input_column_count: int
    output_column_count: int
    row_count_delta: int
    column_count_delta: int
    repair_action_count: int
    repair_actions_by_issue: dict[str, int]
    seconds_per_1000_input_rows: float | None
    automated_repair_actions: int
    manual_steps_avoided_estimate: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_opex_metrics(
    *,
    raw_profile: dict[str, Any],
    cleaned_profile: dict[str, Any],
    evaluation: dict[str, Any],
    raw_profile_time_seconds: float,
    repair_time_seconds: float,
    cleaned_profile_time_seconds: float,
    evaluation_time_seconds: float,
    total_time_seconds: float,
) -> dict[str, Any]:
    input_rows = int(raw_profile.get("row_count", 0))
    output_rows = int(cleaned_profile.get("row_count", 0))
    input_columns = int(raw_profile.get("column_count", 0))
    output_columns = int(cleaned_profile.get("column_count", 0))
    repair_action_count = int(evaluation.get("repair_action_count", 0))

    metrics = OPEXMetrics(
        raw_profile_time_seconds=round(raw_profile_time_seconds, 6),
        repair_time_seconds=round(repair_time_seconds, 6),
        cleaned_profile_time_seconds=round(cleaned_profile_time_seconds, 6),
        evaluation_time_seconds=round(evaluation_time_seconds, 6),
        total_time_seconds=round(total_time_seconds, 6),
        input_row_count=input_rows,
        output_row_count=output_rows,
        input_column_count=input_columns,
        output_column_count=output_columns,
        row_count_delta=output_rows - input_rows,
        column_count_delta=output_columns - input_columns,
        repair_action_count=repair_action_count,
        repair_actions_by_issue=dict(evaluation.get("repair_actions_by_issue", {})),
        seconds_per_1000_input_rows=(
            round(total_time_seconds / input_rows * 1000, 6)
            if input_rows > 0
            else None
        ),
        automated_repair_actions=repair_action_count,
        manual_steps_avoided_estimate=repair_action_count,
    )
    return metrics.to_dict()
