from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .pipeline import AutoCleanMLResult


class ExperimentLogger:
    """
    Persists AutoCleanML experiment artifacts as JSON files.

    The logger intentionally stores serializable metadata, profiles, repair
    actions, and metrics rather than Spark DataFrames.
    """

    def __init__(self, output_dir: str | Path = "autocleanml/experiments"):
        self.output_dir = Path(output_dir)

    def log_run(
        self,
        run_name: str,
        result: AutoCleanMLResult,
        policy: Any | None = None,
        metadata: dict[str, Any] | None = None,
        ml_metrics: Any | None = None,
        thesis_report: Any | None = None,
    ) -> Path:
        run_dir = self._new_run_dir(run_name)
        run_dir.mkdir(parents=True, exist_ok=False)

        artifacts = {
            "metadata": metadata or {},
            "policy": self._to_jsonable(policy),
            "raw_profile": result.raw_profile,
            "repair_actions": result.repair_actions,
            "cleaned_profile": result.cleaned_profile,
            "evaluation": result.evaluation,
            "opex_metrics": result.opex_metrics,
        }

        if ml_metrics is not None:
            artifacts["ml_metrics"] = self._to_jsonable(ml_metrics)
        if thesis_report is not None:
            artifacts["thesis_report"] = self._to_jsonable(thesis_report)

        manifest = {
            "run_name": run_name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_files": {},
        }

        for artifact_name, artifact in artifacts.items():
            filename = f"{artifact_name}.json"
            self._write_json(run_dir / filename, artifact)
            manifest["artifact_files"][artifact_name] = filename

        self._write_json(run_dir / "manifest.json", manifest)
        return run_dir

    def _new_run_dir(self, run_name: str) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        safe_name = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_"
            for char in run_name.strip()
        ).strip("_")
        safe_name = safe_name or "autocleanml_run"
        return self.output_dir / f"{timestamp}_{safe_name}"

    def _write_json(self, path: Path, payload: Any) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(self._to_jsonable(payload), f, indent=2, sort_keys=True)
            f.write("\n")

    def _to_jsonable(self, value: Any) -> Any:
        if value is None:
            return None
        if is_dataclass(value):
            return self._to_jsonable(asdict(value))
        if isinstance(value, dict):
            return {
                str(key): self._to_jsonable(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [self._to_jsonable(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (str, int, float, bool)):
            return value

        simple_string = getattr(value, "simpleString", None)
        if callable(simple_string):
            return simple_string()

        return str(value)
