from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType

_NUMERIC_TYPE_PREFIXES = (
    "double", "float", "int", "bigint", "smallint", "tinyint", "long", "decimal",
)


def _is_numeric_type_string(type_str: str) -> bool:
    return any(type_str.lower().startswith(p) for p in _NUMERIC_TYPE_PREFIXES)


class DataProfiler:
    """
    Profiling module for AutoCleanML.

    This module does not clean data.
    It only measures data-quality issues and returns a profiling report.

    Current profiling groups:
    1. Missingness
    2. Duplicates
    3. Outliers
    4. Key skew
    5. Schema drift
    6. Label noise (null stats + RF confidence-based noise estimation)
    """

    def __init__(
        self,
        outlier_iqr_multiplier: float = 1.5,
        label_noise_min_rows: int = 30,
        label_noise_confidence_threshold: float = 0.65,
    ):
        self.outlier_iqr_multiplier = outlier_iqr_multiplier
        self.label_noise_min_rows = label_noise_min_rows
        self.label_noise_confidence_threshold = label_noise_confidence_threshold

    def profile(
        self,
        df: DataFrame,
        key_columns: list[str] | None = None,
        reference_schema: dict | None = None,
        label_col: str | None = None,
    ) -> dict:
        row_count = df.count()
        current_schema = self._profile_schema(df)

        return {
            "row_count": row_count,
            "column_count": len(df.columns),
            "schema": current_schema,
            "missingness": self._profile_missingness(df, row_count),
            "duplicates": self._profile_duplicates(df, row_count, key_columns),
            "outliers": self._profile_outliers(df, row_count),
            "skew": self._profile_skew(df, key_columns),
            "schema_drift": self._profile_schema_drift(
                current_schema=current_schema,
                reference_schema=reference_schema,
            ),
            "label_noise": self._profile_label_noise(
                df=df,
                label_col=label_col,
                row_count=row_count,
                reference_schema=reference_schema,
            ),
        }

    def _profile_schema(self, df: DataFrame) -> dict:
        return {
            field.name: {
                "data_type": field.dataType.simpleString(),
                "nullable": field.nullable,
            }
            for field in df.schema.fields
        }

    def _profile_missingness(self, df: DataFrame, row_count: int) -> dict:
        if row_count == 0:
            return {}

        null_counts = df.select([
            F.sum(F.col(c).isNull().cast("int")).alias(c)
            for c in df.columns
        ]).collect()[0].asDict()

        report = {}

        for col, count in null_counts.items():
            missing_ratio = count / row_count

            report[col] = {
                "missing_count": int(count),
                "missing_ratio": round(missing_ratio, 4),
                "severity": self._missingness_severity(missing_ratio),
            }

        return report

    def _profile_duplicates(
        self,
        df: DataFrame,
        row_count: int,
        key_columns: list[str] | None,
    ) -> dict:
        if row_count == 0:
            return {
                "duplicate_count": 0,
                "duplicate_ratio": 0.0,
                "severity": "none",
                "exact": {
                    "duplicate_count": 0,
                    "duplicate_ratio": 0.0,
                    "severity": "none",
                },
                "by_key": {},
                "composite_key": {},
            }

        distinct_count = df.distinct().count()
        duplicate_count = row_count - distinct_count
        duplicate_ratio = duplicate_count / row_count
        exact_report = {
            "duplicate_count": duplicate_count,
            "duplicate_ratio": round(duplicate_ratio, 4),
            "severity": self._duplicate_severity(duplicate_ratio),
        }

        return {
            "duplicate_count": duplicate_count,
            "duplicate_ratio": round(duplicate_ratio, 4),
            "severity": self._duplicate_severity(duplicate_ratio),
            "exact": exact_report,
            "by_key": self._profile_key_duplicates(df, row_count, key_columns),
            "composite_key": self._profile_composite_key_duplicates(
                df,
                row_count,
                key_columns,
            ),
        }

    def _profile_outliers(self, df: DataFrame, row_count: int) -> dict:
        if row_count == 0:
            return {}

        numeric_cols = [
            field.name
            for field in df.schema.fields
            if isinstance(field.dataType, NumericType)
        ]

        report = {}

        for col in numeric_cols:
            quantiles = df.approxQuantile(col, [0.25, 0.75], 0.01)

            if len(quantiles) < 2:
                continue

            q1, q3 = quantiles
            iqr = q3 - q1

            if iqr == 0:
                outlier_count = df.filter(
                    F.col(col).isNotNull() & (F.col(col) != q1)
                ).count()
                outlier_ratio = outlier_count / row_count

                report[col] = {
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower_bound": None,
                    "upper_bound": None,
                    "outlier_count": outlier_count,
                    "outlier_ratio": round(outlier_ratio, 4),
                    "severity": self._outlier_severity(outlier_ratio),
                    "method": "zero_iqr_deviation",
                    "repairable": False,
                    "message": (
                        "Q1 and Q3 are equal. Deviations from the dominant "
                        "value are reported, but automatic capping is skipped "
                        "to avoid destructive repairs."
                    ),
                }
                continue

            lower_bound = q1 - self.outlier_iqr_multiplier * iqr
            upper_bound = q3 + self.outlier_iqr_multiplier * iqr

            outlier_count = df.filter(
                (F.col(col) < lower_bound) | (F.col(col) > upper_bound)
            ).count()

            outlier_ratio = outlier_count / row_count

            report[col] = {
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": outlier_count,
                "outlier_ratio": round(outlier_ratio, 4),
                "severity": self._outlier_severity(outlier_ratio),
                "method": "iqr",
                "repairable": True,
            }

        return report

    def _profile_key_duplicates(
        self,
        df: DataFrame,
        row_count: int,
        key_columns: list[str] | None,
    ) -> dict:
        if not key_columns:
            return {}

        valid_keys = [key for key in key_columns if key in df.columns]
        missing_keys = [key for key in key_columns if key not in df.columns]

        report = {}
        for key in missing_keys:
            report[key] = {"error": f"Column '{key}' not found in DataFrame."}

        for key in valid_keys:
            duplicate_groups = df.groupBy(key).count().filter(F.col("count") > 1)
            duplicate_group_count = duplicate_groups.count()
            duplicate_row_count = duplicate_groups.agg(F.sum("count")).collect()[0][0]
            duplicate_row_count = duplicate_row_count or 0
            duplicate_excess_count = duplicate_row_count - duplicate_group_count
            duplicate_ratio = duplicate_row_count / row_count

            report[key] = {
                "duplicate_group_count": duplicate_group_count,
                "duplicate_row_count": duplicate_row_count,
                "duplicate_excess_count": duplicate_excess_count,
                "duplicate_ratio": round(duplicate_ratio, 4),
                "severity": self._duplicate_severity(duplicate_ratio),
                "message": (
                    "Rows sharing this key are not automatically duplicates; "
                    "this metric is diagnostic unless key-based deduplication "
                    "is explicitly enabled."
                ),
            }

        return report

    def _profile_composite_key_duplicates(
        self,
        df: DataFrame,
        row_count: int,
        key_columns: list[str] | None,
    ) -> dict:
        if not key_columns:
            return {}

        valid_keys = [key for key in key_columns if key in df.columns]
        missing_keys = [key for key in key_columns if key not in df.columns]

        if len(valid_keys) < 2:
            return {
                "key_columns": key_columns,
                "valid_key_columns": valid_keys,
                "missing_key_columns": missing_keys,
                "message": (
                    "Composite-key duplicate profiling requires at least two "
                    "valid key columns."
                ),
            }

        duplicate_groups = df.groupBy(*valid_keys).count().filter(F.col("count") > 1)
        duplicate_group_count = duplicate_groups.count()
        duplicate_row_count = duplicate_groups.agg(F.sum("count")).collect()[0][0]
        duplicate_row_count = duplicate_row_count or 0
        duplicate_excess_count = duplicate_row_count - duplicate_group_count
        duplicate_ratio = duplicate_row_count / row_count

        return {
            "key_columns": key_columns,
            "valid_key_columns": valid_keys,
            "missing_key_columns": missing_keys,
            "duplicate_group_count": duplicate_group_count,
            "duplicate_row_count": duplicate_row_count,
            "duplicate_excess_count": duplicate_excess_count,
            "duplicate_ratio": round(duplicate_ratio, 4),
            "severity": self._duplicate_severity(duplicate_ratio),
            "message": (
                "Rows sharing the full key combination are possible duplicate "
                "entities. This is diagnostic unless key-based deduplication is "
                "explicitly enabled."
            ),
        }

    def _profile_skew(self, df: DataFrame, key_columns: list[str] | None) -> dict:
        if not key_columns:
            return {}

        report = {}

        for key in key_columns:
            if key not in df.columns:
                report[key] = {
                    "error": f"Column '{key}' not found in DataFrame."
                }
                continue

            key_counts = df.groupBy(key).count()

            max_count = key_counts.agg(F.max("count")).collect()[0][0]
            median_values = key_counts.approxQuantile("count", [0.5], 0.01)

            if not median_values or median_values[0] == 0:
                skew_ratio = None
            else:
                skew_ratio = max_count / median_values[0]

            report[key] = {
                "max_count": max_count,
                "median_count": median_values[0] if median_values else None,
                "skew_ratio": round(skew_ratio, 4) if skew_ratio is not None else None,
                "severity": self._skew_severity(skew_ratio),
            }

        return report

    def _profile_schema_drift(
        self,
        current_schema: dict,
        reference_schema: dict | None,
    ) -> dict:
        if reference_schema is None:
            return {
                "evaluated": False,
                "message": "No reference schema provided.",
                "added_columns": [],
                "removed_columns": [],
                "type_changes": [],
                "drift_detected": False,
            }

        current_cols = set(current_schema.keys())
        reference_cols = set(reference_schema.keys())

        added_columns = sorted(list(current_cols - reference_cols))
        removed_columns = sorted(list(reference_cols - current_cols))

        type_changes = []

        common_cols = current_cols.intersection(reference_cols)

        for col in common_cols:
            current_type = current_schema[col]["data_type"]
            reference_type = reference_schema[col]["data_type"]

            if current_type != reference_type:
                type_changes.append({
                    "column": col,
                    "reference_type": reference_type,
                    "current_type": current_type,
                })

        drift_detected = bool(added_columns or removed_columns or type_changes)

        return {
            "evaluated": True,
            "added_columns": added_columns,
            "removed_columns": removed_columns,
            "type_changes": type_changes,
            "drift_detected": drift_detected,
        }

    def _profile_label_noise(
        self,
        df: DataFrame,
        label_col: str | None,
        row_count: int,
        reference_schema: dict | None = None,
    ) -> dict:
        if label_col is None:
            return {
                "evaluated": False,
                "message": "No label column provided.",
            }

        if label_col not in df.columns:
            return {
                "evaluated": False,
                "message": f"Label column '{label_col}' not found.",
            }

        if row_count == 0:
            return {
                "evaluated": True,
                "label_col": label_col,
                "label_null_count": 0,
                "label_null_ratio": 0.0,
                "distinct_label_count": 0,
                "confidence_noise": {"evaluated": False, "message": "Empty DataFrame."},
            }

        label_null_count = df.filter(F.col(label_col).isNull()).count()
        distinct_label_count = df.select(label_col).distinct().count()
        label_null_ratio = label_null_count / row_count
        labeled_count = row_count - label_null_count

        label_is_categorical = any(
            f.name == label_col and isinstance(f.dataType, StringType)
            for f in df.schema.fields
        )

        result: dict = {
            "evaluated": True,
            "label_col": label_col,
            "label_null_count": label_null_count,
            "label_null_ratio": round(label_null_ratio, 4),
            "distinct_label_count": distinct_label_count,
        }

        if (
            label_is_categorical
            and distinct_label_count >= 2
            and labeled_count >= self.label_noise_min_rows
        ):
            result["confidence_noise"] = self._estimate_label_confidence_noise(
                df, label_col, reference_schema
            )
        else:
            result["confidence_noise"] = {
                "evaluated": False,
                "message": (
                    "Skipped: confidence scoring requires a categorical label, "
                    f"at least 2 classes, and {self.label_noise_min_rows} labeled rows."
                ),
            }

        return result

    def _estimate_label_confidence_noise(
        self,
        df: DataFrame,
        label_col: str,
        reference_schema: dict | None = None,
    ) -> dict:
        numeric_cols = sorted([
            f.name for f in df.schema.fields
            if isinstance(f.dataType, NumericType) and f.name != label_col
        ])
        string_cols = sorted([
            f.name for f in df.schema.fields
            if isinstance(f.dataType, StringType) and f.name != label_col
        ])

        # Columns that are now string but were originally numeric (schema drift).
        # Restore them so the RF can use them as continuous features.
        type_restored: list[tuple[str, str]] = []
        if reference_schema:
            restored_names = set()
            for col_name in string_cols:
                ref_type = reference_schema.get(col_name, {}).get("data_type", "")
                if _is_numeric_type_string(ref_type):
                    type_restored.append((col_name, ref_type))
                    restored_names.add(col_name)
            string_cols = [c for c in string_cols if c not in restored_names]
            numeric_cols = sorted(numeric_cols + [c for c, _ in type_restored])

        usable_features = numeric_cols + string_cols

        if not usable_features:
            return {
                "evaluated": False,
                "message": "No numeric or string feature columns available.",
            }

        labeled_df = df.filter(F.col(label_col).isNotNull())

        # Cast restored columns to their original types before dropping nulls —
        # non-numeric strings in these columns will become null and be dropped.
        for col_name, ref_type in type_restored:
            labeled_df = labeled_df.withColumn(col_name, F.col(col_name).cast(ref_type))

        labeled_df = labeled_df.dropna(subset=usable_features)
        labeled_count = labeled_df.count()

        if labeled_count < self.label_noise_min_rows:
            return {
                "evaluated": False,
                "message": (
                    f"Too few complete labeled rows after dropping nulls "
                    f"({labeled_count} < {self.label_noise_min_rows})."
                ),
            }

        try:
            pipeline = self._build_label_noise_pipeline(label_col, numeric_cols, string_cols)
            model = pipeline.fit(labeled_df)

            predictions = (
                model.transform(labeled_df)
                .withColumn(
                    "_profiler_max_prob",
                    F.array_max(vector_to_array(F.col("_profiler_probability"))),
                )
                .cache()
            )

            suspected_count = predictions.filter(
                (F.col("_profiler_indexed_label") != F.col("_profiler_prediction"))
                & (F.col("_profiler_max_prob") >= self.label_noise_confidence_threshold)
            ).count()

            stats = predictions.agg(
                F.mean("_profiler_max_prob").alias("avg_confidence"),
                F.sum(
                    (F.col("_profiler_max_prob") < self.label_noise_confidence_threshold).cast("int")
                ).alias("low_confidence_count"),
            ).collect()[0]

            predictions.unpersist()

            suspected_ratio = suspected_count / labeled_count
            avg_conf = stats["avg_confidence"]
            low_conf_ratio = (stats["low_confidence_count"] or 0) / labeled_count

            return {
                "evaluated": True,
                "method": "random_forest_self_evaluation",
                "confidence_threshold": self.label_noise_confidence_threshold,
                "labeled_rows_used": labeled_count,
                "suspected_noise_count": suspected_count,
                "suspected_noise_ratio": round(suspected_ratio, 4),
                "average_prediction_confidence": (
                    round(avg_conf, 4) if avg_conf is not None else None
                ),
                "low_confidence_ratio": round(low_conf_ratio, 4),
                "severity": self._label_noise_severity(suspected_ratio),
                "type_restored_columns": [c for c, _ in type_restored],
                "note": (
                    "Self-evaluation on training data — RF may overfit clean data. "
                    "Treat suspected_noise_ratio as a diagnostic signal, not ground truth."
                ),
            }
        except Exception as exc:
            return {
                "evaluated": False,
                "message": f"Confidence scoring failed: {exc}",
            }

    def _build_label_noise_pipeline(
        self,
        label_col: str,
        numeric_cols: list[str],
        string_cols: list[str],
    ) -> Pipeline:
        label_indexer = StringIndexer(
            inputCol=label_col,
            outputCol="_profiler_indexed_label",
            handleInvalid="skip",
        )
        category_indexers = [
            StringIndexer(
                inputCol=col,
                outputCol=f"_profiler_{col}_idx",
                handleInvalid="keep",
            )
            for col in string_cols
        ]
        category_encoders = [
            OneHotEncoder(
                inputCol=f"_profiler_{col}_idx",
                outputCol=f"_profiler_{col}_vec",
                handleInvalid="keep",
            )
            for col in string_cols
        ]
        assembler = VectorAssembler(
            inputCols=numeric_cols + [f"_profiler_{col}_vec" for col in string_cols],
            outputCol="_profiler_features",
        )
        classifier = RandomForestClassifier(
            featuresCol="_profiler_features",
            labelCol="_profiler_indexed_label",
            probabilityCol="_profiler_probability",
            predictionCol="_profiler_prediction",
            numTrees=50,
            maxDepth=5,
            seed=42,
        )
        return Pipeline(stages=[
            label_indexer,
            *category_indexers,
            *category_encoders,
            assembler,
            classifier,
        ])

    def _missingness_severity(self, missing_ratio: float) -> str:
        if missing_ratio == 0:
            return "none"
        if missing_ratio < 0.10:
            return "low"
        if missing_ratio <= 0.40:
            return "medium"
        return "high"

    def _duplicate_severity(self, duplicate_ratio: float) -> str:
        if duplicate_ratio == 0:
            return "none"
        if duplicate_ratio <= 0.02:
            return "low"
        if duplicate_ratio <= 0.15:
            return "medium"
        return "high"

    def _outlier_severity(self, outlier_ratio: float) -> str:
        if outlier_ratio == 0:
            return "none"
        if outlier_ratio < 0.05:
            return "low"
        return "high"

    def _skew_severity(self, skew_ratio: float | None) -> str:
        if skew_ratio is None:
            return "unknown"
        if skew_ratio <= 5:
            return "low"
        if skew_ratio <= 10:
            return "medium"
        return "high"

    def _label_noise_severity(self, suspected_ratio: float) -> str:
        if suspected_ratio == 0:
            return "none"
        if suspected_ratio < 0.03:
            return "low"
        if suspected_ratio <= 0.10:
            return "medium"
        return "high"
