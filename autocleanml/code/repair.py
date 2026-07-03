from dataclasses import dataclass, field
from typing import Any

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType


@dataclass
class RepairPolicy:
    """
    Dataset-agnostic repair settings for AutoCleanML.

    Policies are intentionally simple and deterministic so they can be logged,
    audited, and reused across datasets.
    """

    missingness_threshold: float = 0.0
    numeric_imputation: str = "median"
    categorical_imputation: str = "constant"
    categorical_fill_value: str = "unknown"
    drop_duplicates: bool = True
    duplicate_strategy: str = "exact"
    outlier_strategy: str = "cap"
    outlier_column_strategies: dict[str, str] = field(default_factory=dict)
    repair_outlier_severities: tuple[str, ...] = ("low", "high")
    align_schema: bool = True
    drop_added_columns: bool = False
    label_imputation: str = "none"
    label_confidence_threshold: float = 0.0
    label_min_training_rows: int = 20
    skew_strategy: str = "none"
    repair_skew_severities: tuple[str, ...] = ("high",)
    skew_target_partitions: int | None = None
    skip_imputation: bool = False


@dataclass
class RepairResult:
    cleaned_df: DataFrame
    actions: list[dict[str, Any]] = field(default_factory=list)


class RepairEngine:
    """
    Rule-driven repair engine for AutoCleanML.

    The engine consumes a profiling report and applies deterministic Spark
    DataFrame transformations for common tabular data-quality issues.
    """

    def __init__(self, policy: RepairPolicy | None = None):
        self.policy = policy or RepairPolicy()

    def repair(
        self,
        df: DataFrame,
        profile_report: dict,
        reference_schema: dict | None = None,
        key_columns: list[str] | None = None,
        label_col: str | None = None,
    ) -> RepairResult:
        cleaned_df = df
        actions: list[dict[str, Any]] = []

        if self.policy.align_schema and reference_schema:
            cleaned_df, schema_actions = self._align_schema(
                cleaned_df,
                reference_schema,
                profile_report.get("schema_drift", {}),
            )
            actions.extend(schema_actions)

        if not self.policy.skip_imputation:
            cleaned_df, missing_actions = self._repair_missingness(
                cleaned_df,
                profile_report.get("missingness", {}),
                skip_columns={label_col} if label_col else set(),
            )
            actions.extend(missing_actions)

        cleaned_df, outlier_actions = self._repair_outliers(
            cleaned_df,
            profile_report.get("outliers", {}),
        )
        actions.extend(outlier_actions)

        cleaned_df, label_actions = self._repair_labels(
            cleaned_df,
            label_col,
            profile_report.get("label_noise", {}),
        )
        actions.extend(label_actions)

        cleaned_df, duplicate_actions = self._repair_duplicates(
            cleaned_df,
            profile_report.get("duplicates", {}),
            key_columns,
        )
        actions.extend(duplicate_actions)

        cleaned_df, skew_actions = self._repair_skew(
            cleaned_df,
            profile_report.get("skew", {}),
        )
        actions.extend(skew_actions)

        return RepairResult(cleaned_df=cleaned_df, actions=actions)

    def _repair_missingness(
        self,
        df: DataFrame,
        missingness_report: dict,
        skip_columns: set[str],
    ) -> tuple[DataFrame, list[dict[str, Any]]]:
        actions: list[dict[str, Any]] = []
        cleaned_df = df

        numeric_cols = {
            field.name
            for field in cleaned_df.schema.fields
            if isinstance(field.dataType, NumericType)
        }
        string_cols = {
            field.name
            for field in cleaned_df.schema.fields
            if isinstance(field.dataType, StringType)
        }

        for col, stats in missingness_report.items():
            if col not in cleaned_df.columns:
                continue
            if col in skip_columns:
                continue

            missing_ratio = stats.get("missing_ratio", 0.0)
            if missing_ratio <= self.policy.missingness_threshold:
                continue

            if col in numeric_cols:
                fill_value = self._numeric_fill_value(cleaned_df, col)
                if fill_value is None:
                    continue

                cleaned_df = cleaned_df.fillna({col: fill_value})
                actions.append({
                    "issue": "missingness",
                    "column": col,
                    "strategy": self.policy.numeric_imputation,
                    "fill_value": fill_value,
                    "missing_ratio": missing_ratio,
                })
            elif col in string_cols:
                fill_value = self._categorical_fill_value(cleaned_df, col)
                if fill_value is None:
                    continue

                cleaned_df = cleaned_df.fillna({col: fill_value})
                actions.append({
                    "issue": "missingness",
                    "column": col,
                    "strategy": self.policy.categorical_imputation,
                    "fill_value": fill_value,
                    "missing_ratio": missing_ratio,
                })

        return cleaned_df, actions

    def _repair_labels(
        self,
        df: DataFrame,
        label_col: str | None,
        label_report: dict,
    ) -> tuple[DataFrame, list[dict[str, Any]]]:
        if self.policy.label_imputation not in {"none", "model"}:
            raise ValueError(
                "Unsupported label_imputation. Expected one of: 'none', 'model'."
            )

        if self.policy.label_imputation == "none":
            return df, []
        if label_col is None or label_col not in df.columns:
            return df, []

        label_null_count = label_report.get("label_null_count", 0)
        if label_null_count <= 0:
            return df, []

        return self._repair_labels_with_model(df, label_col, label_null_count)

    def _repair_labels_with_model(
        self,
        df: DataFrame,
        label_col: str,
        label_null_count: int,
    ) -> tuple[DataFrame, list[dict[str, Any]]]:
        indexed_df = df.withColumn("_autocleanml_row_id", F.monotonically_increasing_id())
        feature_cols = [col for col in df.columns if col != label_col]

        numeric_cols = {
            field.name
            for field in df.schema.fields
            if field.name in feature_cols and isinstance(field.dataType, NumericType)
        }
        string_cols = {
            field.name
            for field in df.schema.fields
            if field.name in feature_cols and isinstance(field.dataType, StringType)
        }
        usable_features = sorted(numeric_cols | string_cols)

        if not usable_features:
            return df, [{
                "issue": "label_noise",
                "strategy": "skip_model_imputation",
                "reason": "No numeric or string feature columns available.",
            }]

        training_df = indexed_df.filter(F.col(label_col).isNotNull()).dropna(
            subset=usable_features
        )
        training_count = training_df.count()
        if training_count < self.policy.label_min_training_rows:
            return df, [{
                "issue": "label_noise",
                "strategy": "skip_model_imputation",
                "reason": "Not enough labeled training rows.",
                "training_row_count": training_count,
                "minimum_training_rows": self.policy.label_min_training_rows,
            }]

        prediction_candidates = indexed_df.filter(F.col(label_col).isNull()).dropna(
            subset=usable_features
        )
        candidate_count = prediction_candidates.count()
        if candidate_count == 0:
            return df, [{
                "issue": "label_noise",
                "strategy": "skip_model_imputation",
                "reason": "No missing-label rows had complete usable features.",
                "label_null_count": label_null_count,
            }]

        pipeline = self._build_label_imputation_pipeline(
            label_col=label_col,
            numeric_cols=sorted(numeric_cols),
            categorical_cols=sorted(string_cols),
        )
        model = pipeline.fit(training_df)
        label_model = model.stages[0]
        labels = list(label_model.labels)
        if not labels:
            return df, [{
                "issue": "label_noise",
                "strategy": "skip_model_imputation",
                "reason": "No label classes were learned.",
                "training_row_count": training_count,
            }]

        prediction_input = prediction_candidates.withColumn(label_col, F.lit(labels[0]))
        predictions = model.transform(prediction_input)
        label_lookup = F.create_map([
            item
            for idx, label in enumerate(labels)
            for item in (F.lit(float(idx)), F.lit(label))
        ])

        prediction_updates = (
            predictions
            .withColumn(
                "_autocleanml_confidence",
                F.array_max(vector_to_array(F.col("_autocleanml_probability"))),
            )
            .filter(
                F.col("_autocleanml_confidence")
                >= self.policy.label_confidence_threshold
            )
            .select(
                "_autocleanml_row_id",
                label_lookup[F.col("_autocleanml_prediction")].alias(
                    "_autocleanml_label"
                ),
                "_autocleanml_confidence",
            )
        )
        imputed_count = prediction_updates.count()

        repaired_df = (
            indexed_df
            .join(prediction_updates, on="_autocleanml_row_id", how="left")
            .withColumn(
                label_col,
                F.when(
                    F.col(label_col).isNull() & F.col("_autocleanml_label").isNotNull(),
                    F.col("_autocleanml_label"),
                ).otherwise(F.col(label_col)),
            )
            .drop(
                "_autocleanml_row_id",
                "_autocleanml_label",
                "_autocleanml_confidence",
            )
        )

        return repaired_df, [{
            "issue": "label_noise",
            "strategy": "model_label_imputation",
            "label_col": label_col,
            "label_null_count": label_null_count,
            "candidate_count": candidate_count,
            "imputed_count": imputed_count,
            "training_row_count": training_count,
            "confidence_threshold": self.policy.label_confidence_threshold,
        }]

    def _build_label_imputation_pipeline(
        self,
        label_col: str,
        numeric_cols: list[str],
        categorical_cols: list[str],
    ) -> Pipeline:
        label_indexer = StringIndexer(
            inputCol=label_col,
            outputCol="_autocleanml_indexed_label",
            handleInvalid="skip",
        )
        category_indexers = [
            StringIndexer(
                inputCol=col,
                outputCol=f"_autocleanml_{col}_idx",
                handleInvalid="keep",
            )
            for col in categorical_cols
        ]
        category_encoders = [
            OneHotEncoder(
                inputCol=f"_autocleanml_{col}_idx",
                outputCol=f"_autocleanml_{col}_vec",
                handleInvalid="keep",
            )
            for col in categorical_cols
        ]
        assembler = VectorAssembler(
            inputCols=numeric_cols + [
                f"_autocleanml_{col}_vec" for col in categorical_cols
            ],
            outputCol="_autocleanml_features",
        )
        classifier = RandomForestClassifier(
            featuresCol="_autocleanml_features",
            labelCol="_autocleanml_indexed_label",
            probabilityCol="_autocleanml_probability",
            predictionCol="_autocleanml_prediction",
            numTrees=50,
            seed=42,
        )

        return Pipeline(
            stages=[
                label_indexer,
                *category_indexers,
                *category_encoders,
                assembler,
                classifier,
            ]
        )

    def _repair_duplicates(
        self,
        df: DataFrame,
        duplicate_report: dict,
        key_columns: list[str] | None,
    ) -> tuple[DataFrame, list[dict[str, Any]]]:
        if not self.policy.drop_duplicates:
            return df, []

        if self.policy.duplicate_strategy not in {"exact", "key"}:
            raise ValueError(
                "Unsupported duplicate_strategy. Expected one of: 'exact', 'key'."
            )

        if self.policy.duplicate_strategy == "key":
            valid_keys = [key for key in key_columns or [] if key in df.columns]
            if not valid_keys:
                return df, []

            duplicate_groups = df.groupBy(*valid_keys).count().filter(F.col("count") > 1)
            duplicate_count = duplicate_groups.agg(F.sum("count")).collect()[0][0] or 0
            if duplicate_count <= 0:
                return df, []

            return df.dropDuplicates(valid_keys), [{
                "issue": "duplicates",
                "strategy": "dropDuplicates_by_key",
                "key_columns": valid_keys,
                "duplicate_row_count": duplicate_count,
            }]

        duplicate_count = duplicate_report.get("exact", {}).get(
            "duplicate_count",
            duplicate_report.get("duplicate_count", 0),
        )
        if duplicate_count <= 0:
            return df, []

        return df.dropDuplicates(), [{
            "issue": "duplicates",
            "strategy": "dropDuplicates",
            "duplicate_count": duplicate_count,
            "duplicate_ratio": duplicate_report.get("duplicate_ratio", 0.0),
        }]

    def _repair_skew(
        self,
        df: DataFrame,
        skew_report: dict,
    ) -> tuple[DataFrame, list[dict[str, Any]]]:
        if self.policy.skew_strategy not in {"none", "repartition"}:
            raise ValueError(
                "Unsupported skew_strategy. Expected one of: 'none', 'repartition'."
            )

        if self.policy.skew_strategy == "none":
            return df, []
        if (
            self.policy.skew_target_partitions is not None
            and self.policy.skew_target_partitions <= 0
        ):
            raise ValueError("skew_target_partitions must be a positive integer.")

        repairable_skew = []
        for col, stats in skew_report.items():
            if col not in df.columns or "error" in stats:
                continue

            severity = stats.get("severity")
            skew_ratio = stats.get("skew_ratio")
            if severity not in self.policy.repair_skew_severities:
                continue
            if skew_ratio is None:
                continue

            repairable_skew.append((skew_ratio, col, stats))

        if not repairable_skew:
            return df, []

        skew_ratio, col, stats = sorted(repairable_skew, reverse=True)[0]
        if self.policy.skew_target_partitions is None:
            repaired_df = df.repartition(F.col(col))
            target_partitions = repaired_df.rdd.getNumPartitions()
        else:
            repaired_df = df.repartition(self.policy.skew_target_partitions, F.col(col))
            target_partitions = self.policy.skew_target_partitions

        return repaired_df, [{
            "issue": "skew",
            "column": col,
            "strategy": "repartition",
            "skew_ratio": skew_ratio,
            "severity": stats.get("severity"),
            "max_count": stats.get("max_count"),
            "median_count": stats.get("median_count"),
            "target_partitions": target_partitions,
            "message": (
                "Repartitioning changes Spark's physical data distribution; "
                "it does not modify column values."
            ),
        }]

    def _repair_outliers(
        self,
        df: DataFrame,
        outlier_report: dict,
    ) -> tuple[DataFrame, list[dict[str, Any]]]:
        if self.policy.outlier_strategy not in {"cap", "none"}:
            raise ValueError(
                "Unsupported outlier_strategy. Expected one of: 'cap', 'none'."
            )
        unsupported_overrides = {
            col: strategy
            for col, strategy in self.policy.outlier_column_strategies.items()
            if strategy not in {"cap", "none"}
        }
        if unsupported_overrides:
            raise ValueError(
                "Unsupported outlier_column_strategies. Expected each column "
                f"strategy to be one of: 'cap', 'none'. Got: {unsupported_overrides}"
            )

        cleaned_df = df
        actions: list[dict[str, Any]] = []

        for col, stats in outlier_report.items():
            if col not in cleaned_df.columns:
                continue

            has_column_override = col in self.policy.outlier_column_strategies
            column_strategy = self.policy.outlier_column_strategies.get(
                col,
                self.policy.outlier_strategy,
            )
            if column_strategy == "none":
                if has_column_override and stats.get("outlier_count", 0) > 0:
                    actions.append({
                        "issue": "outliers",
                        "column": col,
                        "strategy": "none",
                        "outlier_count": stats.get("outlier_count", 0),
                        "outlier_ratio": stats.get("outlier_ratio", 0.0),
                        "message": "Outlier repair skipped by policy.",
                    })
                continue

            if stats.get("severity") not in self.policy.repair_outlier_severities:
                continue

            if not stats.get("repairable", True):
                actions.append({
                    "issue": "outliers",
                    "column": col,
                    "strategy": "skip_non_repairable",
                    "method": stats.get("method"),
                    "outlier_count": stats.get("outlier_count", 0),
                    "message": stats.get("message"),
                })
                continue

            lower_bound = stats.get("lower_bound")
            upper_bound = stats.get("upper_bound")
            if lower_bound is None or upper_bound is None:
                continue

            cleaned_df = cleaned_df.withColumn(
                col,
                F.when(F.col(col) < lower_bound, F.lit(lower_bound))
                .when(F.col(col) > upper_bound, F.lit(upper_bound))
                .otherwise(F.col(col)),
            )
            actions.append({
                "issue": "outliers",
                "column": col,
                "strategy": "cap",
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": stats.get("outlier_count", 0),
                "outlier_ratio": stats.get("outlier_ratio", 0.0),
            })

        return cleaned_df, actions

    def _align_schema(
        self,
        df: DataFrame,
        reference_schema: dict,
        schema_drift_report: dict,
    ) -> tuple[DataFrame, list[dict[str, Any]]]:
        cleaned_df = df
        actions: list[dict[str, Any]] = []

        for col in schema_drift_report.get("removed_columns", []):
            if col in cleaned_df.columns:
                continue

            data_type = reference_schema[col]["data_type"]
            cleaned_df = cleaned_df.withColumn(col, F.lit(None).cast(data_type))
            actions.append({
                "issue": "schema_drift",
                "column": col,
                "strategy": "add_missing_column",
                "data_type": data_type,
            })

        for change in schema_drift_report.get("type_changes", []):
            col = change["column"]
            if col not in cleaned_df.columns:
                continue

            reference_type = change["reference_type"]
            cleaned_df = cleaned_df.withColumn(col, F.col(col).cast(reference_type))
            actions.append({
                "issue": "schema_drift",
                "column": col,
                "strategy": "cast",
                "from_type": change["current_type"],
                "to_type": reference_type,
            })

        if self.policy.drop_added_columns:
            added_columns = [
                col for col in schema_drift_report.get("added_columns", [])
                if col in cleaned_df.columns
            ]
            if added_columns:
                cleaned_df = cleaned_df.drop(*added_columns)
                actions.append({
                    "issue": "schema_drift",
                    "columns": added_columns,
                    "strategy": "drop_added_columns",
                })

        return cleaned_df, actions

    def _numeric_fill_value(self, df: DataFrame, col: str) -> int | float | None:
        if self.policy.numeric_imputation == "median":
            values = df.approxQuantile(col, [0.5], 0.01)
            return values[0] if values else None

        if self.policy.numeric_imputation == "mean":
            return df.select(F.mean(F.col(col)).alias("mean")).collect()[0]["mean"]

        raise ValueError(
            "Unsupported numeric_imputation. Expected one of: 'median', 'mean'."
        )

    def _categorical_fill_value(self, df: DataFrame, col: str) -> str | None:
        if self.policy.categorical_imputation == "constant":
            return self.policy.categorical_fill_value

        if self.policy.categorical_imputation == "mode":
            mode_row = (
                df.filter(F.col(col).isNotNull())
                .groupBy(col)
                .count()
                .orderBy(F.desc("count"), F.asc(col))
                .limit(1)
                .collect()
            )
            return mode_row[0][col] if mode_row else None

        raise ValueError(
            "Unsupported categorical_imputation. Expected one of: 'constant', 'mode'."
        )
