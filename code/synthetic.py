from dataclasses import dataclass, field
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


@dataclass
class SyntheticIssueConfig:
    row_count: int = 1000
    seed: int = 42
    missing_rate: float = 0.05
    duplicate_rate: float = 0.02
    outlier_rate: float = 0.03
    skew_rate: float = 0.60
    schema_drift: bool = True
    label_noise_rate: float = 0.05
    missing_label_rate: float = 0.02


@dataclass
class SyntheticDataset:
    clean_df: DataFrame
    dirty_df: DataFrame
    reference_schema: dict[str, dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


class SyntheticDataGenerator:
    """
    Generates controlled Spark tabular datasets for AutoCleanML experiments.

    The clean dataset acts as a reference. The dirty dataset contains injected
    quality issues with metadata that describes the intended issue rates.
    """

    def __init__(self, spark: SparkSession, config: SyntheticIssueConfig | None = None):
        self.spark = spark
        self.config = config or SyntheticIssueConfig()
        self._validate_config(self.config)

    def generate_classification_dataset(self) -> SyntheticDataset:
        clean_df = self._base_dataset()
        reference_schema = self._schema_profile(clean_df)

        dirty_df = clean_df
        metadata: dict[str, Any] = {
            "config": self.config.__dict__.copy(),
            "issues": {},
        }

        dirty_df = self._inject_schema_drift(dirty_df, metadata)
        dirty_df = self._inject_missingness(dirty_df, metadata)
        dirty_df = self._inject_outliers(dirty_df, metadata)
        dirty_df = self._inject_key_skew(dirty_df, metadata)
        dirty_df = self._inject_label_noise(dirty_df, metadata)
        dirty_df = self._inject_missing_labels(dirty_df, metadata)
        dirty_df = self._inject_duplicates(dirty_df, metadata)

        return SyntheticDataset(
            clean_df=clean_df,
            dirty_df=dirty_df,
            reference_schema=reference_schema,
            metadata=metadata,
        )

    def generate_regression_dataset(self) -> SyntheticDataset:
        clean_df = self._base_regression_dataset()
        reference_schema = self._schema_profile(clean_df)

        dirty_df = clean_df
        metadata: dict[str, Any] = {
            "config": self.config.__dict__.copy(),
            "issues": {},
            "task_type": "regression",
        }

        dirty_df = self._inject_schema_drift(dirty_df, metadata)
        dirty_df = self._inject_missingness(dirty_df, metadata)
        dirty_df = self._inject_outliers(dirty_df, metadata)
        dirty_df = self._inject_key_skew(dirty_df, metadata)
        dirty_df = self._inject_regression_target_noise(dirty_df, metadata)
        dirty_df = self._inject_duplicates(dirty_df, metadata)

        return SyntheticDataset(
            clean_df=clean_df,
            dirty_df=dirty_df,
            reference_schema=reference_schema,
            metadata=metadata,
        )

    def _base_dataset(self) -> DataFrame:
        row_count = self.config.row_count
        seed = self.config.seed

        df = self.spark.range(row_count).withColumnRenamed("id", "row_id")
        df = df.withColumn("feature_num1", (F.rand(seed) * 100).cast("double"))
        df = df.withColumn("feature_num2", (F.rand(seed + 1) * 50).cast("double"))
        df = df.withColumn(
            "category",
            F.when((F.col("row_id") % 3) == 0, F.lit("A"))
            .when((F.col("row_id") % 3) == 1, F.lit("B"))
            .otherwise(F.lit("C")),
        )
        df = df.withColumn(
            "join_key",
            F.concat(F.lit("key_"), (F.col("row_id") % 20).cast("string")),
        )
        df = df.withColumn(
            "label",
            F.when(
                (F.col("feature_num1") + F.col("feature_num2")) > 75,
                F.lit("yes"),
            ).otherwise(F.lit("no")),
        )

        return df

    def _base_regression_dataset(self) -> DataFrame:
        row_count = self.config.row_count
        seed = self.config.seed

        df = self.spark.range(row_count).withColumnRenamed("id", "row_id")
        df = df.withColumn("feature_num1", (F.rand(seed) * 100).cast("double"))
        df = df.withColumn("feature_num2", (F.rand(seed + 1) * 50).cast("double"))
        df = df.withColumn(
            "category",
            F.when((F.col("row_id") % 3) == 0, F.lit("A"))
            .when((F.col("row_id") % 3) == 1, F.lit("B"))
            .otherwise(F.lit("C")),
        )
        df = df.withColumn(
            "join_key",
            F.concat(F.lit("key_"), (F.col("row_id") % 20).cast("string")),
        )
        df = df.withColumn(
            "target",
            (
                F.col("feature_num1") * F.lit(2.4)
                + F.col("feature_num2") * F.lit(1.7)
                + F.when(F.col("category") == "A", F.lit(15.0))
                .when(F.col("category") == "B", F.lit(-8.0))
                .otherwise(F.lit(3.0))
                + (F.rand(seed + 2) * 10.0)
            ).cast("double"),
        )

        return df

    def _inject_missingness(self, df: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        rate = self.config.missing_rate
        if rate == 0:
            return df

        dirty_df = df.withColumn(
            "feature_num1",
            F.when(F.rand(self.config.seed + 10) < rate, None).otherwise(
                F.col("feature_num1")
            ),
        )
        dirty_df = dirty_df.withColumn(
            "category",
            F.when(F.rand(self.config.seed + 11) < rate, None).otherwise(
                F.col("category")
            ),
        )
        metadata["issues"]["missingness"] = {
            "columns": ["feature_num1", "category"],
            "rate": rate,
        }
        return dirty_df

    def _inject_outliers(self, df: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        rate = self.config.outlier_rate
        if rate == 0:
            return df

        dirty_df = df.withColumn(
            "feature_num2",
            F.when(F.rand(self.config.seed + 20) < rate, F.lit(1000.0)).otherwise(
                F.col("feature_num2")
            ),
        )
        metadata["issues"]["outliers"] = {
            "column": "feature_num2",
            "rate": rate,
            "outlier_value": 1000.0,
        }
        return dirty_df

    def _inject_key_skew(self, df: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        rate = self.config.skew_rate
        if rate == 0:
            return df

        dirty_df = df.withColumn(
            "join_key",
            F.when(F.rand(self.config.seed + 30) < rate, F.lit("heavy_key")).otherwise(
                F.col("join_key")
            ),
        )
        metadata["issues"]["key_skew"] = {
            "column": "join_key",
            "heavy_key": "heavy_key",
            "rate": rate,
        }
        return dirty_df

    def _inject_label_noise(self, df: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        rate = self.config.label_noise_rate
        if rate == 0:
            return df

        dirty_df = df.withColumn(
            "label",
            F.when(
                F.rand(self.config.seed + 40) < rate,
                F.when(F.col("label") == "yes", F.lit("no")).otherwise(F.lit("yes")),
            ).otherwise(F.col("label")),
        )
        metadata["issues"]["label_noise"] = {
            "column": "label",
            "rate": rate,
            "strategy": "flip_binary_label",
        }
        return dirty_df

    def _inject_regression_target_noise(
        self,
        df: DataFrame,
        metadata: dict[str, Any],
    ) -> DataFrame:
        rate = self.config.label_noise_rate
        if rate == 0 or "target" not in df.columns:
            return df

        dirty_df = df.withColumn(
            "target",
            F.when(
                F.rand(self.config.seed + 40) < rate,
                F.col("target") + F.lit(100.0),
            ).otherwise(F.col("target")),
        )
        metadata["issues"]["target_noise"] = {
            "column": "target",
            "rate": rate,
            "strategy": "add_large_positive_error",
        }
        return dirty_df

    def _inject_missing_labels(
        self,
        df: DataFrame,
        metadata: dict[str, Any],
    ) -> DataFrame:
        rate = self.config.missing_label_rate
        if rate == 0:
            return df

        dirty_df = df.withColumn(
            "label",
            F.when(F.rand(self.config.seed + 50) < rate, None).otherwise(
                F.col("label")
            ),
        )
        metadata["issues"]["missing_labels"] = {
            "column": "label",
            "rate": rate,
        }
        return dirty_df

    def _inject_schema_drift(self, df: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        if not self.config.schema_drift:
            return df

        dirty_df = (
            df.withColumn("feature_num1", F.col("feature_num1").cast("string"))
            .withColumn("new_source_col", F.lit("synthetic_v2"))
        )
        metadata["issues"]["schema_drift"] = {
            "removed_columns": [],
            "type_changes": [{"column": "feature_num1", "to_type": "string"}],
            "added_columns": ["new_source_col"],
        }
        return dirty_df

    def _inject_duplicates(self, df: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        rate = self.config.duplicate_rate
        if rate == 0:
            return df

        duplicate_df = df.filter(F.rand(self.config.seed + 60) < rate)
        dirty_df = df.unionByName(duplicate_df)
        metadata["issues"]["duplicates"] = {
            "rate": rate,
        }
        return dirty_df

    def _schema_profile(self, df: DataFrame) -> dict[str, dict[str, Any]]:
        return {
            field.name: {
                "data_type": field.dataType.simpleString(),
                "nullable": field.nullable,
            }
            for field in df.schema.fields
        }

    # ------------------------------------------------------------------
    # Employee Attrition Dataset (classification)
    # ------------------------------------------------------------------

    def generate_employee_attrition_dataset(self) -> SyntheticDataset:
        clean_df = self._base_employee_dataset()
        reference_schema = self._schema_profile(clean_df)
        dirty_df = clean_df
        metadata: dict[str, Any] = {
            "config": self.config.__dict__.copy(),
            "issues": {},
            "task_type": "classification",
        }
        dirty_df = self._inject_employee_outliers(dirty_df, metadata)
        dirty_df = self._inject_employee_schema_drift(dirty_df, metadata)
        dirty_df = self._inject_employee_missingness(dirty_df, metadata)
        dirty_df = self._inject_employee_skew(dirty_df, metadata)
        dirty_df = self._inject_employee_label_noise(dirty_df, metadata)
        dirty_df = self._inject_duplicates(dirty_df, metadata)
        return SyntheticDataset(
            clean_df=clean_df,
            dirty_df=dirty_df,
            reference_schema=reference_schema,
            metadata=metadata,
        )

    def _base_employee_dataset(self) -> DataFrame:
        row_count = self.config.row_count
        seed = self.config.seed
        df = self.spark.range(row_count).withColumnRenamed("id", "row_id")

        df = df.withColumn("age", (F.rand(seed) * 43 + 22).cast("int"))
        df = df.withColumn("tenure_years", (F.rand(seed + 1) * 30).cast("double"))
        df = df.withColumn("performance_score", (F.rand(seed + 2) * 4.0 + 1.0).cast("double"))
        df = df.withColumn("hours_per_week", (F.rand(seed + 3) * 25 + 35).cast("int"))
        df = df.withColumn("training_hours", (F.rand(seed + 4) * 80).cast("int"))

        df = df.withColumn(
            "department",
            F.when(F.col("row_id") % 6 == 0, "Engineering")
            .when(F.col("row_id") % 6 == 1, "Sales")
            .when(F.col("row_id") % 6 == 2, "HR")
            .when(F.col("row_id") % 6 == 3, "Finance")
            .when(F.col("row_id") % 6 == 4, "Marketing")
            .otherwise("Operations"),
        )
        df = df.withColumn(
            "education_level",
            F.when(F.col("row_id") % 4 == 0, "HighSchool")
            .when(F.col("row_id") % 4 == 1, "Bachelor")
            .when(F.col("row_id") % 4 == 2, "Master")
            .otherwise("PhD"),
        )
        df = df.withColumn(
            "employment_type",
            F.when(F.col("row_id") % 6 < 4, "FullTime")
            .when(F.col("row_id") % 6 == 4, "PartTime")
            .otherwise("Contract"),
        )
        df = df.withColumn(
            "location",
            F.when(F.col("row_id") % 3 == 0, "Remote")
            .when(F.col("row_id") % 3 == 1, "OnSite")
            .otherwise("Hybrid"),
        )
        df = df.withColumn(
            "salary",
            (
                F.when(F.col("education_level") == "PhD", F.lit(100000.0))
                .when(F.col("education_level") == "Master", F.lit(75000.0))
                .when(F.col("education_level") == "Bachelor", F.lit(55000.0))
                .otherwise(F.lit(40000.0))
                + F.when(F.col("department") == "Engineering", F.lit(20000.0))
                .when(F.col("department") == "Finance", F.lit(15000.0))
                .when(F.col("department") == "Sales", F.lit(10000.0))
                .otherwise(F.lit(0.0))
                + F.col("tenure_years") * F.lit(1000.0)
                + (F.rand(seed + 7) * 20000.0 - 10000.0)
            ).cast("double"),
        )
        df = df.withColumn(
            "attrition",
            F.when(
                (F.col("performance_score") < 2.0)
                | (F.col("hours_per_week") > 57)
                | (F.col("salary") < 50000.0),
                F.lit("yes"),
            ).otherwise(F.lit("no")),
        )
        return df.drop("row_id")

    def _inject_employee_schema_drift(
        self, df: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        if not self.config.schema_drift:
            return df
        dirty_df = (
            df.withColumn("salary", F.col("salary").cast("string"))
            .withColumn("source_system", F.lit("hr_v2"))
        )
        metadata["issues"]["schema_drift"] = {
            "type_changes": [{"column": "salary", "to_type": "string"}],
            "added_columns": ["source_system"],
        }
        return dirty_df

    def _inject_employee_missingness(
        self, df: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        rate = self.config.missing_rate
        if rate == 0:
            return df
        dirty_df = df.withColumn(
            "performance_score",
            F.when(F.rand(self.config.seed + 10) < rate, None).otherwise(
                F.col("performance_score")
            ),
        ).withColumn(
            "department",
            F.when(F.rand(self.config.seed + 11) < rate, None).otherwise(
                F.col("department")
            ),
        )
        metadata["issues"]["missingness"] = {
            "columns": ["performance_score", "department"],
            "rate": rate,
        }
        return dirty_df

    def _inject_employee_outliers(
        self, df: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        rate = self.config.outlier_rate
        if rate == 0:
            return df
        dirty_df = df.withColumn(
            "salary",
            F.when(F.rand(self.config.seed + 20) < rate, F.lit(500000.0)).otherwise(
                F.col("salary")
            ),
        )
        metadata["issues"]["outliers"] = {
            "column": "salary",
            "rate": rate,
            "outlier_value": 500000.0,
        }
        return dirty_df

    def _inject_employee_skew(
        self, df: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        rate = self.config.skew_rate
        if rate == 0:
            return df
        dirty_df = df.withColumn(
            "location",
            F.when(F.rand(self.config.seed + 30) < rate, F.lit("OnSite")).otherwise(
                F.col("location")
            ),
        )
        metadata["issues"]["key_skew"] = {
            "column": "location",
            "heavy_key": "OnSite",
            "rate": rate,
        }
        return dirty_df

    def _inject_employee_label_noise(
        self, df: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        rate = self.config.label_noise_rate
        if rate == 0:
            return df
        dirty_df = df.withColumn(
            "attrition",
            F.when(
                F.rand(self.config.seed + 40) < rate,
                F.when(F.col("attrition") == "yes", F.lit("no")).otherwise(F.lit("yes")),
            ).otherwise(F.col("attrition")),
        )
        metadata["issues"]["label_noise"] = {
            "column": "attrition",
            "rate": rate,
            "strategy": "flip_binary_label",
        }
        return dirty_df

    # ------------------------------------------------------------------
    # House Price Dataset (regression)
    # ------------------------------------------------------------------

    def generate_house_price_dataset(self) -> SyntheticDataset:
        clean_df = self._base_house_price_dataset()
        reference_schema = self._schema_profile(clean_df)
        dirty_df = clean_df
        metadata: dict[str, Any] = {
            "config": self.config.__dict__.copy(),
            "issues": {},
            "task_type": "regression",
        }
        dirty_df = self._inject_house_schema_drift(dirty_df, metadata)
        dirty_df = self._inject_house_missingness(dirty_df, metadata)
        dirty_df = self._inject_house_outliers(dirty_df, metadata)
        dirty_df = self._inject_house_skew(dirty_df, metadata)
        dirty_df = self._inject_house_target_noise(dirty_df, metadata)
        dirty_df = self._inject_duplicates(dirty_df, metadata)
        return SyntheticDataset(
            clean_df=clean_df,
            dirty_df=dirty_df,
            reference_schema=reference_schema,
            metadata=metadata,
        )

    def _base_house_price_dataset(self) -> DataFrame:
        row_count = self.config.row_count
        seed = self.config.seed
        df = self.spark.range(row_count).withColumnRenamed("id", "row_id")

        df = df.withColumn("sqft_living", (F.rand(seed) * 4500 + 500).cast("double"))
        df = df.withColumn("sqft_lot", (F.rand(seed + 1) * 19000 + 1000).cast("double"))
        df = df.withColumn("bedrooms", (F.rand(seed + 2) * 6 + 1).cast("int"))
        df = df.withColumn("bathrooms", (F.rand(seed + 3) * 4.0 + 1.0).cast("double"))
        df = df.withColumn("age_years", (F.rand(seed + 4) * 100).cast("int"))
        df = df.withColumn("distance_to_center", (F.rand(seed + 5) * 29.5 + 0.5).cast("double"))
        df = df.withColumn("school_rating", (F.rand(seed + 6) * 9.0 + 1.0).cast("double"))

        df = df.withColumn(
            "neighborhood",
            F.when(F.col("row_id") % 5 == 0, "Downtown")
            .when(F.col("row_id") % 5 == 1, "Suburbs")
            .when(F.col("row_id") % 5 == 2, "Rural")
            .when(F.col("row_id") % 5 == 3, "Waterfront")
            .otherwise("Industrial"),
        )
        df = df.withColumn(
            "house_type",
            F.when(F.col("row_id") % 4 == 0, "SingleFamily")
            .when(F.col("row_id") % 4 == 1, "Condo")
            .when(F.col("row_id") % 4 == 2, "Townhouse")
            .otherwise("MultiFamily"),
        )
        df = df.withColumn(
            "condition",
            F.when(F.col("row_id") % 4 == 0, "Excellent")
            .when(F.col("row_id") % 4 == 1, "Good")
            .when(F.col("row_id") % 4 == 2, "Fair")
            .otherwise("Poor"),
        )
        df = df.withColumn(
            "price",
            F.greatest(
                F.lit(50000.0),
                (
                    F.col("sqft_living") * F.lit(200.0)
                    + F.col("sqft_lot") * F.lit(3.0)
                    + F.col("bedrooms").cast("double") * F.lit(15000.0)
                    + F.col("bathrooms") * F.lit(12000.0)
                    - F.col("age_years").cast("double") * F.lit(800.0)
                    + F.col("school_rating") * F.lit(18000.0)
                    - F.col("distance_to_center") * F.lit(4000.0)
                    + F.when(F.col("neighborhood") == "Waterfront", F.lit(150000.0))
                    .when(F.col("neighborhood") == "Downtown", F.lit(80000.0))
                    .when(F.col("neighborhood") == "Suburbs", F.lit(20000.0))
                    .when(F.col("neighborhood") == "Industrial", F.lit(-30000.0))
                    .otherwise(F.lit(0.0))
                    + F.when(F.col("house_type") == "SingleFamily", F.lit(30000.0))
                    .when(F.col("house_type") == "Condo", F.lit(-10000.0))
                    .otherwise(F.lit(0.0))
                    + F.when(F.col("condition") == "Excellent", F.lit(40000.0))
                    .when(F.col("condition") == "Good", F.lit(10000.0))
                    .when(F.col("condition") == "Fair", F.lit(-15000.0))
                    .otherwise(F.lit(-35000.0))
                    + (F.rand(seed + 10) * 40000.0 - 20000.0)
                ).cast("double"),
            ),
        )
        return df.drop("row_id")

    def _inject_house_schema_drift(
        self, df: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        if not self.config.schema_drift:
            return df
        dirty_df = (
            df.withColumn("sqft_living", F.col("sqft_living").cast("string"))
            .withColumn("data_source", F.lit("mls_v2"))
        )
        metadata["issues"]["schema_drift"] = {
            "type_changes": [{"column": "sqft_living", "to_type": "string"}],
            "added_columns": ["data_source"],
        }
        return dirty_df

    def _inject_house_missingness(
        self, df: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        rate = self.config.missing_rate
        if rate == 0:
            return df
        dirty_df = df.withColumn(
            "school_rating",
            F.when(F.rand(self.config.seed + 10) < rate, None).otherwise(
                F.col("school_rating")
            ),
        ).withColumn(
            "neighborhood",
            F.when(F.rand(self.config.seed + 11) < rate, None).otherwise(
                F.col("neighborhood")
            ),
        )
        metadata["issues"]["missingness"] = {
            "columns": ["school_rating", "neighborhood"],
            "rate": rate,
        }
        return dirty_df

    def _inject_house_outliers(
        self, df: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        rate = self.config.outlier_rate
        if rate == 0:
            return df
        dirty_df = df.withColumn(
            "sqft_lot",
            F.when(F.rand(self.config.seed + 20) < rate, F.lit(100000.0)).otherwise(
                F.col("sqft_lot")
            ),
        )
        metadata["issues"]["outliers"] = {
            "column": "sqft_lot",
            "rate": rate,
            "outlier_value": 100000.0,
        }
        return dirty_df

    def _inject_house_skew(
        self, df: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        rate = self.config.skew_rate
        if rate == 0:
            return df
        dirty_df = df.withColumn(
            "neighborhood",
            F.when(F.rand(self.config.seed + 30) < rate, F.lit("Suburbs")).otherwise(
                F.col("neighborhood")
            ),
        )
        metadata["issues"]["key_skew"] = {
            "column": "neighborhood",
            "heavy_key": "Suburbs",
            "rate": rate,
        }
        return dirty_df

    def _inject_house_target_noise(
        self, df: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        rate = self.config.label_noise_rate
        if rate == 0 or "price" not in df.columns:
            return df
        dirty_df = df.withColumn(
            "price",
            F.when(
                F.rand(self.config.seed + 40) < rate,
                F.col("price") + F.lit(300000.0),
            ).otherwise(F.col("price")),
        )
        metadata["issues"]["target_noise"] = {
            "column": "price",
            "rate": rate,
            "strategy": "add_large_positive_error",
        }
        return dirty_df

    def _validate_config(self, config: SyntheticIssueConfig) -> None:
        if config.row_count <= 0:
            raise ValueError("row_count must be positive.")

        for field_name in (
            "missing_rate",
            "duplicate_rate",
            "outlier_rate",
            "skew_rate",
            "label_noise_rate",
            "missing_label_rate",
        ):
            value = getattr(config, field_name)
            if value < 0 or value > 1:
                raise ValueError(f"{field_name} must be between 0 and 1.")
