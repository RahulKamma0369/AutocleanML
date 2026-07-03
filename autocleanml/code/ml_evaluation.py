from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
)
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


@dataclass
class MLEvaluationResult:
    raw_metrics: dict[str, Any]
    cleaned_metrics: dict[str, Any]
    delta: dict[str, Any]


class SparkMLClassificationEvaluator:
    """
    Spark ML evaluator for raw-vs-cleaned classification experiments.

    Raw data may contain nulls, so rows with null feature or label values are
    dropped only for ML compatibility. The row counts are reported to make that
    baseline choice explicit.
    """

    def __init__(
        self,
        seed: int = 42,
        train_fraction: float = 0.8,
        validation_folds: int = 1,
    ):
        if train_fraction <= 0 or train_fraction >= 1:
            raise ValueError("train_fraction must be between 0 and 1.")
        if validation_folds < 1:
            raise ValueError("validation_folds must be at least 1.")

        self.seed = seed
        self.train_fraction = train_fraction
        self.validation_folds = validation_folds

    def evaluate_logistic_regression(
        self,
        raw_df: DataFrame,
        cleaned_df: DataFrame,
        label_col: str,
        numeric_cols: list[str],
        categorical_cols: list[str],
    ) -> MLEvaluationResult:
        raw_metrics = self._fit_and_score(
            df=raw_df,
            label_col=label_col,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            seed=self.seed,
            include_stability=True,
        )
        cleaned_metrics = self._fit_and_score(
            df=cleaned_df,
            label_col=label_col,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            seed=self.seed,
            include_stability=True,
        )

        return MLEvaluationResult(
            raw_metrics=raw_metrics,
            cleaned_metrics=cleaned_metrics,
            delta=self._metric_delta(raw_metrics, cleaned_metrics),
        )

    def _fit_and_score(
        self,
        df: DataFrame,
        label_col: str,
        numeric_cols: list[str],
        categorical_cols: list[str],
        seed: int,
        include_stability: bool,
    ) -> dict[str, Any]:
        selected_cols = numeric_cols + categorical_cols + [label_col]
        ml_df = df.select(*selected_cols)
        for col in numeric_cols:
            ml_df = ml_df.withColumn(col, F.col(col).cast("double"))
        ml_df = ml_df.dropna()
        ml_row_count = ml_df.count()
        if ml_row_count == 0:
            raise ValueError("No complete rows available for ML evaluation.")

        train_df, test_df = ml_df.randomSplit(
            [self.train_fraction, 1.0 - self.train_fraction],
            seed=seed,
        )
        train_row_count = train_df.count()
        test_row_count = test_df.count()
        if train_row_count == 0 or test_row_count == 0:
            raise ValueError(
                "Train/test split produced an empty split. Use more rows or "
                "adjust train_fraction."
            )

        pipeline = self._build_pipeline(label_col, numeric_cols, categorical_cols)
        model = pipeline.fit(train_df)
        predictions = model.transform(test_df)

        return {
            "ml_row_count": ml_row_count,
            "train_row_count": train_row_count,
            "test_row_count": test_row_count,
            "accuracy": self._evaluate(predictions, "accuracy"),
            "f1": self._evaluate(predictions, "f1"),
            "weighted_precision": self._evaluate(predictions, "weightedPrecision"),
            "weighted_recall": self._evaluate(predictions, "weightedRecall"),
            "auc": self._evaluate_auc(predictions),
            "fold_stability": (
                self._fold_stability(
                    df=ml_df,
                    label_col=label_col,
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                )
                if include_stability
                else None
            ),
        }

    def _build_pipeline(
        self,
        label_col: str,
        numeric_cols: list[str],
        categorical_cols: list[str],
    ) -> Pipeline:
        label_indexer = StringIndexer(
            inputCol=label_col,
            outputCol="indexed_label",
            handleInvalid="skip",
        )

        category_indexers = [
            StringIndexer(
                inputCol=col,
                outputCol=f"{col}_idx",
                handleInvalid="keep",
            )
            for col in categorical_cols
        ]
        category_encoders = [
            OneHotEncoder(
                inputCol=f"{col}_idx",
                outputCol=f"{col}_vec",
                handleInvalid="keep",
            )
            for col in categorical_cols
        ]
        assembler = VectorAssembler(
            inputCols=numeric_cols + [f"{col}_vec" for col in categorical_cols],
            outputCol="features",
        )
        classifier = LogisticRegression(
            featuresCol="features",
            labelCol="indexed_label",
            predictionCol="prediction",
            maxIter=50,
            regParam=0.0,
            elasticNetParam=0.0,
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

    def _evaluate(self, predictions: DataFrame, metric_name: str) -> float:
        evaluator = MulticlassClassificationEvaluator(
            labelCol="indexed_label",
            predictionCol="prediction",
            metricName=metric_name,
        )
        return round(evaluator.evaluate(predictions), 4)

    def _evaluate_auc(self, predictions: DataFrame) -> float | None:
        label_count = predictions.select("indexed_label").distinct().count()
        if label_count != 2:
            return None

        evaluator = BinaryClassificationEvaluator(
            labelCol="indexed_label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC",
        )
        return round(evaluator.evaluate(predictions), 4)

    def _fold_stability(
        self,
        df: DataFrame,
        label_col: str,
        numeric_cols: list[str],
        categorical_cols: list[str],
    ) -> dict[str, Any] | None:
        if self.validation_folds < 2:
            return None

        fold_metrics = []
        for fold in range(self.validation_folds):
            metrics = self._fit_and_score(
                df=df,
                label_col=label_col,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                seed=self.seed + fold + 1,
                include_stability=False,
            )
            fold_metrics.append(
                {
                    "fold": fold + 1,
                    "accuracy": metrics["accuracy"],
                    "f1": metrics["f1"],
                    "auc": metrics["auc"],
                }
            )

        return _summarize_fold_metrics(fold_metrics, ["accuracy", "f1", "auc"])

    def _metric_delta(
        self,
        raw_metrics: dict[str, Any],
        cleaned_metrics: dict[str, Any],
    ) -> dict[str, float | int | None]:
        delta: dict[str, float | int | None] = {}
        for metric in [
            "accuracy",
            "f1",
            "auc",
            "weighted_precision",
            "weighted_recall",
        ]:
            raw_value = raw_metrics.get(metric)
            cleaned_value = cleaned_metrics.get(metric)
            delta[metric] = (
                round(cleaned_value - raw_value, 4)
                if raw_value is not None and cleaned_value is not None
                else None
            )
        delta["ml_row_count"] = (
            cleaned_metrics["ml_row_count"] - raw_metrics["ml_row_count"]
        )
        return delta


class SparkMLRegressionEvaluator:
    """
    Spark ML evaluator for raw-vs-cleaned regression experiments.
    """

    def __init__(
        self,
        seed: int = 42,
        train_fraction: float = 0.8,
        validation_folds: int = 1,
    ):
        if train_fraction <= 0 or train_fraction >= 1:
            raise ValueError("train_fraction must be between 0 and 1.")
        if validation_folds < 1:
            raise ValueError("validation_folds must be at least 1.")

        self.seed = seed
        self.train_fraction = train_fraction
        self.validation_folds = validation_folds

    def evaluate_linear_regression(
        self,
        raw_df: DataFrame,
        cleaned_df: DataFrame,
        label_col: str,
        numeric_cols: list[str],
        categorical_cols: list[str],
    ) -> MLEvaluationResult:
        raw_metrics = self._fit_and_score(
            df=raw_df,
            label_col=label_col,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            seed=self.seed,
            include_stability=True,
        )
        cleaned_metrics = self._fit_and_score(
            df=cleaned_df,
            label_col=label_col,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            seed=self.seed,
            include_stability=True,
        )

        return MLEvaluationResult(
            raw_metrics=raw_metrics,
            cleaned_metrics=cleaned_metrics,
            delta={
                "rmse": round(cleaned_metrics["rmse"] - raw_metrics["rmse"], 4),
                "mae": round(cleaned_metrics["mae"] - raw_metrics["mae"], 4),
                "r2": round(cleaned_metrics["r2"] - raw_metrics["r2"], 4),
                "ml_row_count": (
                    cleaned_metrics["ml_row_count"] - raw_metrics["ml_row_count"]
                ),
            },
        )

    def _fit_and_score(
        self,
        df: DataFrame,
        label_col: str,
        numeric_cols: list[str],
        categorical_cols: list[str],
        seed: int,
        include_stability: bool,
    ) -> dict[str, Any]:
        selected_cols = numeric_cols + categorical_cols + [label_col]
        ml_df = df.select(*selected_cols).withColumn(
            label_col,
            F.col(label_col).cast("double"),
        )
        for col in numeric_cols:
            ml_df = ml_df.withColumn(col, F.col(col).cast("double"))
        ml_df = ml_df.dropna()
        ml_row_count = ml_df.count()
        if ml_row_count == 0:
            raise ValueError("No complete rows available for ML evaluation.")

        train_df, test_df = ml_df.randomSplit(
            [self.train_fraction, 1.0 - self.train_fraction],
            seed=seed,
        )
        train_row_count = train_df.count()
        test_row_count = test_df.count()
        if train_row_count == 0 or test_row_count == 0:
            raise ValueError(
                "Train/test split produced an empty split. Use more rows or "
                "adjust train_fraction."
            )

        pipeline = self._build_pipeline(label_col, numeric_cols, categorical_cols)
        model = pipeline.fit(train_df)
        predictions = model.transform(test_df)

        return {
            "ml_row_count": ml_row_count,
            "train_row_count": train_row_count,
            "test_row_count": test_row_count,
            "rmse": self._evaluate(predictions, label_col, "rmse"),
            "mae": self._evaluate(predictions, label_col, "mae"),
            "r2": self._evaluate(predictions, label_col, "r2"),
            "fold_stability": (
                self._fold_stability(
                    df=ml_df,
                    label_col=label_col,
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                )
                if include_stability
                else None
            ),
        }

    def _build_pipeline(
        self,
        label_col: str,
        numeric_cols: list[str],
        categorical_cols: list[str],
    ) -> Pipeline:
        category_indexers = [
            StringIndexer(
                inputCol=col,
                outputCol=f"{col}_idx",
                handleInvalid="keep",
            )
            for col in categorical_cols
        ]
        category_encoders = [
            OneHotEncoder(
                inputCol=f"{col}_idx",
                outputCol=f"{col}_vec",
                handleInvalid="keep",
            )
            for col in categorical_cols
        ]
        assembler = VectorAssembler(
            inputCols=numeric_cols + [f"{col}_vec" for col in categorical_cols],
            outputCol="features",
        )
        regressor = LinearRegression(
            featuresCol="features",
            labelCol=label_col,
            predictionCol="prediction",
            maxIter=50,
            regParam=0.0,
            elasticNetParam=0.0,
        )

        return Pipeline(
            stages=[
                *category_indexers,
                *category_encoders,
                assembler,
                regressor,
            ]
        )

    def _evaluate(
        self,
        predictions: DataFrame,
        label_col: str,
        metric_name: str,
    ) -> float:
        evaluator = RegressionEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName=metric_name,
        )
        return round(evaluator.evaluate(predictions), 4)

    def _fold_stability(
        self,
        df: DataFrame,
        label_col: str,
        numeric_cols: list[str],
        categorical_cols: list[str],
    ) -> dict[str, Any] | None:
        if self.validation_folds < 2:
            return None

        fold_metrics = []
        for fold in range(self.validation_folds):
            metrics = self._fit_and_score(
                df=df,
                label_col=label_col,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                seed=self.seed + fold + 1,
                include_stability=False,
            )
            fold_metrics.append(
                {
                    "fold": fold + 1,
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                }
            )

        return _summarize_fold_metrics(fold_metrics, ["rmse", "mae", "r2"])


def _summarize_fold_metrics(
    fold_metrics: list[dict[str, Any]],
    metric_names: list[str],
) -> dict[str, Any]:
    summary = {"folds": fold_metrics, "summary": {}}
    for metric_name in metric_names:
        values = [
            fold[metric_name]
            for fold in fold_metrics
            if fold.get(metric_name) is not None
        ]
        summary["summary"][metric_name] = (
            {
                "mean": round(mean(values), 4),
                "stddev": round(pstdev(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }
            if values
            else None
        )
    return summary
