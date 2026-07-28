import pytest
import numpy as np
import pandas as pd
from src.data.pipeline import MLOpsDataPipeline
from src.models.tracking import MLflowModelTracker
from src.monitoring.drift_detector import StatisticalDriftDetector


def test_data_pipeline_generation_and_prep():
    pipeline = MLOpsDataPipeline(seed=42)
    df = pipeline.generate_data(num_samples=100)
    assert len(df) == 100
    assert "target" in df.columns

    X_train, X_test, y_train, y_test = pipeline.prepare_train_test(df)
    assert len(X_train) == 80
    assert len(X_test) == 20


def test_mlflow_tracking_and_metrics():
    pipeline = MLOpsDataPipeline(seed=42)
    df = pipeline.generate_data(num_samples=100)
    X_train, X_test, y_train, y_test = pipeline.prepare_train_test(df)

    tracker = MLflowModelTracker(experiment_name="PyTest_MLOps_Experiment")
    params = {"n_estimators": 10, "max_depth": 3, "random_state": 42}
    metrics, run_id = tracker.train_and_log(X_train, X_test, y_train, y_test, params)

    assert "r2_score" in metrics
    assert "rmse" in metrics
    assert run_id is not None


def test_drift_detection():
    pipeline = MLOpsDataPipeline(seed=42)
    df_ref = pipeline.generate_data(num_samples=100)

    # Shift current data to simulate drift
    df_curr = pipeline.generate_data(num_samples=100)
    df_curr["feature_1"] += 5.0

    detector = StatisticalDriftDetector(alpha=0.05)
    results = detector.detect_drift(df_ref, df_curr)

    assert results["overall_drift_detected"] is True
    assert "feature_1" in results["drifted_features"]
