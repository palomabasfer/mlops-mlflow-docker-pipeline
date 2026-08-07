import os
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class MLflowModelTracker:
    """Manages MLflow experiment tracking, hyperparameter logging, and model registry."""

    def __init__(self, experiment_name: str = "MLOps_Production_Pipeline"):
        self.experiment_name = experiment_name
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_experiment(experiment_name)
            except Exception:
                pass

    def train_and_log(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        params: Dict[str, Any]
    ) -> Tuple[Dict[str, float], str]:
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2_score": r2}
        run_id = "mock_mlflow_run_id_01"

        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run() as run:
                    mlflow.log_params(params)
                    mlflow.log_metrics(metrics)
                    mlflow.sklearn.log_model(model, "model")
                    run_id = run.info.run_id
            except Exception:
                pass

        return metrics, run_id
