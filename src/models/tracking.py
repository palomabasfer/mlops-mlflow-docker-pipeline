import os
import numpy as np
from typing import Dict, Any
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn


class MLflowModelTracker:
    """Manages MLflow experiment tracking, hyperparameter logging, and model registry."""

    def __init__(self, experiment_name: str = "MLOps_Production_Pipeline"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def train_and_log(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        params: Dict[str, Any]
    ) -> Tuple[Dict[str, float], str]:
        with mlflow.start_run() as run:
            model = GradientBoostingRegressor(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            mse = float(mean_squared_error(y_test, y_pred))
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))

            metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2_score": r2}

            # Log parameters and metrics
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            # Log model artifact
            mlflow.sklearn.log_model(model, "model")

            return metrics, run.info.run_id
