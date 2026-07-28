import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MLOpsDataPipeline:
    """Automated data pipeline for data ingestion, cleaning, scaling, and splitting."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.scaler = StandardScaler()

    def generate_data(self, num_samples: int = 1000) -> pd.DataFrame:
        np.random.seed(self.seed)
        num_features = 5
        X = np.random.randn(num_samples, num_features)
        weights = np.array([2.5, -1.5, 0.5, 3.0, -0.8])
        bias = 10.0
        noise = np.random.normal(0, 0.5, size=num_samples)

        y = X @ weights + bias + noise

        feature_names = [f"feature_{i}" for i in range(1, num_features + 1)]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
        return df

    def prepare_train_test(
        self, df: pd.DataFrame, target_col: str = "target", test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test
