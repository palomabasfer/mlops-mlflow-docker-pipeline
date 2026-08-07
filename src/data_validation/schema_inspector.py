import pandas as pd
from typing import Dict, Any

def validate_dataframe_schema(df: pd.DataFrame, expected_columns: list) -> Dict[str, Any]:
    missing = set(expected_columns) - set(df.columns)
    return {
        "valid": len(missing) == 0,
        "missing_columns": list(missing)
    }
