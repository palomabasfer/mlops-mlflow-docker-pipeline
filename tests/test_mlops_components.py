import pandas as pd
from src.registry.model_promoter import MLflowModelPromoter
from src.data_validation.schema_inspector import validate_dataframe_schema

def test_promoter():
    p = MLflowModelPromoter()
    res = p.promote_model("churn_model", 1, "Production")
    assert res["stage"] == "Production"

def test_schema_validator():
    df = pd.DataFrame({"a": [1]})
    res = validate_dataframe_schema(df, ["a"])
    assert res["valid"] is True
