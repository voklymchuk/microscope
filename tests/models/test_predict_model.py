import json
import pandas as pd
from joblib import dump
from sklearn.linear_model import LinearRegression
from microscope.models.model_config import TARGET_NAME
from microscope.models.predict_model import main
from microscope.utils import to_file


def test_evaluate(runner, df):
    to_file(df, "dataset.csv")
    X, y = df.drop(columns=[TARGET_NAME]), df[TARGET_NAME]
    linreg = LinearRegression()
    linreg.fit(X, y)
    dump(linreg, "linearReg")
    result = runner.invoke(main, ["evaluate", "dataset.csv", ".", "report.json"])
    assert result.exit_code == 0
    with open("report.json", "r") as f:
        report = json.load(f)
    assert "linearReg" in report
