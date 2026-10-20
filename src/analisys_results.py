import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from constants import *


def load(file: Path) -> pd.DataFrame:
    df = pd.read_csv(file)

    df["error"] = df["source"] != df["prediction"]

    return df


def model_error_analysis(model_name: str, df: pd.DataFrame) -> dict:
    res = {
        "model_name": model_name,
        "n_reviews":len(df),
        "accuracy": accuracy_score(df[Y_COL], df["prediction"]),
        "f1_score": f1_score(df[Y_COL], df["prediction"]),
        "n_errors": df["error"].sum(),
        "actual_0_predicted_1": len(df[(df[Y_COL] == 0) & (df["prediction"] == 1)]),
        "actual_1_predicted_0": len(df[(df[Y_COL] == 1) & (df["prediction"] == 0)]),
        "error_rate": df["error"].mean(),
    }
    return res


def main():
    prediction_files = {
        "naive_bayes": RESULTS_DIR / ModelNames.NAIVE_BAYES.value / "naive_bayes_3_predictions.csv",
        "gradient_boosting": RESULTS_DIR / ModelNames.GRADIENT_BOOSTING.value / "gradient_boosting_predictions.csv",
        "random_forest": RESULTS_DIR / ModelNames.RANDOM_FOREST.value / "random_forest_predictions.csv",
    }
    predictions = {
        name: load(file)
        for name, file in prediction_files.items()
    }
    comparison = pd.DataFrame([
        model_error_analysis(name, df)
        for name, df in predictions.items()
    ])
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()