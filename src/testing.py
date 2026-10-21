from typing import Tuple

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from constants import *

def print_classification_report(y_true, y_pred) -> Tuple[str, pd.DataFrame]:
    report = classification_report(y_true, y_pred, digits=4)
    print(report)

    matrix = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=[0, 1]),
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )
    print("Confusion matrix:")
    print(matrix)

    return report, matrix


def test_model(model_type: str, model_name: str) -> None:
    # 1 - Load pipeline and test data
    model_path = MODELS_DIR / model_type / f"{model_name}.joblib"
    pipeline = joblib.load(model_path)
    test_df = pd.read_csv(TEST_DATA_FILE)

    # 2 - Predict
    test_df["prediction"] = pipeline.predict(
        test_df[X_COL].fillna("")
    )

    # 3 - Get classification report and confusion matrix
    report, conf_matrix = print_classification_report(y_pred=test_df["prediction"], y_true=test_df[Y_COL])

    # 4 - Save results
    output_dir = RESULTS_DIR / model_type
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{model_name}_predictions.csv"
    test_df.to_csv(output_path, index=False)

    (output_dir / f"{model_name}_metrics.txt").write_text(
        report, encoding="utf-8"
    )

    conf_matrix.to_csv(output_dir / f"{model_name}_confusion_matrix.csv")

    print(f"Saved predictions: {output_path}")

    return




def main():
    model_type = ModelNames.LOGISTIC_REGRESSION_L1
    model_name = "logistic_regression_l1"

    test_model(
        model_type=model_type,
        model_name=model_name,
    )


if __name__ == "__main__":
    main()