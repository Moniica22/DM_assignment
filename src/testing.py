import joblib
import pandas as pd

from constants import *


def test_model(model_type: str, model_name: str):
    # 1 - Load pipeline and test data
    model_path = MODELS_DIR / model_type / f"{model_name}.joblib"
    pipeline = joblib.load(model_path)
    test_df = pd.read_csv(TEST_DATA_FILE)

    # 2 - Predict
    test_df["prediction"] = pipeline.predict(
        test_df[X_COL].fillna("")
    )

    # 3 - Save results
    output_dir = RESULTS_DIR / model_type
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{model_name}_predictions.csv"
    test_df.to_csv(output_path, index=False)

    print(f"Saved predictions: {output_path}")

    return




def main():
    model_type = ModelNames.NAIVE_BAYES.value
    model_name = "naive_bayes"

    test_model(
        model_type=model_type,
        model_name=model_name,
    )


if __name__ == "__main__":
    main()