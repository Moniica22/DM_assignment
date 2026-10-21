import argparse 

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar

from constants import *


MODEL_FILES = {
    "naive_bayes": MODELS_DIR / "naive_bayes" / "naive_bayes_3.joblib",
    "logistic_regression": MODELS_DIR / "logistic_regression_l1" / "logistic_regression_l1.joblib",
    "classification_tree": MODELS_DIR / "classification_tree" / "classification_tree.joblib",
    "random_forest": MODELS_DIR / "random_forest" / "random_forest.joblib",
    "gradient_boosting": MODELS_DIR / "gradient_boosting" / "gradient_boosting.joblib",
}

def load_model(model_name:str):
    path = Path(MODEL_FILES[model_name])
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    return joblib.load(path)


def load_test():
    df = pd.read_csv(TEST_DATA_FILE)
    return df


def metrics(model, y_true, y_pred):
    return {
        "model": model,
        "n_reviews": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "n_correct": int((y_true == y_pred).sum()),
        "n_errors": int((y_true != y_pred).sum())
    }


def save_table(df, path, index):
    df.to_csv(path, index=index)


def run_sentiment(models, output_dir):
    model = models[0]

    df = load_test()
    sentiment = df["Sentiment"].replace({"POSITIVE": "POS", "NEGATIVE": "NEG"})

    pred = load_model(model).predict(df[X_COL])
    res = []

    for label in ["POS", "NEG"]:
        group = df[sentiment == label]

        y_true = group[Y_COL]
        y_pred = pred[sentiment == label]

        row = metrics(model, y_true, y_pred)

        row["sentiment"] = label

        res.append(row)

    metrics_table = pd.DataFrame(res).set_index("sentiment")

    comparison_table = metrics_table[["n_correct", "n_errors"]]

    test = chi2_contingency(comparison_table, correction=False)

    test_results = pd.DataFrame([{
        "test": "Pearson chi-square",
        "chi_square": test.statistic,
        "p_value": test.pvalue,
        "alpha": 0.01,
        "significant": test.pvalue < 0.01,
        "positive_minus_negative_accuracy": (float(metrics_table.loc["POS", "accuracy"]) 
                                    - float(metrics_table.loc["NEG", "accuracy"])),
        }])
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_table.to_csv(output_dir / "metrics_by_sentiment.csv")
    comparison_table.to_csv(output_dir / "correctness_table.csv")
    test_results.to_csv(output_dir / "accuracy_test.csv", index=False)

    print("\nMetrics by sentiment:")
    print(metrics_table.to_string())

    print("\nAccuracy comparison:")
    print(test_results.to_string(index=False))


def run_comparison(models, output_dir):
    if len(models) != 2:
        raise ValueError("McNemar's test requires exactly two models.")

    model_a, model_b = models

    df = load_test()
    y_true = df[Y_COL].to_numpy()

    y_pred_a = load_model(model_a).predict(df[X_COL].fillna(""))
    y_pred_b = load_model(model_b).predict(df[X_COL].fillna(""))

    a_correct = y_pred_a == y_true
    b_correct = y_pred_b == y_true

    contingency_table = np.array([
        [
            np.sum(a_correct & b_correct),
            np.sum(a_correct & ~b_correct),
        ],
        [
            np.sum(~a_correct & b_correct),
            np.sum(~a_correct & ~b_correct),
        ],
    ])

    result = mcnemar(
        contingency_table,
        exact=True,
    )

    alpha = 0.05 #Arbitrary

    contingency_df = pd.DataFrame(
        contingency_table,
        index=[
            f"{model_a}_correct",
            f"{model_a}_incorrect",
        ],
        columns=[
            f"{model_b}_correct",
            f"{model_b}_incorrect",
        ],
    )

    test_results = pd.DataFrame([{
        "test": "McNemar exact test",
        "model_a": model_a,
        "model_b": model_b,
        "n_reviews": len(y_true),
        "model_a_accuracy": float(np.mean(a_correct)),
        "model_b_accuracy": float(np.mean(b_correct)),
        "both_correct": int(contingency_table[0, 0]),
        "model_a_correct_model_b_incorrect": int(contingency_table[0, 1]),
        "model_a_incorrect_model_b_correct": int(contingency_table[1, 0]),
        "both_incorrect": int(contingency_table[1, 1]),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "alpha": alpha,
        "significant": bool(result.pvalue < alpha),
    }])

    output_dir.mkdir(parents=True, exist_ok=True)

    contingency_df.to_csv(output_dir / "mcnemar_contingency_table.csv")
    test_results.to_csv(output_dir / "mcnemar_test.csv", index=False)

    print("\nMcNemar contingency table:")
    print(contingency_df.to_string())

    print("\nMcNemar test:")
    print(test_results.to_string(index=False))




def main():
    parser = argparse.ArgumentParser(description=__doc__) 
    parser.add_argument("--mode", required=True, choices=["comparison", "sentiment", "feature_importance"])
    parser.add_argument("--models", required=True, nargs="+", choices=list(MODEL_FILES))
    args = parser.parse_args()

    if args.mode == "sentiment" and args.models != ["random_forest"]:
        parser.error("Sentiment analysis requires --models random_forest.")

    output_dir = RESULTS_DIR / "analysis" / args.mode / "_vs_".join(args.models)
    output_dir.mkdir(parents=True, exist_ok=True)

    functions = {
        "comparison": run_comparison,
        "sentiment": run_sentiment,
        } # add the other 2 funs 
    functions[args.mode](args.models, output_dir)

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()