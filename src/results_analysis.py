import argparse 

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from typing import List

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

def save_top_features(df_top_features:pd.DataFrame, model_name:str, output_dir) -> None:

    path = output_dir / f"top_features_{model_name}.csv"
    df_top_features.to_csv(path, index=False)
    print(f"results saved to {path}")

    return

def get_NB_feature_importance(classifier, feature_names, num_top_features):
    importance_scores = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
    abs_importance_scores = np.abs(importance_scores)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "score": importance_scores,
        "abs_score": abs_importance_scores,
        "favours_class": ""
    })
    importance_df.loc[importance_df['score'] > 0, 'favours_class'] = '1'
    importance_df.loc[importance_df['score'] < 0, 'favours_class'] = '0'
    importance_df.loc[importance_df['score'] == 0, 'favours_class'] = 'neither'

    top_features = importance_df.sort_values(by="abs_score", ascending=False).iloc[:num_top_features].copy()
    print(top_features)

    return top_features


def get_LR_feature_importance(classifier, feature_names, num_top_features):
    importance_scores = classifier.coef_[0]
    abs_importance_scores = np.abs(importance_scores)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "score": importance_scores,
        "abs_score": abs_importance_scores,
        "favours_class": ""
    })
    importance_df.loc[importance_df['score'] > 0, 'favours_class'] = '1'
    importance_df.loc[importance_df['score'] < 0, 'favours_class'] = '0'
    importance_df.loc[importance_df['score'] == 0, 'favours_class'] = 'neither'

    top_features = importance_df.sort_values(by="abs_score", ascending=False).iloc[:num_top_features].copy()
    print(top_features)

    return top_features


def get_CT_feature_importance(classifier, feature_names, num_top_features):
    importance_scores = classifier.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "score": importance_scores,
    })

    top_features = importance_df.sort_values(by="score", ascending=False).iloc[:num_top_features].copy()
    print(top_features)

    return top_features

def get_RF_feature_importance(classifier, feature_names, num_top_features):
    importance_scores = classifier.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "score": importance_scores,
    })

    top_features = importance_df.sort_values(by="score", ascending=False).iloc[:num_top_features].copy()
    print(top_features)

    return top_features

def get_GB_feature_importance(classifier, feature_names, num_top_features):
    importance_scores = classifier.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "score": importance_scores,
    })

    top_features = importance_df.sort_values(by="score", ascending=False).iloc[:num_top_features].copy()
    print(top_features)

    return top_features



def run_feature_importance(model_names: List[str], output_dir, num_top_features=5) -> None:
    for model_name in model_names:
        pipeline = load_model(model_name)

        feature_names = pipeline[:-1].get_feature_names_out()
        classifier = pipeline.named_steps["classifier"]

        if model_name == "naive_bayes":
            top_features = get_NB_feature_importance(classifier, feature_names, num_top_features)

        elif model_name == "logistic_regression":
            top_features = get_LR_feature_importance(classifier, feature_names, num_top_features)

        elif model_name == "classification_tree":
            top_features = get_CT_feature_importance(classifier, feature_names, num_top_features)

        elif model_name == "random_forest":
            top_features = get_RF_feature_importance(classifier, feature_names, num_top_features)

        elif model_name == "gradient_boosting":
            top_features = get_GB_feature_importance(classifier, feature_names, num_top_features)

        else:
            raise ValueError(f"Unsupported model type: {model_name}")

        save_top_features(top_features, model_name, output_dir)

    return


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




def main():
    parser = argparse.ArgumentParser(description=__doc__) 
    parser.add_argument("--mode", required=True, choices=["comparison", "sentiment", "feature_importance"])
    parser.add_argument("--models", required=True, nargs="+", choices=list(MODEL_FILES))
    args = parser.parse_args()

    if args.mode == "sentiment" and args.models != ["random_forest"]:
        parser.error("Sentiment analysis requires --models random_forest.")

    if args.mode in ["comparison", "sentiment"]:
        output_dir = RESULTS_DIR / "analysis" / args.mode / "_vs_".join(args.models)

    else:
        output_dir = RESULTS_DIR / "analysis" / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    functions = {
        "sentiment": run_sentiment,
        "feature_importance": run_feature_importance,
    }
    functions[args.mode](args.models, output_dir)

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()