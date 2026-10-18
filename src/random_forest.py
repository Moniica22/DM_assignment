import pandas as pd
import numpy as np
import joblib
import json
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid

from constants import *


model_name = ModelNames.RANDOM_FOREST.value
model_path = RANDOM_FOREST_DIR

def save_training_results(pipeline: Pipeline, best_params:dict, results: pd.DataFrame, param_grid: dict,model_dir: Path, model_name: str) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{model_name}.joblib"
    joblib.dump(pipeline, model_path)

    forest = pipeline.named_steps["classifier"] 
    with open(model_dir / f"{model_name}_best_params.json", "w") as file:
        json.dump(
            {
                "selection_method": "out-of-bag",
                "selection_metric": "accuracy",
                "best_params": best_params,
                "best_oob_score": forest.oob_score_,
                "param_grid": param_grid,
                "fixed_forest_params":{
                    "n_estimators": forest.n_estimators,
                    "bootstrap": forest.bootstrap,
                    "oob_score": forest.oob_score,
                    "random_state": forest.random_state,
                },
            },
            file,
            indent=2
        )

    results.to_csv(
        model_dir / f"{model_name}_oob_results.csv",
        index=False
    )

    print(f"Saved pipeline: {model_path}")
    return model_path


def load_data() -> tuple[pd.Series, pd.Series]:
    train_df = pd.read_csv(TRAIN_DATA_FILE)
    X_train = train_df[X_COL]
    Y_train = train_df[Y_COL]

    return X_train, Y_train


def build_model() -> Pipeline:
    return Pipeline([
        ("bow", CountVectorizer()), 
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            bootstrap=True, 
            oob_score=True, # calculate OOB score during fitting
            random_state=RANDOM_STATE,
            n_jobs=-1, # allows to use all available CPUs for training
        )),

    ])


def run_oob_search_and_train_model(X_train, Y_train, param_grid:dict) -> tuple[Pipeline, dict, pd.DataFrame]:
    configurations = list(ParameterGrid(param_grid))
    best_pipeline = None
    best_params = None
    best_score = -np.inf
    results = []

    for i, params in enumerate(configurations, start=1):
        pipeline = build_model()
        pipeline.set_params(**params)
        pipeline.fit(X_train, Y_train)

        forest = pipeline.named_steps["classifier"] # step of the pipeline that contains the trained Random Forest model
        score = forest.oob_score_ # its OOB score
        results.append({
            **params, # unpack the parameters of the current configuration
            "oob_score":score
        })
        print(f"Configuration {i}/{len(configurations)}: {params}, OOB score: {score:.4f}")

        if score>best_score:
            best_score = score
            best_pipeline = pipeline
            best_params = params

    
    print("Best parameters:", best_params)
    print("Best OOB score:", best_score)

    return best_pipeline, best_params, pd.DataFrame(results)


def main():
    # 1 - Load Data
    X_train, Y_train = load_data()

    # 2 - Create hyperparameter search grid
    param_grid = { # 36 candidate forests
        "bow__min_df": [1, 2, 5],
        "classifier__max_depth": [10, 30, None],
        "classifier__min_samples_leaf": [1, 3],
        "classifier__max_features": ["sqrt", 0.1],
    }
    # 3 - Create and run each candidate; select based on OOB score
    best_pipeline, best_params, results = run_oob_search_and_train_model(X_train, Y_train, param_grid)

    # 4 - Save pipeline
    save_training_results(best_pipeline, best_params, results, param_grid, model_path, model_name)

    return

if __name__ == "__main__":
    main()