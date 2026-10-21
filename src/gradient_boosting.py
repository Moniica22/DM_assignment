import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import joblib
import json

from constants import *



model_name = ModelNames.GRADIENT_BOOSTING.value
model_path = GRADIENT_BOOSTING_DIR

def save_training_results(search: GridSearchCV, model_dir: Path, model_name: str) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{model_name}_2.joblib"
    joblib.dump(search.best_estimator_, model_path)

    with open(model_dir / f"{model_name}_best_params_2.json", "w") as file:
        json.dump(
            {
                "best_params": search.best_params_,
                "best_cv_score": search.best_score_,
                "param_grid": search.param_grid,
            },
            file,
            indent=2
        )

    pd.DataFrame(search.cv_results_).to_csv(
        model_dir / f"{model_name}_cv_results_2.csv",
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
    pipeline = Pipeline([
        ("bow", CountVectorizer()),
        ("classifier", GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ])
    return pipeline


def create_cv_folds() -> StratifiedKFold:
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    return cv


def create_param_search_grid(pipeline:Pipeline, cv:StratifiedKFold, param_grid:dict) -> GridSearchCV:
    search_grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        refit=True,      
        verbose=2,
    )

    return search_grid


def run_cv_search_and_train_model(X_train, Y_train, search:GridSearchCV) -> GridSearchCV:
    search.fit(X_train, Y_train)
    print("Best parameters:", search.best_params_)
    print("Best CV accuracy:", search.best_score_)

    return search


def main():
    # 1 - Load Data
    X_train, Y_train = load_data()

    # 2 - Build model
    pipeline = build_model()

    # 3 - CV Folds object
    cv = create_cv_folds()

    # 4 - Create Search Object
    param_grid = { # 81 candidates
        "bow__min_df": [1, 2, 5],
        "classifier__n_estimators": [100, 300, 500],
        "classifier__learning_rate": [0.05, 0.1, 0.3],
        "classifier__max_depth": [1, 3, 5] # lower than in random forest because it accumulates corrections
    }
    search = create_param_search_grid(pipeline, cv, param_grid)

    # 5 - Run param search
    search = run_cv_search_and_train_model(X_train, Y_train, search)

    # 6 - Save pipeline
    save_training_results(search, model_path, model_name)

    return

if __name__ == "__main__":
    main()