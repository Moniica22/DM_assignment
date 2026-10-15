from pathlib import Path
from enum import Enum


class ModelNames(str, Enum):
    NAIVE_BAYES = "naive_bayes"
    LOGISTIC_REGRESSION_L1 = "logistic_regression_l1"
    CLASSIFICATION_TREE = "classification_tree"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"

MODELS_DIR = PROJECT_ROOT / "models"
NAIVE_BAYES_DIR = MODELS_DIR / "naive_bayes"
LOGISTIC_REGRESSION_DIR = MODELS_DIR / "logistic_regression_l1"
RANDOM_FOREST_DIR = MODELS_DIR / "random_forest"
GRADIENT_BOOSTING_DIR = MODELS_DIR / "gradient_boosting"
CLASSIFICATION_TREE_DIR = MODELS_DIR / "classification_tree"

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"