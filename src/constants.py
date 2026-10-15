from pathlib import Path
from enum import Enum

# Model Names
class ModelNames(str, Enum):
    NAIVE_BAYES = "naive_bayes"
    LOGISTIC_REGRESSION_L1 = "logistic_regression_l1"
    CLASSIFICATION_TREE = "classification_tree"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Src
SRC_DIR = PROJECT_ROOT / "src"

# Models
MODELS_DIR = PROJECT_ROOT / "models"
NAIVE_BAYES_DIR = MODELS_DIR / "naive_bayes"
LOGISTIC_REGRESSION_DIR = MODELS_DIR / "logistic_regression_l1"
RANDOM_FOREST_DIR = MODELS_DIR / "random_forest"
GRADIENT_BOOSTING_DIR = MODELS_DIR / "gradient_boosting"
CLASSIFICATION_TREE_DIR = MODELS_DIR / "classification_tree"

# Data
DATA_DIR = PROJECT_ROOT / "data"
ALL_DATA_FILE = DATA_DIR / "all_data.csv"
ENGLISH_REVIEWS_FILE = DATA_DIR / "english_reviews.csv"
TRAIN_DATA_FILE = DATA_DIR / "train_data.csv"
TEST_DATA_FILE = DATA_DIR / "test_data.csv"

# Results
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_STATE = 42