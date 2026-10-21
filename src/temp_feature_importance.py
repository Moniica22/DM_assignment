import joblib
import numpy as np
import pandas as pd
import os

from constants import *





def main() -> None:
    model_type = ModelNames.RANDOM_FOREST.value
    model_name = "random_forest"

    feature_importance(model_type, model_name)

    return

if __name__ == "__main__":
    main()