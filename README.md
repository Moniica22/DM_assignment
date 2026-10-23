# DM_assignment

This repository contains the code for a data mining assignment on classifying review text by its `source` label. The project preprocesses the review data, trains and evaluates several machine-learning classifiers, and generates analysis results for the report.

## Project structure

- `data/` – input data and generated train/test datasets.
- `src/` – preprocessing, model-training, testing, and results-analysis scripts.
- `models/` – saved trained pipelines and model-selection results.
- `results/` – predictions, evaluation metrics, confusion matrices, and statistical-analysis outputs.
- `requirements.txt` – required Python packages.

## Setup

Python 3.9 or newer is recommended. From the repository root, create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## How to run the code

Run all commands from the repository root so that the generated files are written to the expected `data/`, `models/`, and `results/` directories.

### 1. Preprocess the data

The preprocessing script reads `data/all_data.csv`, keeps English reviews, cleans and combines the review-text fields, removes invalid or empty records and duplicates, and creates a stratified train/test split:

```bash
python src/preprocessing.py
```

This generates:

- `data/english_reviews.csv`
- `data/train_data.csv`
- `data/test_data.csv`

### 2. Train the models

Run the model scripts individually:

```bash
python src/naive_bayes.py
python src/logistic_regression_l1.py
python src/classification_tree.py
python src/random_forest.py
python src/gradient_boosting.py
```

Each script trains a bag-of-words classifier, performs its configured hyperparameter search (or out-of-bag search for the random forest), and saves the trained pipeline and model-selection results under the corresponding directory in `models/`.

### 3. Evaluate a trained model

`src/testing.py` loads a saved model, predicts labels for `data/test_data.csv`, prints a classification report and confusion matrix, and saves the evaluation outputs under `results/`.

The default `main()` configuration evaluates the L1-regularized logistic-regression model:

```bash
python src/testing.py
```

To evaluate a different model, update the `model_type` and `model_name` values in `src/testing.py` before running it. The available model names are:

- `naive_bayes`
- `logistic_regression_l1`
- `classification_tree`
- `random_forest`
- `gradient_boosting`

### 4. Generate report analyses

`src/results_analysis.py` generates the comparative analyses used in the report. It requires trained model files in `models/` and accepts the following command-line modes:

#### Compare two models with McNemar's exact test

```bash
python src/results_analysis.py --mode comparison --models random_forest gradient_boosting
```

#### Compare random-forest performance by sentiment

```bash
python src/results_analysis.py --mode sentiment --models random_forest
```

#### Extract the most important features

```bash
python src/results_analysis.py --mode feature_importance --models naive_bayes logistic_regression classification_tree random_forest gradient_boosting
```

The analysis files are saved under `results/analysis/`.

## Scripts that produce the key report results

- **Data preparation and train/test split:** `src/preprocessing.py` produces the cleaned English-review data and the stratified training and test datasets.
- **Naive Bayes model and cross-validation results:** `src/naive_bayes.py`.
- **L1 logistic-regression model and cross-validation results:** `src/logistic_regression_l1.py`.
- **Classification-tree model and cross-validation results:** `src/classification_tree.py`.
- **Random-forest model and out-of-bag results:** `src/random_forest.py`.
- **Gradient-boosting model and cross-validation results:** `src/gradient_boosting.py`.
- **Test-set predictions, classification reports, and confusion matrices:** `src/testing.py`.
- **Model comparison, sentiment-specific performance, statistical tests, and feature-importance tables:** `src/results_analysis.py`.

The model-training scripts save their main outputs in `models/<model_name>/`. Evaluation outputs are saved in `results/<model_name>/`, while the comparative analyses are saved in `results/analysis/`.

## External libraries

Install the required packages with `pip install -r requirements.txt`:

- `joblib` – saving and loading trained model pipelines.
- `pandas` – reading, processing, and writing tabular data.
- `matplotlib` – plotting support.
- `seaborn` – statistical visualisation support.
- `scikit-learn` – preprocessing, vectorisation, model training, hyperparameter search, and evaluation metrics.
- `statsmodels` – McNemar's exact test used in model comparison.

The analysis script also imports `scipy.stats.chi2_contingency`; install SciPy if it is not already installed as a dependency of the environment:

```bash
pip install scipy
```
