import pandas as pd

from constants import *

# Read Data end extract english reviews
def read_all_data() -> pd.DataFrame:
    df = pd.read_csv(ALL_DATA_FILE)

    return df

def extract_english_reviews(all_reviews) -> pd.DataFrame:

    english_reviews = all_reviews[all_reviews["Review_Language"] == "English"].copy()

    english_reviews.to_csv(ENGLISH_REVIEWS_FILE, index=False)

    print(f"Saved {len(english_reviews)} English reviews. Find them at {ENGLISH_REVIEWS_FILE}")
    return english_reviews

# Preprocessing Functions

def remove_invalid_source_labels(df: pd.DataFrame) -> pd.DataFrame:
    valid = df["source"].isin([0, 1])

    print(f"Removed {len(df) - (valid).sum()} rows with missing/invalid source labels.")

    return df[valid].copy()

def fill_missing_review_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    columns = ["Upside_Review", "Downside_Review"]

    missing_per_col = df[columns].isna().sum(axis=0)
    missing_count = missing_per_col[0] + missing_per_col[1]
    df[columns] = df[columns].fillna("")

    print(f"Missing review texts:\n{missing_per_col}")

    print(f"\nReplaced {missing_count} missing text sections with empty strings.")

    return df

def clean_problematic_characters(df: pd.DataFrame) -> pd.DataFrame:
    return df

def normalize_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    return df

def concatenate_reviews(df: pd.DataFrame) -> pd.DataFrame:
    return df

def remove_empty_reviews(df: pd.DataFrame) -> pd.DataFrame:
    return df

def check_duplicate_reviews(df: pd.DataFrame) -> pd.DataFrame:
    return df

def check_sentiment_labels(df: pd.DataFrame) -> pd.DataFrame:
    return df

preprocessing_steps = [
    ("remove_invalid_source_labels", remove_invalid_source_labels, True),
    ("fill_missing_review_text", fill_missing_review_text, True),
    ("clean_problematic_characters", clean_problematic_characters, False),
    ("normalize_whitespace", normalize_whitespace, True),
    ("concatenate_reviews", concatenate_reviews, True),
    ("remove_empty_reviews", remove_empty_reviews, True),
    ("check_duplicate_reviews", check_duplicate_reviews, True),
    ("check_sentiment_labels", check_sentiment_labels, False),
]

def main():
    # 1 - Read Data
    all_reviews = read_all_data()

    # 2 - Extract English Reviews (and save)
    df = extract_english_reviews(all_reviews)

    # 3 - Apply preprocessing steps
    for (step_name,step,enabled) in preprocessing_steps:
        if enabled:
            print(f"\n - Running {step_name}")
            df = step(df)

    # 4 - Stratified Split


    return

if __name__ == "__main__":
    main()

