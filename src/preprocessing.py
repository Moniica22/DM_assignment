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
    df = df.copy()

    # removes any '   ', '\t', '\n' etc
    for column in ["Upside_Review", "Downside_Review"]:
        df[column] = df[column].str.replace(r"\s+", " ", regex=True).str.strip()

    return df


def concatenate_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Review_Text"] = (df["Upside_Review"] + " " + df["Downside_Review"]).str.strip()

    return df


def remove_empty_reviews(df: pd.DataFrame) -> pd.DataFrame:
    orig_size = df.shape[0]

    # Empty reviews are those that after concatenation are only an empty space
    df = df[df["Review_Text"].str.strip() != ""].copy()

    print(f"Removed {orig_size-df.shape[0]} empty reviews")

    return df


def handle_duplicate_reviews(df: pd.DataFrame) -> pd.DataFrame:
    orig_size_before_identicals = df.shape[0]
    # Remove Duplicates
    df = df.drop_duplicates().copy()

    # Remove rows with identical x-y
    orig_size_before_same_xy = df.shape[0]
    # Keep a standard row each time for reproducibility
    df = df.sample(frac=1, random_state = RANDOM_STATE).reset_index(drop=True)
    df = df.drop_duplicates(['Review_Text', 'source']).copy()

    print(f"Removed {orig_size_before_identicals - orig_size_before_same_xy} fully identical rows.")
    print(f"Removed {orig_size_before_same_xy - df.shape[0]} rows with same text and source.")

    return df



preprocessing_steps = [
    ("remove_invalid_source_labels", remove_invalid_source_labels, True),
    ("fill_missing_review_text", fill_missing_review_text, True),
    ("clean_problematic_characters", clean_problematic_characters, False),
    ("normalize_whitespace", normalize_whitespace, True),
    ("concatenate_reviews", concatenate_reviews, True),
    ("remove_empty_reviews", remove_empty_reviews, True),
    ("check_duplicate_reviews", handle_duplicate_reviews, True),
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

