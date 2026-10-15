import pandas as pd

from constants import *


def read_all_data() -> pd.DataFrame:
    df = pd.read_csv(ALL_DATA_FILE)

    return df

def extract_english_reviews(all_reviews) -> pd.DataFrame:

    english_reviews = all_reviews[all_reviews["Review_Language"] == "English"].copy()

    english_reviews.to_csv(ENGLISH_REVIEWS_FILE, index=False)

    print(f"Saved {len(english_reviews)} English reviews. Find them at {ENGLISH_REVIEWS_FILE}")
    return english_reviews


def main():
    # 1 - Read Data
    all_reviews = read_all_data()

    # 2 - Extract English Reviews (and save)
    english_reviews = extract_english_reviews(all_reviews)


    


    return

if __name__ == "__main__":
    main()

