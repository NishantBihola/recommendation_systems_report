# BN XP 12957 — Goodreads Recommender Systems
# Popularity-based (IMDb-style) + Content-based (TF-IDF on authors)

import argparse
import os
import sys
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# 0) Robust dataset loader
# -----------------------------
def load_books(csv_path: str) -> pd.DataFrame:
    """
    Load Goodreads CSV robustly (handles malformed lines, encodings, and weird quotes/commas).
    Normalizes column names and enforces basic schema required by the recommenders.
    """
    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    try:
        # Fast path
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError:
        print("[WARN] ParserError with default engine. Retrying with python engine and robust options...")
        try:
            df = pd.read_csv(
                csv_path,
                engine="python",
                sep=",",
                quotechar='"',
                escapechar="\\",
                on_bad_lines="skip",   # skip malformed lines
            )
        except UnicodeDecodeError:
            df = pd.read_csv(
                csv_path,
                engine="python",
                sep=",",
                quotechar='"',
                escapechar="\\",
                on_bad_lines="skip",
                encoding="latin1",
            )

    # Normalize columns to snake_case
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Common renames/normalization
    renames = {
        "isbn13": "isbn_13",
        "  num_pages": "num_pages",  # some files have leading spaces
    }
    for old, new in renames.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # Expected columns (we’ll keep whatever exists among these)
    expected = [
        "bookid", "title", "authors", "average_rating", "isbn", "isbn_13",
        "language_code", "num_pages", "ratings_count", "text_reviews_count"
    ]
    present = [c for c in expected if c in df.columns]
    if not present:
        print("[ERROR] None of the expected columns found. Check your CSV.", file=sys.stderr)
        sys.exit(1)

    # Limit to expected columns that exist
    df = df[present].copy()

    # Ensure types used by recommenders
    if "average_rating" in df.columns:
        df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce")
    if "ratings_count" in df.columns:
        df["ratings_count"] = pd.to_numeric(df["ratings_count"], errors="coerce").fillna(0).astype(int)

    # Drop rows missing critical fields
    critical = [c for c in ["title", "authors", "average_rating", "ratings_count"] if c in df.columns]
    df = df.dropna(subset=critical)

    # Remove duplicate titles (keep the one with most ratings)
    if all(c in df.columns for c in ["title", "ratings_count"]):
        df = (
            df.sort_values(["title", "ratings_count"], ascending=[True, False])
              .drop_duplicates(subset=["title"], keep="first")
              .reset_index(drop=True)
        )

    print(f"[INFO] Loaded {len(df):,} rows after cleaning.")
    return df


# -----------------------------
# 1) Popularity-based recommender
# IMDb weighted rating:
# score = (v/(v+m))*R + (m/(v+m))*C
#   R = book’s average_rating
#   v = book’s ratings_count
#   C = mean of average_rating across all books
#   m = ratings_count threshold (e.g., 90th percentile)
# -----------------------------
def compute_popularity_scores(df: pd.DataFrame, min_percentile: float = 0.90) -> pd.DataFrame:
    if "ratings_count" not in df.columns or "average_rating" not in df.columns:
        raise ValueError("Dataframe missing required columns for popularity scoring.")

    C = df["average_rating"].mean()
    m = float(np.percentile(df["ratings_count"], min_percentile * 100))

    def weighted_score(row):
        v = float(row["ratings_count"])
        R = float(row["average_rating"])
        denom = v + m
        return ((v / denom) * R + (m / denom) * C) if denom > 0 else 0.0

    # Use IMDb-style cut: only items above threshold
    qualified = df[df["ratings_count"] >= m].copy()
    if qualified.empty:
        # fallback if dataset is small
        qualified = df.copy()

    qualified["popularity_score"] = qualified.apply(weighted_score, axis=1)
    ranked = qualified.sort_values("popularity_score", ascending=False).reset_index(drop=True)
    return ranked


def recommend_popularity(df: pd.DataFrame, top_n: int = 10, min_percentile: float = 0.90) -> pd.DataFrame:
    ranked = compute_popularity_scores(df, min_percentile=min_percentile)
    cols = [c for c in ["title", "authors", "average_rating", "ratings_count", "popularity_score"] if c in ranked.columns]
    return ranked.head(top_n)[cols]


# -----------------------------
# 2) Content-based recommender (authors)
# TF-IDF on 'authors' -> cosine similarity
# -----------------------------
def build_author_tfidf(df: pd.DataFrame):
    if "authors" not in df.columns:
        raise ValueError("'authors' column is required for content-based recommender.")

    corpus = df["authors"].fillna("").astype(str).values
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        lowercase=True,
        stop_words=None
    )
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X


def build_title_index(df: pd.DataFrame):
    if "title" not in df.columns:
        raise ValueError("'title' column is required to index books by title.")
    return {t.lower().strip(): i for i, t in enumerate(df["title"].astype(str).values)}


def recommend_similar_by_title(df: pd.DataFrame, tfidf_matrix, title_index_map: dict,
                               query_title: str, top_n: int = 10) -> pd.DataFrame:
    q = query_title.lower().strip()
    if q not in title_index_map:
        # Try fuzzy-ish match (startswith) to help users
        candidates = [t for t in title_index_map.keys() if t.startswith(q)]
        if candidates:
            q = candidates[0]
        else:
            raise ValueError(f"Title not found in dataset: '{query_title}'")

    idx = title_index_map[q]
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).ravel()
    order = np.argsort(-sims)
    order = [i for i in order if i != idx][:top_n]

    cols = [c for c in ["title", "authors", "average_rating", "ratings_count"] if c in df.columns]
    out = df.iloc[order][cols].copy()
    out["similarity"] = sims[order]
    return out.reset_index(drop=True)


# -----------------------------
# 3) CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Goodreads Recommender Systems (Popularity + Content)")
    ap.add_argument("--csv", default="data/books.csv", help="Path to Goodreads CSV (default: data/books.csv)")
    ap.add_argument("--topn", type=int, default=10, help="Number of recommendations to show (default: 10)")
    ap.add_argument("--minp", type=float, default=0.90, help="Popularity min percentile for ratings_count (default: 0.90)")
    ap.add_argument("--title", type=str, default=None, help="Book title for content-based recommendations")
    return ap.parse_args()


def main():
    args = parse_args()
    df = load_books(args.csv)

    # Popularity-based recommendations
    print("\n=== Popularity-based Recommendations (IMDb-weighted) ===")
    pop = recommend_popularity(df, top_n=args.topn, min_percentile=args.minp)
    if pop.empty:
        print("[WARN] No popularity recommendations produced.")
    else:
        print(pop.to_string(index=False))

    # Content-based recommendations (authors)
    print("\n=== Content-based Recommendations (TF-IDF on authors) ===")
    vectorizer, X = build_author_tfidf(df)
    title_map = build_title_index(df)

    # Seed title
    if args.title:
        seed_title = args.title
    else:
        seed_title = pop.iloc[0]["title"] if not pop.empty else df.iloc[0]["title"]
        print(f"[INFO] No --title provided. Using seed: {seed_title!r}")

    try:
        similar = recommend_similar_by_title(df, X, title_map, seed_title, top_n=args.topn)
        print(f"\nSimilar to: {seed_title}\n")
        print(similar.to_string(index=False))
    except ValueError as e:
        print(f"[WARN] {e}")
        # Fallback: show a few items so notebook still has output
        fallback_cols = [c for c in ["title", "authors", "average_rating", "ratings_count"] if c in df.columns]
        print(df[fallback_cols].head(args.topn).to_string(index=False))


if __name__ == "__main__":
    main()
