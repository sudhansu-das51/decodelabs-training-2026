# =============================================================================
#  TECH STACK RECOMMENDER — Content-Based AI Recommendation Engine
#  Project 3 | DecodeLabs Industrial Training Kit | Batch 2026
#  Developer  : Subin Das
#  Description: A content-based recommendation engine that maps a user's raw
#               skills to the most relevant career/job roles using TF-IDF
#               vectorization and Cosine Similarity — NOT collaborative
#               filtering, and NOT simple binary overlap matching (both are
#               explicitly ruled out by the project brief).
#
#  Architecture: IPO Model -> Input (User State) -> Process (Similarity Logic)
#                -> Output (Top-N List)
#
#  4-Step Ranking Pipeline (mandatory per spec):
#       1. INGESTION  -> capture user skills (minimum 3 inputs required)
#       2. SCORING    -> Cosine Similarity between user vector & every role
#       3. SORTING    -> rank all roles by descending similarity score
#       4. FILTERING  -> truncate to Top-N (Top 3) to prevent choice overload
# =============================================================================


import sys
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------------------------------------------------
#  CONFIGURATION
# -----------------------------------------------------------------------------

DATASET_PATH      = "raw_skills.csv"
MIN_REQUIRED_SKILLS = 3     # PDF Step 1 mandate: minimum 3 user inputs
TOP_N             = 3        # PDF Step 4 mandate: Top-3 output list


def print_section(title: str) -> None:
    """Prints a clean, professional section divider in the terminal."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# -----------------------------------------------------------------------------
#  STEP 0 — LOAD THE ITEM DATASET (Job Roles = "Items")
# -----------------------------------------------------------------------------

def load_role_dataset(path: str) -> pd.DataFrame:
    """
    Loads raw_skills.csv, where each row is a job role ('item' in our
    recommendation engine) and its associated skill tags. This is the
    item-attribute side of the content-based filtering model.
    """
    print_section("STEP 0: LOADING THE ITEM DATASET (raw_skills.csv)")

    df = pd.read_csv(path)

    # Normalize: lowercase + strip whitespace so vocabulary matches exactly.
    # The PDF explicitly warns that naming mismatches (e.g. "Web Design" vs
    # "Frontend Development") will break the similarity math — so every
    # skill string must be normalized into ONE consistent vocabulary space.
    #
    # IMPORTANT: multi-word skills (e.g. "Machine Learning", "Data Analysis")
    # are joined with underscores into a SINGLE token (machine_learning).
    # Without this, a word-level tokenizer would split "Machine Learning"
    # into the separate tokens "machine" and "learning", silently merging
    # its identity with unrelated terms like "Data Analysis" or "Deep
    # Learning" — which defeats the PDF's explicit "same vocabulary space"
    # requirement for genuinely distinct skills.
    def normalize_skills(raw: str) -> str:
        tags = [tag.strip().lower() for tag in raw.split(",")]
        tags = [tag.replace(" ", "_") for tag in tags if tag]
        return " ".join(tags)

    df["skills_clean"] = df["skills"].apply(normalize_skills)

    print(f"  Total Job Roles (items) : {len(df)}")
    print(f"  Sample roles             : {', '.join(df['role'].head(5).tolist())}")
    print("\n  First 3 rows of dataset:")
    print(df[["role", "skills"]].head(3).to_string(index=False))

    return df


# -----------------------------------------------------------------------------
#  STEP 1 — INGESTION: CAPTURE USER STATE
# -----------------------------------------------------------------------------

def ingest_user_skills(user_skills: list[str]) -> list[str]:
    """
    Validates and normalizes the user's raw skill input.

    Mandatory rule from the PDF: the script MUST accept a minimum of three
    user inputs to ensure sufficient data density for accurate matching.
    Fewer than 3 -> reject and raise, exactly as the spec demands.
    """
    print_section("STEP 1: INGESTION — CAPTURING USER STATE")

    # Defensive cleanup: strip whitespace, drop empty strings
    cleaned = [s.strip() for s in user_skills if s.strip()]

    print(f"  Raw input received   : {user_skills}")
    print(f"  Cleaned input         : {cleaned}")
    print(f"  Input count            : {len(cleaned)}")

    if len(cleaned) < MIN_REQUIRED_SKILLS:
        raise ValueError(
            f"Insufficient input. The PDF spec mandates a MINIMUM of "
            f"{MIN_REQUIRED_SKILLS} skills for sufficient data density. "
            f"You provided only {len(cleaned)}."
        )

    print(f"  ✓ Validation passed — minimum {MIN_REQUIRED_SKILLS} skills satisfied.")
    return cleaned


# -----------------------------------------------------------------------------
#  STEP 2 — VECTOR MAPPING + TF-IDF WEIGHTING (Bridging the Language Barrier)
# -----------------------------------------------------------------------------

def build_tfidf_vector_space(df: pd.DataFrame, user_skills: list[str]):
    """
    Transforms BOTH the item skill-sets and the user's profile into the
    SAME numerical vocabulary space using TF-IDF weighting.

    Why TF-IDF and not simple binary (1/0) overlap?
    The PDF explicitly states binary vectors "lack the nuance required for
    professional-grade applications" — they treat generic, high-frequency
    words (e.g. "git", "python") the same as highly specific, descriptive
    tags (e.g. "kubernetes", "cryptography"). TF-IDF penalizes generic terms
    and rewards specific ones via the Inverse Document Frequency component.

    Crucially: the user's query string is appended into the SAME corpus
    fed to TfidfVectorizer.fit_transform(), guaranteeing items and the user
    profile map into the exact same vocabulary space (the PDF's "Bridging
    the Language Barrier" requirement).
    """
    print_section("STEP 2A: VECTOR MAPPING — TF-IDF WEIGHTING")

    # Build the user's "document" string from their skill list — IDENTICAL
    # normalization rules as the item dataset (lowercase, multi-word skills
    # joined with underscores) so multi-word skills like "Machine Learning"
    # map to the exact same single token on both sides of the vocabulary.
    user_document = " ".join(
        s.strip().lower().replace(" ", "_") for s in user_skills
    )

    # Combine item documents + the user document into ONE corpus so that
    # TfidfVectorizer learns a single shared vocabulary across both sides.
    corpus = df["skills_clean"].tolist() + [user_document]

    vectorizer = TfidfVectorizer(token_pattern=r"[a-zA-Z0-9_\+\#]+")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Last row of the matrix = the user's vector; everything before it =
    # the item (job role) vectors, in the same order as the dataframe.
    item_vectors = tfidf_matrix[:-1]
    user_vector  = tfidf_matrix[-1]

    vocabulary = vectorizer.get_feature_names_out()

    print(f"  Shared vocabulary size  : {len(vocabulary)} unique skill terms")
    print(f"  Item vector matrix shape: {item_vectors.shape}")
    print(f"  User vector shape       : {user_vector.shape}")
    print(f"  User document (tokens)  : '{user_document}'")

    # Show the TF-IDF weight given to each of the user's own input skills,
    # to demonstrate the weighting mechanics described on the "Mathematical
    # Mechanics of TF-IDF" slide — specific/rare skills should score higher
    # than generic/common ones.
    print("\n  TF-IDF weight assigned to each user skill term:")
    user_dense = user_vector.toarray().flatten()
    term_weights = [(vocabulary[i], user_dense[i]) for i in range(len(vocabulary)) if user_dense[i] > 0]
    term_weights.sort(key=lambda x: -x[1])
    for term, weight in term_weights:
        print(f"      - {term:<20}: {weight:.4f}")

    return item_vectors, user_vector, vocabulary


# -----------------------------------------------------------------------------
#  STEP 2B — SCORING: COSINE SIMILARITY (NOT Euclidean Distance)
# -----------------------------------------------------------------------------

def score_roles_by_similarity(df: pd.DataFrame, item_vectors, user_vector) -> pd.DataFrame:
    """
    Calculates the Cosine Similarity between the user's TF-IDF vector and
    every job role's TF-IDF vector.

    Why Cosine Similarity and NOT Euclidean Distance?
    The PDF dedicates an entire slide ("Why Euclidean Distance Fails at
    Scale") to ruling this out: Euclidean distance is sensitive to vector
    MAGNITUDE, so a role description with more total skill tags would
    appear "farther away" even if its skill DIRECTION (i.e. relative
    composition of skills) is identical to the user's profile. Cosine
    similarity measures only the angle between vectors, making it
    invariant to magnitude — the industry-standard choice for this exact
    problem, as stated explicitly in the brief.
    """
    print_section("STEP 2B: SCORING — COSINE SIMILARITY")

    # cosine_similarity returns a (1, n_items) array since user_vector is
    # a single row; we flatten it into a 1-D array of per-role scores.
    similarity_scores = cosine_similarity(user_vector, item_vectors).flatten()

    df_scored = df.copy()
    df_scored["similarity_score"] = similarity_scores

    print("  Cosine similarity calculated for every role in the dataset:")
    print("  (Score range: 0 = no overlap, 1 = perfect alignment)\n")
    for _, row in df_scored[["role", "similarity_score"]].iterrows():
        print(f"      {row['role']:<28}: {row['similarity_score']:.4f}")

    return df_scored


# -----------------------------------------------------------------------------
#  STEP 3 — SORTING: RANK BY DESCENDING SIMILARITY
# -----------------------------------------------------------------------------

def sort_by_relevance(df_scored: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts the scored dataset in descending order of cosine similarity,
    pushing the most relevant roles to the top of the data structure —
    exactly as mandated by Pipeline Step 3 in the brief.
    """
    print_section("STEP 3: SORTING — RANKING BY RELEVANCE")

    df_sorted = df_scored.sort_values(
        by="similarity_score", ascending=False
    ).reset_index(drop=True)

    print("  Full ranked list (highest to lowest similarity):")
    for i, row in df_sorted.iterrows():
        print(f"      #{i + 1:<3} {row['role']:<28}: {row['similarity_score']:.4f}")

    return df_sorted


# -----------------------------------------------------------------------------
#  STEP 4 — FILTERING: TRUNCATE TO TOP-N
# -----------------------------------------------------------------------------

def filter_top_n(df_sorted: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """
    Truncates the sorted list to the Top-N items — preventing "choice
    overload" as explicitly required by Pipeline Step 4 in the brief.
    """
    print_section(f"STEP 4: FILTERING — TOP-{n} OUTPUT")

    df_top = df_sorted.head(n).reset_index(drop=True)

    print(f"  Truncated to top {n} highest-scoring matches.")
    print(f"  ✓ Choice overload avoided — user receives a focused shortlist.")

    return df_top


# -----------------------------------------------------------------------------
#  COLD START SAFETY CHECK (Achilles Heel — addressed per the brief)
# -----------------------------------------------------------------------------

def check_cold_start(df_top: pd.DataFrame) -> bool:
    """
    Detects the 'Cold Start' failure mode described in the PDF: if the
    user's skills share ZERO vocabulary overlap with any role in the
    dataset, every cosine similarity score collapses to 0 — the system
    has nothing meaningful to recommend. We surface this explicitly
    instead of silently returning a meaningless ranked list.
    """
    if df_top["similarity_score"].max() == 0.0:
        print("\n  ⚠ COLD START DETECTED: None of your skills matched any")
        print("    known role vocabulary. Per the PDF's 'Bypassing the Cold")
        print("    Start' guidance, falling back to a general onboarding")
        print("    prompt rather than displaying a meaningless 0% match.")
        return True
    return False


# -----------------------------------------------------------------------------
#  OUTPUT — DISPLAY FINAL RECOMMENDATIONS
# -----------------------------------------------------------------------------

def display_recommendations(df_top: pd.DataFrame, user_skills: list[str]) -> None:
    """
    Presents the final Top-N recommendation list to the user in a clean,
    human-readable format — the 'Output (Top-N List)' stage of the IPO
    architecture.
    """
    print_section("OUTPUT — YOUR TOP TECH STACK RECOMMENDATIONS")

    print(f"  Based on your input skills: {', '.join(user_skills)}\n")

    is_cold_start = check_cold_start(df_top)
    if is_cold_start:
        return

    for i, row in df_top.iterrows():
        match_pct = row["similarity_score"] * 100
        print(f"  #{i + 1}  {row['role']}")
        print(f"       Match Score : {match_pct:.1f}%")
        print(f"       Required Skills : {row['skills']}")
        print()


# -----------------------------------------------------------------------------
#  MAIN PIPELINE — ORCHESTRATES THE FULL 4-STEP RANKING ARCHITECTURE
# -----------------------------------------------------------------------------

def run_recommender(user_skills: list[str]) -> pd.DataFrame:
    """
    Executes the complete content-based recommendation pipeline end-to-end:
    Load dataset -> Ingest user input -> Vectorize (TF-IDF) -> Score (Cosine)
    -> Sort -> Filter (Top-N) -> Display.
    """
    print("\n" + "#" * 70)
    print("#   TECH STACK RECOMMENDER — CONTENT-BASED AI RECOMMENDATION ENGINE")
    print("#   DecodeLabs | Project 3 | Batch 2026")
    print("#" * 70)

    # --- STEP 0: Load items ---
    df = load_role_dataset(DATASET_PATH)

    # --- STEP 1: Ingestion ---
    validated_skills = ingest_user_skills(user_skills)

    # --- STEP 2A: Vector Mapping (TF-IDF) ---
    item_vectors, user_vector, vocabulary = build_tfidf_vector_space(df, validated_skills)

    # --- STEP 2B: Scoring (Cosine Similarity) ---
    df_scored = score_roles_by_similarity(df, item_vectors, user_vector)

    # --- STEP 3: Sorting ---
    df_sorted = sort_by_relevance(df_scored)

    # --- STEP 4: Filtering (Top-N) ---
    df_top = filter_top_n(df_sorted, n=TOP_N)

    # --- Output ---
    display_recommendations(df_top, validated_skills)

    print_section("PIPELINE COMPLETE ✓")
    print(f"  4-step ranking pipeline executed successfully:")
    print(f"    1. Ingestion  -> {len(validated_skills)} skills validated")
    print(f"    2. Scoring    -> {len(df)} roles scored via cosine similarity")
    print(f"    3. Sorting    -> roles ranked by descending relevance")
    print(f"    4. Filtering  -> truncated to Top-{TOP_N} recommendations")
    print()

    return df_top


# -----------------------------------------------------------------------------
#  INTERACTIVE CLI ENTRY POINT
# -----------------------------------------------------------------------------

def main() -> None:
    """
    Interactive command-line entry point. Prompts the user for skills
    (minimum 3, comma-separated) and runs the recommender pipeline.
    """
    print("\nWelcome to the Tech Stack Recommender!")
    print(f"Enter at least {MIN_REQUIRED_SKILLS} skills, separated by commas.")
    print("Example: Python, Cloud, Automation\n")

    raw = input("Your skills: ").strip()

    if not raw:
        print("\nNo input detected. Running a demo with example skills instead...")
        demo_skills = ["Python", "Cloud", "Automation"]
        run_recommender(demo_skills)
        return

    user_skills = [s.strip() for s in raw.split(",")]

    try:
        run_recommender(user_skills)
    except ValueError as e:
        print(f"\n  ✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
