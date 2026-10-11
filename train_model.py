"""
train_model.py
--------------
Trains a TF-IDF + Logistic Regression pipeline to classify news articles as
FAKE or REAL, evaluates it on a held-out test set, and saves:
  - models/news_classifier.joblib  (the trained pipeline)
  - models/metrics.json            (evaluation numbers for the Streamlit app)

Usage:
    python train_model.py

Dataset expected in the data/ folder (see README.md for download instructions):
    Option A: data/Fake.csv  +  data/True.csv   (Kaggle ISOT dataset)
    Option B: data/news.csv  with columns 'text' and 'label'
"""

import io
import json
import os
import re
import sys

# Ensure UTF-8 output on Windows consoles to prevent UnicodeEncodeError
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "data"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "news_classifier.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# Columns the Kaggle ISOT dataset uses for article text
EXPECTED_TEXT_COLS = ["title", "text"]


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
def clean_text(raw: str) -> str:
    """
    Clean an article string for TF-IDF feature extraction.

    Steps:
      1. Strip news agency datelines (e.g. "WASHINGTON (Reuters) -") and
         publisher signatures to prevent shortcut learning and dataset bias.
      2. Convert to lower-case.
      3. Remove URLs (http/https/www).
      4. Strip HTML / XML tags.
      5. Remove publisher metadata shortcuts ('reuters', 'image via', 'getty images').
      6. Remove bare numbers (page numbers, years that add noise).
      7. Drop any character that is not a letter or whitespace.
      8. Collapse runs of whitespace and strip leading/trailing spaces.
    """
    text = str(raw)
    # Strip dateline prefix (e.g. "WASHINGTON (Reuters) - ", "(Reuters) - ")
    text = re.sub(r"^[a-zA-Z\s,./-]+?\(reuters\)\s*[-:]\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\([a-zA-Z\s,./-]*reuters\)\s*[-:]\s*", " ", text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # remove URLs
    text = re.sub(r"<[^>]+>", " ", text)                  # strip HTML tags
    text = re.sub(r"\breuters\b", " ", text)              # remove publisher leakage shortcut
    text = re.sub(r"\b(image via|featured image|getty images)\b", " ", text) # remove photo watermark tags
    text = re.sub(r"\b\d+\b", " ", text)                  # remove bare numbers
    text = re.sub(r"[^a-z\s]", " ", text)                 # keep only letters
    text = re.sub(r"\s+", " ", text).strip()              # normalise whitespace
    return text


# ---------------------------------------------------------------------------
# STEP 1 – Dataset loading and validation
# ---------------------------------------------------------------------------
def load_dataset() -> pd.DataFrame:
    """
    Load the news dataset from disk.

    Supports two layouts:
      - Dual-file (preferred): data/Fake.csv  +  data/True.csv
      - Single-file fallback:  data/news.csv  (must have 'text' and 'label' cols)

    Returns a DataFrame with exactly two columns: 'content' and 'label'.
    'label' values are always either 'FAKE' or 'REAL' (upper-case strings).
    """
    def _find_file(filename: str):
        p1 = os.path.join(DATA_DIR, filename)
        p2 = os.path.join(DATA_DIR, filename + ".zip")
        if os.path.exists(p1):
            return p1
        if os.path.exists(p2):
            return p2
        return None

    fake_file = _find_file("Fake.csv")
    true_file = _find_file("True.csv")
    single_file = _find_file("news.csv")

    # ── Detect which layout is available ──────────────────────────────────
    if fake_file and true_file:
        print(f"  [INFO] Found {fake_file} and {true_file} – loading both.")
        fake_df = pd.read_csv(fake_file)
        true_df = pd.read_csv(true_file)

        if fake_df.empty:
            raise ValueError(f"{fake_file} loaded but is empty.")
        if true_df.empty:
            raise ValueError(f"{true_file} loaded but is empty.")

        print(f"         {os.path.basename(fake_file)}  →  {len(fake_df):,} rows  |  columns: {list(fake_df.columns)}")
        print(f"         {os.path.basename(true_file)}  →  {len(true_df):,} rows  |  columns: {list(true_df.columns)}")

        # Assign labels BEFORE merging to avoid any ambiguity
        fake_df["label"] = "FAKE"
        true_df["label"] = "REAL"

        data = pd.concat([fake_df, true_df], ignore_index=True)

    elif single_file:
        print("  [INFO] data/Fake.csv and data/True.csv not found.")
        print(f"  [INFO] Falling back to {single_file} …")
        data = pd.read_csv(single_file)

        if data.empty:
            raise ValueError(f"{single_file} is empty.")

        missing_cols = {"text", "label"} - set(data.columns)
        if missing_cols:
            raise ValueError(
                f"\n  {single_file} is missing required column(s): {missing_cols}\n"
                "  The file must contain at least:\n"
                "    - 'text'  column with the article body\n"
                "    - 'label' column with values 'FAKE' or 'REAL'"
            )
        print(f"         {os.path.basename(single_file)}  →  {len(data):,} rows  |  columns: {list(data.columns)}")

    elif fake_file and not true_file:
        raise FileNotFoundError(
            f"\n"
            f"  ── Incomplete dataset ─────────────────────────────────────\n"
            f"\n"
            f"  Found {fake_file}, but data/True.csv is missing!\n"
            f"  Both Fake.csv and True.csv must be present to train the model.\n"
            f"\n"
            f"  Please download True.csv from Kaggle and place it in data/:\n"
            f"  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset\n"
            f"  ───────────────────────────────────────────────────────────\n"
        )

    elif true_file and not fake_file:
        raise FileNotFoundError(
            f"\n"
            f"  ── Incomplete dataset ─────────────────────────────────────\n"
            f"\n"
            f"  Found {true_file}, but data/Fake.csv is missing!\n"
            f"  Both Fake.csv and True.csv must be present to train the model.\n"
            f"\n"
            f"  Please download Fake.csv from Kaggle and place it in data/:\n"
            f"  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset\n"
            f"  ───────────────────────────────────────────────────────────\n"
        )

    else:
        raise FileNotFoundError(
            "\n"
            "  ── Dataset not found ──────────────────────────────────────\n"
            "\n"
            "  No dataset files were found inside the  data/  folder.\n"
            "  Please download one of the following and place the file(s) there:\n"
            "\n"
            "  Option A (recommended):\n"
            "    Download data/Fake.csv + data/True.csv from Kaggle:\n"
            "    https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset\n"
            "\n"
            "  Option B:\n"
            "    Provide data/news.csv with columns 'text' and 'label'.\n"
            "    Label values must be exactly 'FAKE' or 'REAL' (case-insensitive).\n"
            "  ───────────────────────────────────────────────────────────\n"
        )

    # ── Build a single 'content' column from all available text columns ───
    text_cols_present = [c for c in EXPECTED_TEXT_COLS if c in data.columns]

    if not text_cols_present:
        raise ValueError(
            f"\n  No usable text column found in the dataset.\n"
            f"  Expected at least one of: {EXPECTED_TEXT_COLS}\n"
            f"  Columns actually present: {list(data.columns)}"
        )

    print(f"  [INFO] Combining text columns: {text_cols_present}")

    # Fill NaN in text columns with "" before joining so that a missing title
    # does not produce the literal string 'nan' inside the article content.
    data["content"] = (
        data[text_cols_present]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )

    # ── Normalise label column ────────────────────────────────────────────
    data["label"] = data["label"].astype(str).str.upper().str.strip()

    # ── Clean the combined article text ───────────────────────────────────
    print("  [INFO] Cleaning text (lower-case, remove URLs, HTML, punctuation) …")
    data["content"] = data["content"].map(clean_text)

    # ── Remove unusable rows ──────────────────────────────────────────────
    rows_before = len(data)

    # Articles that are empty (or effectively empty) after cleaning are useless
    data = data[data["content"].str.len() > 10]
    # Remove any rows whose label is not FAKE or REAL
    data = data[data["label"].isin(["FAKE", "REAL"])]
    # Remove exact duplicate articles to avoid inflated test scores
    data = data.drop_duplicates(subset=["content"])

    rows_dropped = rows_before - len(data)
    if rows_dropped > 0:
        print(f"  [INFO] Removed {rows_dropped:,} rows "
              f"(empty content / invalid label / exact duplicates).")

    return data[["content", "label"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# STEP 2 – Dataset summary
# ---------------------------------------------------------------------------
def print_dataset_summary(data: pd.DataFrame) -> None:
    """Print shape, class distribution, and content length statistics."""
    print()
    print("=" * 62)
    print("  DATASET SUMMARY")
    print("=" * 62)
    print(f"  Total articles after cleaning : {len(data):,}")
    print()

    counts = data["label"].value_counts()
    print("  Class distribution:")
    for label, count in counts.items():
        pct = count / len(data) * 100
        bar = "█" * int(pct / 2)
        print(f"    {label:<6}  {count:>6,} articles  ({pct:.1f}%)  {bar}")

    print()
    char_len = data["content"].str.len()
    print("  Article character length (after cleaning):")
    print(f"    Minimum : {char_len.min():,}")
    print(f"    Median  : {int(char_len.median()):,}")
    print(f"    Maximum : {char_len.max():,}")
    print(f"    Mean    : {char_len.mean():.0f}")
    print("=" * 62)
    print()


# ---------------------------------------------------------------------------
# STEP 3 – Train / test split  (no data leakage)
# ---------------------------------------------------------------------------
def split_data(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split into training and test sets.

    • test_size=0.2  → 80 % train / 20 % test.
    • stratify       → both classes keep the same ratio in both splits.
    • random_state   → makes the split reproducible across runs.

    Data leakage is avoided because:
      - The split happens BEFORE TF-IDF is fitted.
      - The Pipeline fits TF-IDF only on X_train and applies that
        vocabulary when transforming X_test.  The test set never
        influences what words the vectorizer learns.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data["content"],
        data["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=data["label"],
    )

    print(f"  Split: 80% train / 20% test | stratified | random_state={random_state}")
    print(f"    Training set : {len(X_train):,} articles")
    print(f"    Test set     : {len(X_test):,} articles")
    print()
    print("  Training class distribution:")
    for label, count in y_train.value_counts().items():
        print(f"    {label:<6}  {count:,}  ({count / len(y_train) * 100:.1f}%)")
    print()
    print("  Test class distribution:")
    for label, count in y_test.value_counts().items():
        print(f"    {label:<6}  {count:,}  ({count / len(y_test) * 100:.1f}%)")
    print()

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# STEP 4 – Build, train and evaluate
# ---------------------------------------------------------------------------
def save_confusion_matrix_plot(cm: list, labels: list, output_path: str) -> None:
    """
    Save a colour-coded confusion matrix heatmap as a PNG image.

    The heatmap makes it easy to spot at a glance how many articles are
    correctly classified vs. confused between the two classes.

    Parameters
    ----------
    cm          : 2-D list produced by sklearn confusion_matrix().tolist()
    labels      : list of class names in the same order as cm rows/columns
    output_path : file path where the PNG will be written
    """
    cm_array = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))

    # Blues colour map – darker = more samples in that cell
    im = ax.imshow(cm_array, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    # Tick labels on both axes
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)

    # Print the count inside each cell; use white text on dark cells for contrast
    thresh = cm_array.max() / 2.0
    for row in range(len(labels)):
        for col in range(len(labels)):
            color = "white" if cm_array[row, col] > thresh else "black"
            ax.text(col, row, format(cm_array[row, col], ",d"),
                    ha="center", va="center", color=color, fontsize=12)

    ax.set_ylabel("Actual label", fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)   # free memory – do not display the plot in training mode


def build_and_train(X_train, X_test, y_train, y_test, model_dir: str = MODEL_DIR):
    """
    Build a scikit-learn Pipeline (TF-IDF → Logistic Regression), train it
    on X_train / y_train, evaluate strictly on X_test / y_test, and return
    the fitted pipeline together with a dictionary of evaluation metrics.

    ── Class-label assumption ──────────────────────────────────────────────
    This function expects exactly two string labels: 'FAKE' and 'REAL'.
    They are assigned before the data is loaded (Fake.csv → 'FAKE',
    True.csv → 'REAL') so there is no guesswork about which file maps to
    which class.  If you use a different dataset, make sure label values are
    normalised to these two strings (load_dataset() does this automatically).

    ── Why a Pipeline? ─────────────────────────────────────────────────────
    scikit-learn's Pipeline chains the vectorizer and classifier into a single
    object.  When you call pipeline.fit(X_train, y_train):
      • TfidfVectorizer.fit_transform() is called on X_train only.
      • The learned vocabulary is then used to transform X_test via
        pipeline.predict() – X_test never influences the vocabulary.
    This single object is what gets saved to disk, so the Streamlit app loads
    the exact same vectorizer (with the exact same vocabulary) that was used
    during training.

    ── Hyperparameter rationale ────────────────────────────────────────────
    TfidfVectorizer:
      max_features=50000   Keep the 50 k highest-scoring unigrams and bigrams.
                           Fewer features → faster but less accurate.
                           More features → slower with diminishing returns.
      ngram_range=(1, 2)   Capture single words ('breaking', 'Reuters') AND
                           two-word phrases ('breaking news', 'white house').
      stop_words="english" Drop very common words (the, is, of …) that carry
                           no classification signal.
      min_df=2             Ignore tokens that appear in only one document –
                           they are almost certainly typos or noise.
      sublinear_tf=True    Replace raw term frequency tf with log(1 + tf).
                           This prevents one very frequent word from dominating
                           the feature vector for long articles.

    LogisticRegression:
      C=1.0 (default)      Regularisation strength (inverse).  Larger C = less
                           regularisation.  1.0 is a safe starting point for
                           TF-IDF features.
      max_iter=1000        The default of 100 often does not converge on large
                           text datasets.  1000 gives the solver enough steps.
      class_weight="balanced"
                           If one class has more samples, the solver up-weights
                           the minority class so both errors are penalised
                           equally.  Important when FAKE and REAL counts differ.
      solver="lbfgs"       Memory-efficient quasi-Newton solver.  Works well
                           for dense feature spaces like TF-IDF.
    """
    print("  Building TF-IDF + Logistic Regression pipeline …")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])

    print("  Training … (this may take 30–90 seconds on a large dataset)")
    model.fit(X_train, y_train)
    print("  Training complete.")
    print()

    # ── Predictions on the held-out test set ONLY ─────────────────────────
    # We NEVER call model.predict() on X_train for evaluation purposes.
    # Using training data for evaluation would give falsely high scores
    # because the model has already "seen" those examples.
    predictions = model.predict(X_test)

    # model.classes_ is set by scikit-learn after fit(); it lists the unique
    # label strings in sorted order, e.g. ['FAKE', 'REAL'].
    labels = list(model.classes_)

    # ── Overall metrics (weighted average across both classes) ────────────
    acc  = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, average="weighted", zero_division=0)
    rec  = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1   = f1_score(y_test, predictions, average="weighted", zero_division=0)

    # ── Per-class metrics (output_dict=True returns a nested dict) ─────────
    # This is more informative than the weighted average alone: it shows
    # how well the model performs on EACH class individually.
    report_dict = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )
    per_class = {
        label: {
            "precision": round(report_dict[label]["precision"], 4),
            "recall":    round(report_dict[label]["recall"],    4),
            "f1_score":  round(report_dict[label]["f1-score"],  4),
            "support":   int(report_dict[label]["support"]),
        }
        for label in labels
    }

    # ── Confusion matrix ──────────────────────────────────────────────────
    # cm[i][j] = number of articles whose actual class is labels[i]
    #            and whose predicted class is labels[j].
    # Diagonal cells are correct predictions; off-diagonal are errors.
    cm = confusion_matrix(y_test, predictions, labels=labels).tolist()

    # ── Console output ────────────────────────────────────────────────────
    print("=" * 62)
    print("  EVALUATION RESULTS  (held-out 20 % test set)")
    print("=" * 62)
    print(f"  Accuracy        : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  Precision (wtd) : {prec:.4f}")
    print(f"  Recall    (wtd) : {rec:.4f}")
    print(f"  F1-score  (wtd) : {f1:.4f}")
    print()

    # Per-class table
    col_w = 10
    print(f"  {'Class':<8}  {'Precision':>{col_w}}  {'Recall':>{col_w}}  "
          f"{'F1-score':>{col_w}}  {'Support':>{col_w}}")
    print(f"  {'-'*8}  {'-'*col_w}  {'-'*col_w}  {'-'*col_w}  {'-'*col_w}")
    for label in labels:
        m = per_class[label]
        print(f"  {label:<8}  {m['precision']:>{col_w}.4f}  "
              f"{m['recall']:>{col_w}.4f}  "
              f"{m['f1_score']:>{col_w}.4f}  "
              f"{m['support']:>{col_w},d}")
    print()

    # Confusion matrix grid
    print("  Confusion matrix  (rows = actual label, cols = predicted label)")
    header = "  " + " " * 8 + "".join(f"  {lb:>8}" for lb in labels)
    print(header)
    for i, row in enumerate(cm):
        cells = "".join(f"  {v:>8,d}" for v in row)
        print(f"  {labels[i]:<8}{cells}")
    print()
    print("  Full scikit-learn classification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("=" * 62)
    print()

    # ── Save confusion matrix heatmap ─────────────────────────────────────
    cm_plot_path = os.path.join(model_dir, "confusion_matrix.png")
    save_confusion_matrix_plot(cm, labels, cm_plot_path)
    abs_cm_path = os.path.abspath(cm_plot_path)
    print(f"  ✓ Confusion matrix plot  →  {abs_cm_path}")

    # ── Build the metrics dictionary for metrics.json ─────────────────────
    metrics = {
        "accuracy":         round(acc,  4),
        "precision":        round(prec, 4),
        "recall":           round(rec,  4),
        "f1_score":         round(f1,   4),
        "per_class":        per_class,
        "confusion_matrix": cm,
        "labels":           labels,
        "train_size":       int(len(X_train)),
        "test_size":        int(len(X_test)),
    }

    return model, metrics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    print()
    print("=" * 62)
    print("  NEWS ARTICLE CLASSIFIER  –  Training Script")
    print("=" * 62)
    print()

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("STEP 1/4 – Loading and cleaning dataset …")
    data = load_dataset()

    if data["label"].nunique() < 2:
        raise ValueError(
            "After loading and cleaning, the dataset contains only one class.\n"
            "Both 'FAKE' and 'REAL' labels must be present to train a classifier."
        )

    print("STEP 2/4 – Dataset summary")
    print_dataset_summary(data)

    print("STEP 3/4 – Splitting into train / test sets …")
    X_train, X_test, y_train, y_test = split_data(data)

    print("STEP 4/4 – Training and evaluating model …")
    model, metrics = build_and_train(X_train, X_test, y_train, y_test)

    # ── Save artefacts ────────────────────────────────────────────────────
    # joblib serialises the entire Pipeline object (vectorizer vocabulary +
    # classifier weights) into a single file.  The Streamlit app loads this
    # one file and gets both the TF-IDF transformer and the classifier back.
    joblib.dump(model, MODEL_PATH)
    abs_model_path = os.path.abspath(MODEL_PATH)
    print(f"  ✓ Trained model saved  →  {abs_model_path}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    abs_metrics_path = os.path.abspath(METRICS_PATH)
    print(f"  ✓ Metrics saved        →  {abs_metrics_path}")

    print()
    print("  All done!  Next step:  streamlit run app.py")
    print()


if __name__ == "__main__":
    main()
