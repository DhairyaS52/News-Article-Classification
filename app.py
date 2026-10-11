"""
app.py
------
Streamlit web application for the News Article Classifier project.

Loads the trained scikit-learn Pipeline (TF-IDF + Logistic Regression) that
was saved by train_model.py and uses it to classify articles pasted by the
user.  No retraining happens here – the model is loaded once at start-up and
reused for every prediction.

Run:
    streamlit run app.py
"""

import json
import os
import re

import joblib
import streamlit as st

# Import text cleaning used during training for exact preprocessing parity
try:
    from train_model import clean_text
except ImportError:
    def clean_text(raw: str) -> str:
        text = str(raw)
        text = re.sub(r"^[a-zA-Z\s,./-]+?\(reuters\)\s*[-:]\s*", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\([a-zA-Z\s,./-]*reuters\)\s*[-:]\s*", " ", text, flags=re.IGNORECASE)
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\breuters\b", " ", text)
        text = re.sub(r"\b(image via|featured image|getty images)\b", " ", text)
        text = re.sub(r"\b\d+\b", " ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


# ---------------------------------------------------------------------------
# Paths – must match the constants in train_model.py exactly
# ---------------------------------------------------------------------------
MODEL_PATH   = os.path.join("models", "news_classifier.joblib")
METRICS_PATH = os.path.join("models", "metrics.json")
CM_IMG_PATH  = os.path.join("models", "confusion_matrix.png")

MIN_CHARS = 30   # shortest article text we will accept

# ---------------------------------------------------------------------------
# Page configuration  (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="News Article Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS  – clean, professional look without external frameworks
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Header band ── */
.app-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 14px;
    padding: 2rem 2.5rem 1.6rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.25);
}
.app-header h1 {
    color: #ffffff;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.35rem;
    letter-spacing: -0.5px;
}
.app-header p {
    color: #94a3b8;
    font-size: 0.97rem;
    margin: 0;
    line-height: 1.55;
}

/* ── Prediction banners ── */
.result-fake {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border-left: 5px solid #ef4444;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin: 1rem 0;
}
.result-real {
    background: linear-gradient(135deg, #14532d, #166534);
    border-left: 5px solid #22c55e;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin: 1rem 0;
}
.result-uncertain {
    background: linear-gradient(135deg, #78350f, #92400e);
    border-left: 5px solid #f59e0b;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin: 1rem 0;
}
.result-label {
    font-size: 1.55rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.5px;
    margin: 0 0 0.2rem;
}
.result-sub {
    font-size: 0.88rem;
    color: rgba(255,255,255,0.75);
    margin: 0;
}

/* ── Confidence bar ── */
.conf-bar-bg {
    background: #1e293b;
    border-radius: 999px;
    height: 10px;
    margin-top: 0.5rem;
    overflow: hidden;
}
.conf-bar-fill-fake {
    background: linear-gradient(90deg, #ef4444, #f87171);
    height: 100%;
    border-radius: 999px;
    transition: width 0.4s ease;
}
.conf-bar-fill-real {
    background: linear-gradient(90deg, #22c55e, #86efac);
    height: 100%;
    border-radius: 999px;
    transition: width 0.4s ease;
}
.conf-bar-fill-uncertain {
    background: linear-gradient(90deg, #f59e0b, #fbbf24);
    height: 100%;
    border-radius: 999px;
    transition: width 0.4s ease;
}

/* ── Score cards ── */
.score-card {
    background: #1e293b;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    text-align: center;
}
.score-card .sc-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 0.35rem;
}
.score-card .sc-value {
    font-size: 1.55rem;
    font-weight: 700;
    color: #f1f5f9;
}

/* ── Metric cards (sidebar) ── */
.metric-card {
    background: #1e293b;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.7rem;
}
.metric-card .mc-label {
    font-size: 0.75rem;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
.metric-card .mc-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: #38bdf8;
}

/* ── Disclaimer ── */
.disclaimer {
    background: #1e293b;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    margin-top: 1.2rem;
    font-size: 0.84rem;
    color: #94a3b8;
    line-height: 1.55;
}

/* ── Streamlit widget tweaks ── */
.stTextArea textarea {
    background: #0f172a !important;
    color: #e2e8f0 !important;
    border: 1.5px solid #334155 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.93rem !important;
}
.stTextArea textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.15) !important;
}
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 2rem !important;
    font-size: 0.97rem !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s ease !important;
}
div[data-testid="stButton"] button:hover {
    opacity: 0.88 !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper – load model once per session (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_model_cached(path: str):
    return joblib.load(path)


def load_model(path: str):
    """
    Load the trained scikit-learn Pipeline from disk.

    @st.cache_resource caches the model once loaded, but we only invoke the
    cache if the file actually exists so that None is never permanently cached
    if the app was opened before training finished.
    """
    if not os.path.exists(path):
        return None
    return _load_model_cached(path)


@st.cache_data(show_spinner=False)
def _load_metrics_cached(path: str):
    with open(path) as f:
        return json.load(f)


def load_metrics(path: str):
    """Load metrics.json produced by train_model.py, or return None."""
    if not os.path.exists(path):
        return None
    return _load_metrics_cached(path)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(metrics: dict | None) -> None:
    with st.sidebar:
        st.markdown("## 📊 Model Info")
        st.markdown("---")

        if metrics:
            st.markdown("**Overall Test-Set Performance**")

            def mc(label, value):
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="mc-label">{label}</div>'
                    f'<div class="mc-value">{value}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            mc("Accuracy",  f"{metrics['accuracy']  * 100:.2f}%")
            mc("Precision", f"{metrics['precision'] * 100:.2f}%")
            mc("Recall",    f"{metrics['recall']    * 100:.2f}%")
            mc("F1-Score",  f"{metrics['f1_score']  * 100:.2f}%")

            st.markdown("---")
            st.markdown("**Train / Test Split**")
            st.markdown(
                f"- Training: **{metrics.get('train_size', '—'):,}** articles\n"
                f"- Test: **{metrics.get('test_size', '—'):,}** articles"
            )

            # Per-class table if available
            per_class = metrics.get("per_class")
            if per_class:
                st.markdown("---")
                st.markdown("**Per-Class Metrics**")
                for lbl, vals in per_class.items():
                    icon = "🔴" if lbl == "FAKE" else "🟢"
                    with st.expander(f"{icon} {lbl}", expanded=False):
                        col1, col2 = st.columns(2)
                        col1.metric("Precision", f"{vals['precision']*100:.1f}%")
                        col2.metric("Recall",    f"{vals['recall']*100:.1f}%")
                        col3, col4 = st.columns(2)
                        col3.metric("F1-Score",  f"{vals['f1_score']*100:.1f}%")
                        col4.metric("Support",   f"{vals['support']:,}")

            # Confusion matrix image
            if os.path.exists(CM_IMG_PATH):
                st.markdown("---")
                st.markdown("**Confusion Matrix**")
                st.image(CM_IMG_PATH, use_container_width=True)

        else:
            st.info(
                "No metrics found.\n\n"
                "Run `python train_model.py` first to generate "
                "`models/metrics.json`."
            )

        st.markdown("---")
        st.markdown("**Pipeline**")
        st.markdown(
            "TF-IDF (50 k features, 1–2 grams)  \n"
            "→ Logistic Regression (balanced)"
        )
        st.markdown("**Dataset**")
        st.markdown(
            "[Kaggle ISOT](https://www.kaggle.com/datasets/"
            "clmentbisaillon/fake-and-real-news-dataset)"
        )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
def main() -> None:

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="app-header">
            <h1>📰 News Article Classifier</h1>
            <p>
                Paste a news headline and article text below.
                The model uses <strong>TF-IDF</strong> feature extraction and a
                <strong>Logistic Regression</strong> classifier to predict whether
                the article is <strong>FAKE</strong> or <strong>REAL</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load model & metrics ─────────────────────────────────────────────────
    model   = load_model(MODEL_PATH)
    metrics = load_metrics(METRICS_PATH)

    render_sidebar(metrics)

    # ── Model missing – explain and stop ─────────────────────────────────────
    if model is None:
        st.error(
            "**Trained model not found** at `models/news_classifier.joblib`\n\n"
            "The model file is created when you run the training script. "
            "Follow these steps:\n\n"
            "1. Download `Fake.csv` and `True.csv` from [Kaggle]"
            "(https://www.kaggle.com/datasets/clmentbisaillon/"
            "fake-and-real-news-dataset) and place them in the `data/` folder.\n"
            "2. Open a terminal in this project folder and run:\n"
            "```\npython train_model.py\n```\n"
            "3. Once training finishes, refresh this page."
        )
        st.stop()

    # ── Input area ───────────────────────────────────────────────────────────
    st.markdown("### Paste your article below")

    article = st.text_area(
        label="article_input",
        label_visibility="collapsed",
        height=240,
        placeholder=(
            "Paste the full news headline and article text here.\n\n"
            "Example: \"BREAKING – Scientists discover new treatment for … "
            "According to a study published in …\""
        ),
        key="article_input",
    )

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        classify_clicked = st.button(
            "🔍 Classify Article",
            type="primary",
            use_container_width=True,
        )

    # ── Validation ───────────────────────────────────────────────────────────
    if classify_clicked:
        stripped = article.strip()

        if len(stripped) == 0:
            st.warning("⚠️ Please paste an article before clicking Classify.")
            st.stop()

        if len(stripped) < MIN_CHARS:
            st.warning(
                f"⚠️ The text is too short ({len(stripped)} characters). "
                f"Please paste at least {MIN_CHARS} characters so the model "
                "has enough content to analyse."
            )
            st.stop()

        # ── Preprocessing & prediction ────────────────────────────────────────
        # Apply the exact same text-cleaning used during model training
        cleaned_input = clean_text(stripped)
        if len(cleaned_input) == 0:
            st.warning(
                "⚠️ After removing symbols, numbers, and URLs, no readable words remained. "
                "Please paste an article with readable English text."
            )
            st.stop()

        # The Pipeline object handles TF-IDF transformation internally using
        # the vocabulary that was learned from the training data.
        with st.spinner("Analysing article …"):
            prediction    = model.predict([cleaned_input])[0]          # 'FAKE' or 'REAL'
            probabilities = model.predict_proba([cleaned_input])[0]    # [p_class0, p_class1]
            classes       = list(model.classes_)                       # e.g. ['FAKE', 'REAL']


        label      = str(prediction).upper()
        confidence = float(max(probabilities)) * 100

        # Check for uncertain / borderline prediction (confidence near 50%)
        is_uncertain = confidence < 58.0

        # ── Prediction banner ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Result")

        if is_uncertain:
            icon          = "⚠️"
            css_class     = "result-uncertain"
            bar_class     = "conf-bar-fill-uncertain"
            display_label = f"UNCERTAIN ({label} lean)"
            sub_text      = (
                "The model is near its 50/50 decision boundary. "
                "The article lacks strong distinguishing signals from the training dataset."
            )
        elif label == "FAKE":
            icon          = "🚨"
            css_class     = "result-fake"
            bar_class     = "conf-bar-fill-fake"
            display_label = label
            sub_text      = "The model considers this article likely to be fabricated or misleading."
        else:
            icon          = "✅"
            css_class     = "result-real"
            bar_class     = "conf-bar-fill-real"
            display_label = label
            sub_text      = "The model considers this article likely to be authentic reporting."

        st.markdown(
            f"""
            <div class="{css_class}">
                <div class="result-label">{icon}  {display_label}</div>
                <div class="result-sub">{sub_text}</div>
                <div class="conf-bar-bg">
                    <div class="{bar_class}" style="width:{confidence:.1f}%"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Confidence + score cards ──────────────────────────────────────────
        st.markdown("#### Model confidence")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(
                f'<div class="score-card">'
                f'<div class="sc-label">Confidence</div>'
                f'<div class="sc-value">{confidence:.1f}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        for cls, prob in zip(classes, probabilities):
            col = c2 if cls == classes[0] else c3
            with col:
                st.markdown(
                    f'<div class="score-card">'
                    f'<div class="sc-label">P({cls})</div>'
                    f'<div class="sc-value">{prob * 100:.1f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Probability bar chart ─────────────────────────────────────────────
        st.markdown("#### Class probability breakdown")
        for cls, prob in zip(classes, probabilities):
            bar_pct = prob * 100
            colour  = "#ef4444" if cls == "FAKE" else "#22c55e"
            st.markdown(
                f"""
                <div style="margin-bottom:0.6rem">
                    <div style="display:flex;justify-content:space-between;
                                font-size:0.85rem;color:#94a3b8;margin-bottom:4px">
                        <span style="font-weight:600">{cls}</span>
                        <span>{bar_pct:.2f}%</span>
                    </div>
                    <div style="background:#1e293b;border-radius:999px;height:8px;overflow:hidden">
                        <div style="width:{bar_pct:.2f}%;background:{colour};
                                    height:100%;border-radius:999px;
                                    transition:width 0.4s ease"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if is_uncertain:
            st.warning(
                f"⚠️ **Low Confidence / Inconclusive Prediction Warning**\n\n"
                f"- The model's prediction confidence is only **{confidence:.1f}%** "
                f"({classes[0]}: **{probabilities[0]*100:.1f}%**, {classes[1]}: **{probabilities[1]*100:.1f}%**).\n"
                f"- **Why this happens:** The model was trained on 2016–2017 political news. "
                f"Articles on topics outside that domain (such as modern space missions like Artemis, science, or technology) "
                f"do not contain the specific political vocabulary learned by the model. Because neither fake nor real markers dominate, "
                f"the prediction hovers near a 50/50 toss-up.\n"
                f"- **Caution:** Do not treat this prediction as definitive. Please verify the claims using primary sources."
            )

        # ── Disclaimer ────────────────────────────────────────────────────────
        st.markdown(
            """
            <div class="disclaimer">
                ⚠️ <strong>Disclaimer:</strong>
                This prediction is produced by a machine-learning model trained on a specific
                dataset and <em>does not constitute proof</em> that an article is true or false.
                Model accuracy is limited by the training data and writing style.
                Always verify important claims using reputable, independent sources
                before drawing conclusions.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── How it works expander (always visible) ────────────────────────────────
    with st.expander("ℹ️  How does this work?", expanded=False):
        st.markdown(
            """
            **Step-by-step:**

            1. **Training** (`train_model.py`): A labelled dataset of real and fake news
               articles is loaded, cleaned, and split into training and test sets.
               A **TF-IDF vectorizer** converts the article text into numerical features
               (up to 50,000 unigrams and bigrams). A **Logistic Regression** classifier
               is then trained on these features.

            2. **Saving**: The entire scikit-learn Pipeline (vectorizer + classifier) is
               saved to `models/news_classifier.joblib` using `joblib`.

            3. **Prediction** (this app): The saved Pipeline is loaded once.
               When you paste an article and click *Classify*, the Pipeline
               applies the **same TF-IDF vocabulary** learned during training
               and the classifier outputs a probability for each class.
               The class with the higher probability is shown as the prediction.

            **Why can the model be wrong?**
            - It was trained on one specific dataset; unseen writing styles may confuse it.
            - Satire, parody, and opinion pieces can look like "fake news" to a word-frequency model.
            - Dataset bias (e.g. source-specific language patterns) can inflate apparent accuracy.
            """
        )


if __name__ == "__main__":
    main()
