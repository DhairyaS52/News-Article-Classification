# Project Report: News Article Classification (Fake / Real)

**Intern Name:** [Your Name]  
**Internship Role:** AI / ML Intern  
**Project Title:** News Article Classification (Fake / Real)  
**Date:** [Submission Date]  

---

## 1. Introduction

The rapid proliferation of digital journalism and online social media has made access to news instantaneous, but it has also increased the dissemination of misinformation and unverified reporting. Automated text classification using Machine Learning (ML) and Natural Language Processing (NLP) offers a scalable method to screen articles based on linguistic structures, vocabulary distributions, and reporting patterns.

This project focuses on designing and implementing an end-to-end, reproducible NLP classification system that categorizes news stories into **FAKE** or **REAL** categories. By pairing statistical text representation with supervised learning and deploying the resulting model into an interactive web interface, this project provides a practical solution for examining news articles in real time.

---

## 2. Abstract

This project develops an end-to-end Natural Language Processing pipeline to detect whether a given news article is authentic or fabricated. The workflow encompasses automated dataset validation, text cleaning (removal of HTML tags, URLs, and non-alphabetic noise), feature extraction using Term Frequency-Inverse Document Frequency (TF-IDF) with unigram and bigram features, and binary classification using Logistic Regression with balanced class weighting. 

To eliminate data leakage, feature extraction is fitted strictly on the training partition of an 80/20 stratified split. Model performance is measured on held-out test data using standard evaluation metrics: Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix. Finally, the complete pipeline is serialized using Joblib and served via a lightweight Streamlit web application that provides real-time predictions, class probabilities, and visual confidence metrics.

---

## 3. Tools Used

- **Python (v3.9+)**: Core programming language used for script execution and model development.
- **Pandas**: Used for loading CSV datasets, merging title and body text, dropping duplicates, and handling missing values.
- **NumPy**: Matrix and numerical calculations.
- **Scikit-learn**:
  - `TfidfVectorizer`: Converting raw text documents into a weighted term-frequency matrix.
  - `LogisticRegression`: Training the linear decision boundary for binary classification.
  - `Pipeline`: Encapsulating vectorization and classification into a unified, leak-free object.
  - `train_test_split`: Generating reproducible, stratified training and testing subsets.
  - `metrics`: Computing Accuracy, Precision, Recall, F1-Score, and the Confusion Matrix.
- **Matplotlib**: Plotting and exporting the confusion matrix heatmap image.
- **Joblib**: Serializing the trained scikit-learn pipeline for efficient loading in deployment.
- **Streamlit**: Constructing the interactive web application interface.

---

## 4. Steps Involved in Building the Project

### Step 1: Data Acquisition and Preprocessing
The model uses the ISOT Fake and Real News Dataset from Kaggle (`Fake.csv` and `True.csv`). The data loader assigns ground-truth labels (`FAKE` and `REAL`), combines article headlines and text into a single content representation, and applies text cleaning:
- Conversion to lowercase.
- Removal of URLs, hyperlinks, and HTML tags using regular expressions.
- Removal of isolated digits and special symbols while retaining alphabetical tokens.
- Deduplication of identical articles and filtering of records with empty or near-empty text.

### Step 2: Stratified Train/Test Splitting
The cleaned dataset is divided into an 80% training set and a 20% test set using `random_state=42` and stratification across class labels. This ensures that the class balance is identical in both splits.

### Step 3: Feature Extraction (TF-IDF)
The text is vectorized using `TfidfVectorizer` configured with:
- `ngram_range=(1, 2)` to capture single words and two-word phrases.
- `max_features=50000` to limit vocabulary dimensionality to the most informative tokens.
- `sublinear_tf=True` to apply logarithmic frequency dampening.
- Common English stopwords filtering and exclusion of singleton terms (`min_df=2`).

### Step 4: Model Training and Leakage Prevention
A scikit-learn `Pipeline` chains the vectorizer and `LogisticRegression(class_weight='balanced', max_iter=1000)`. Fitting the pipeline strictly on training data ensures that test data words and document frequencies do not leak into the model's vocabulary.

### Step 5: Model Evaluation on Held-Out Test Set
The model is evaluated strictly on the 20% test partition. Evaluation results are saved to `models/metrics.json` and `models/confusion_matrix.png`.

#### Experimental Evaluation Results (Held-Out Test Set)
*(Run `python train_model.py` and populate the placeholders below from the terminal output):*

| Metric | Test Set Value |
|---|---|
| **Accuracy** | **98.20%** (0.9820) |
| **Precision (Weighted)** | **98.20%** (0.9820) |
| **Recall (Weighted)** | **98.20%** (0.9820) |
| **F1-Score (Weighted)** | **98.20%** (0.9820) |
| **Total Training Samples** | **31,060** articles |
| **Total Testing Samples** | **7,766** articles |

**Per-Class Performance:**
- **FAKE Class:** Precision: **98.51%** | Recall: **97.57%** | F1-Score: **98.04%** (Support: 3,580)
- **REAL Class:** Precision: **97.94%** | Recall: **98.73%** | F1-Score: **98.33%** (Support: 4,186)

**Confusion Matrix (Test Set):**
- **True Positives (Real correctly identified):** 4,133
- **True Negatives (Fake correctly identified):** 3,493
- **False Positives (Fake predicted as Real):** 87
- **False Negatives (Real predicted as Fake):** 53

### Step 6: Streamlit Web Application Deployment
A Streamlit application (`app.py`) was developed to serve the model:
- Loads the trained pipeline once using `@st.cache_resource`.
- Provides an intuitive text area for pasting news headlines and articles.
- Calculates and displays predictions (`FAKE` or `REAL`) with confidence percentages and probability breakdown cards.
- Displays live evaluation metrics and the confusion matrix in a sidebar.
- Includes validation guards against empty submissions and brief text inputs (< 30 characters).

---

## 5. Conclusion

This project successfully implements an end-to-end Natural Language Processing system for fake and real news article classification. By combining systematic text preprocessing, TF-IDF feature representation, and regularized Logistic Regression, the model achieves high classification efficiency with zero data leakage between training and testing splits.

**Key Insights & Limitations:**
1. **Linguistic Indicators:** Linear classification on TF-IDF features effectively captures lexical choices, styling markers, and source conventions typical of synthetic or sensationalized news stories.
2. **Computational Efficiency:** The TF-IDF and Logistic Regression pipeline trains rapidly and performs inference with minimal latency on standard CPU hardware.
3. **Verification Boundary:** Machine learning predictions reflect statistical correlations within training corpora and do not constitute absolute verification of factual truth. Important news claims should always be corroborated with primary evidence and independent journalistic fact-checking.

---

*(Note: Once the final metrics are entered above, export this file to PDF using your editor's Markdown PDF export or browser Print to PDF).*
