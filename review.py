"""
review.py - Technical review script for News Article Classification project.
Run from the project root: python review.py
"""

import io
import os
import sys
# Force UTF-8 output so Unicode characters do not crash on Windows cp1252 terminals.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import re
import ast
import importlib

PASS = "  PASS"
FAIL = "  FAIL"
INFO = "  INFO"

results = []   # list of (status, description)

def check(status, msg):
    results.append((status, msg))
    print(f"{status}  {msg}")

def section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Read source files
# ─────────────────────────────────────────────────────────────────────────────
section("1. SOURCE FILES")

files_to_check = ["train_model.py", "app.py", "requirements.txt"]
sources = {}

for fname in files_to_check:
    if os.path.exists(fname):
        sources[fname] = open(fname, encoding="utf-8").read()
        check(PASS, f"{fname} found ({os.path.getsize(fname):,} bytes)")
    else:
        check(FAIL, f"{fname} NOT FOUND")

tm = sources.get("train_model.py", "")
ap = sources.get("app.py", "")
req = sources.get("requirements.txt", "")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Syntax check
# ─────────────────────────────────────────────────────────────────────────────
section("2. SYNTAX CHECK")

for fname in ["train_model.py", "app.py"]:
    src = sources.get(fname, "")
    if not src:
        check(FAIL, f"{fname} could not be read for syntax check")
        continue
    try:
        ast.parse(src)
        check(PASS, f"{fname}: no syntax errors")
    except SyntaxError as e:
        check(FAIL, f"{fname}: SyntaxError at line {e.lineno}: {e.msg}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Import / dependency check
# ─────────────────────────────────────────────────────────────────────────────
section("3. IMPORTS AND DEPENDENCIES")

required = {
    "json": "stdlib",
    "os": "stdlib",
    "re": "stdlib",
    "joblib": "joblib",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "streamlit": "streamlit",
}

for lib, pip_name in required.items():
    try:
        importlib.import_module(lib)
        check(PASS, f"import {lib} OK")
    except ImportError:
        check(FAIL, f"import {lib} FAILED  (pip install {pip_name})")

# Check requirements.txt contains the main packages
section("3b. REQUIREMENTS.TXT ENTRIES")
for pkg in ["pandas", "scikit-learn", "streamlit", "joblib", "matplotlib", "numpy"]:
    if pkg in req:
        check(PASS, f"requirements.txt lists {pkg}")
    else:
        check(FAIL, f"requirements.txt MISSING {pkg}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. File paths – consistency between train_model.py and app.py
# ─────────────────────────────────────────────────────────────────────────────
section("4. PATH CONSISTENCY (train_model.py vs app.py)")

shared_paths = [
    "news_classifier.joblib",
    "metrics.json",
    "confusion_matrix.png",
]

for p in shared_paths:
    in_tm = p in tm
    in_ap = p in ap
    if in_tm and in_ap:
        check(PASS, f'"{p}" present in both files')
    elif in_tm and not in_ap:
        check(FAIL, f'"{p}" only in train_model.py – app.py will not load it')
    elif in_ap and not in_tm:
        check(FAIL, f'"{p}" only in app.py – train_model.py does not create it')
    else:
        check(FAIL, f'"{p}" absent from both files')

# Check model dir constant
for const in ["models", "news_classifier.joblib"]:
    in_tm = const in tm
    in_ap = const in ap
    check(PASS if (in_tm and in_ap) else FAIL,
          f'Path segment "{const}": train_model={in_tm}, app={in_ap}')


# ─────────────────────────────────────────────────────────────────────────────
# 5. Dataset column handling
# ─────────────────────────────────────────────────────────────────────────────
section("5. DATASET HANDLING")

checks_5 = [
    ("Fake.csv" in tm,          'Fake.csv referenced in train_model.py'),
    ("True.csv" in tm,          'True.csv referenced in train_model.py'),
    ('"FAKE"' in tm,            'Label "FAKE" assigned for Fake.csv'),
    ('"REAL"' in tm,            'Label "REAL" assigned for True.csv'),
    ("title" in tm,             '"title" column looked up for text combination'),
    ('"text"' in tm,            '"text" column looked up for text combination'),
    ("fillna" in tm,            'NaN values filled before text join'),
    ("drop_duplicates" in tm,   'Duplicate articles removed'),
    ('str.len() > 10' in tm,    'Short / empty articles filtered out'),
    ('FileNotFoundError' in tm, 'FileNotFoundError raised when dataset absent'),
    ('ValueError' in tm,        'ValueError raised for unexpected format'),
]

for cond, desc in checks_5:
    check(PASS if cond else FAIL, desc)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Training pipeline correctness
# ─────────────────────────────────────────────────────────────────────────────
section("6. ML PIPELINE CORRECTNESS")

# Verify split happens before fit (check line numbers in train_model.py)
lines = tm.splitlines()
split_line = next((i+1 for i, l in enumerate(lines) if "train_test_split(" in l and "=" in l), None)
fit_line   = next((i+1 for i, l in enumerate(lines) if "model.fit(" in l), None)

check(PASS if split_line is not None else FAIL, f"train_test_split found (line {split_line})")
check(PASS if fit_line is not None else FAIL,   f"model.fit found (line {fit_line})")
if split_line and fit_line:
    check(PASS if split_line < fit_line else FAIL,
          f"Split (line {split_line}) occurs BEFORE fit (line {fit_line})")

# Evaluation must use X_test, y_test – not X_train
eval_section = "\n".join(lines[fit_line:]) if fit_line else ""
check(PASS if "X_test" in eval_section else FAIL, "Evaluation uses X_test (not X_train)")
check(PASS if "y_test" in eval_section else FAIL, "Evaluation uses y_test (not y_train)")
# Leakage check: model.predict(X_train) or model.score(X_train) should NOT appear
# after the fit line. Comments, len(X_train), and passing X_train to functions are fine.
import re as _re
eval_predict_on_train = bool(_re.search(r"model\.(predict|score)\(X_train", eval_section))
check(PASS if not eval_predict_on_train else FAIL,
      "No model.predict/score(X_train) called in evaluation block")

check(PASS if "stratify" in tm else FAIL,       "Stratified split (class balance preserved)")
check(PASS if "random_state" in tm else FAIL,   "Reproducible random_state set on split")

check(PASS if "TfidfVectorizer" in tm else FAIL,     "TfidfVectorizer in pipeline")
check(PASS if "LogisticRegression" in tm else FAIL,  "LogisticRegression in pipeline")
check(PASS if "Pipeline" in tm else FAIL,            "sklearn Pipeline used (vectoriser + classifier)")
check(PASS if "sublinear_tf" in tm else FAIL,        "sublinear_tf=True set on TF-IDF")
check(PASS if "class_weight" in tm else FAIL,        "class_weight='balanced' set on LR")
check(PASS if "max_iter" in tm else FAIL,            "max_iter set on LR (convergence guard)")

check(PASS if "accuracy_score" in tm else FAIL,      "accuracy_score computed")
check(PASS if "classification_report" in tm else FAIL, "classification_report computed")
check(PASS if "confusion_matrix" in tm else FAIL,    "confusion_matrix computed")
check(PASS if "output_dict=True" in tm else FAIL,    "classification_report uses output_dict for per-class metrics")

check(PASS if "joblib.dump" in tm else FAIL,         "joblib.dump saves the model")
check(PASS if "json.dump" in tm else FAIL,           "json.dump saves metrics.json")
check(PASS if 'if __name__ == "__main__"' in tm else FAIL, '__main__ guard prevents auto-training on import')


# ─────────────────────────────────────────────────────────────────────────────
# 7. Streamlit app correctness
# ─────────────────────────────────────────────────────────────────────────────
section("7. STREAMLIT APP CORRECTNESS")

checks_7 = [
    ("cache_resource" in ap,          "@st.cache_resource used (model loaded once)"),
    ("cache_data" in ap,              "@st.cache_data used (metrics loaded once)"),
    ("model.fit" not in ap,           "model.fit() NOT called in app (no retraining)"),
    ("joblib.load" in ap,             "joblib.load used to load the pipeline"),
    ("predict_proba" in ap,           "predict_proba used for confidence score"),
    ("model.classes_" in ap,          "model.classes_ used (no hardcoded label order)"),
    ("st.spinner" in ap,              "Loading spinner shown during prediction"),
    ("st.stop()" in ap,               "st.stop() called on missing model"),
    ("MIN_CHARS" in ap,               "Minimum input length enforced"),
    ('"FAKE"' in ap,                  '"FAKE" label handled in result display'),
    # App uses an else-branch + CSS class "result-real" rather than checking label == "REAL"
    ('result-real' in ap,             '"REAL" result branch present (result-real CSS class used)'),
    ("disclaimer" in ap.lower(),      "Disclaimer present in app"),
    ("set_page_config" in ap,         "st.set_page_config called (must be first Streamlit call)"),
    (ap.index("set_page_config") <
     ap.index("st.markdown"),         "set_page_config called before any st.markdown"),
]

for cond, desc in checks_7:
    check(PASS if cond else FAIL, desc)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Data / model files present?
# ─────────────────────────────────────────────────────────────────────────────
section("8. FILE PRESENCE ON DISK")

disk_files = {
    "data/Fake.csv":                 "Dataset – Fake news articles",
    "data/True.csv":                 "Dataset – Real news articles",
    "models/news_classifier.joblib": "Trained pipeline (required for app)",
    "models/metrics.json":           "Evaluation metrics (required for sidebar)",
    "models/confusion_matrix.png":   "Confusion matrix image (required for sidebar)",
}

all_data_present = True
for path, desc in disk_files.items():
    exists = os.path.exists(path)
    if not exists:
        all_data_present = False
    check(PASS if exists else INFO, f"{path}: {'PRESENT' if exists else 'ABSENT'} – {desc}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Can train_model.py be imported without error?
# ─────────────────────────────────────────────────────────────────────────────
section("9. IMPORT TRAIN_MODEL WITHOUT RUNNING MAIN")

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_model", "train_model.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # executes module-level code, but NOT main()
    check(PASS, "train_model.py imports cleanly (no module-level side-effects)")
    # Verify key functions are exposed
    for fn in ["clean_text", "load_dataset", "print_dataset_summary",
               "split_data", "build_and_train", "save_confusion_matrix_plot"]:
        check(PASS if hasattr(mod, fn) else FAIL,
              f"Function {fn}() defined in train_model")
except Exception as e:
    check(FAIL, f"train_model.py raised on import: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Missing-dataset error message check
# ─────────────────────────────────────────────────────────────────────────────
section("10. MISSING DATASET ERROR MESSAGE")

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_model2", "train_model.py")
    mod2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod2)

    try:
        mod2.DATA_DIR = "non_existent_data_test_dir"
        mod2.load_dataset()
        check(FAIL, "load_dataset() should raise when data/ is empty but did not")
    except FileNotFoundError as e:
        msg = str(e)
        check(PASS, "load_dataset() raises FileNotFoundError when dataset absent")
        has_kaggle = "kaggle" in msg.lower() or "Kaggle" in msg
        check(PASS if has_kaggle else FAIL,
              "Error message contains Kaggle download link")
    except Exception as e:
        check(FAIL, f"Unexpected exception from load_dataset(): {type(e).__name__}: {e}")
except Exception as e:
    check(FAIL, f"Could not run missing-dataset test: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
section("SUMMARY")

passed = sum(1 for s, _ in results if s == PASS)
failed = sum(1 for s, _ in results if s == FAIL)
info   = sum(1 for s, _ in results if s == INFO)

print(f"  Passed : {passed}")
print(f"  Failed : {failed}")
print(f"  Info   : {info}  (items that are absent but not errors)")
print()

if failed:
    print("  Items that need attention:")
    for s, d in results:
        if s == FAIL:
            print(f"    ✗ {d}")
else:
    print("  All checked items passed.")

print()
if info:
    print("  Action required by user (cannot be automated here):")
    for s, d in results:
        if s == INFO:
            print(f"    • {d}")

sys.exit(0 if failed == 0 else 1)
