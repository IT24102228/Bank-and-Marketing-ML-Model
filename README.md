# Bank Marketing: Predicting Term Deposit Subscriptions

**IT3091 Machine Learning and Data Warehousing · Assignment 52 · SLIIT**

A four-person project that predicts which bank customers are likely to subscribe to a term deposit after a telemarketing call, and turns that prediction into a practical tool: a ranked **calling queue** a call centre can work through.

The project covers the full path from raw data to a working app:

```
raw data  →  cleaning  →  feature engineering  →  model training & tuning  →  evaluation & decision rules  →  Streamlit app
 (Member 1)              (Member 2)                (Member 3)                  (Member 4)                      (Member 4)
```

---

## Table of contents

1. [Quick facts](#1-quick-facts)
2. [The problem and the data](#2-the-problem-and-the-data)
3. [Team and responsibilities](#3-team-and-responsibilities)
4. [Repository structure](#4-repository-structure)
5. [Results](#5-results)
6. [Run the app](#6-run-the-app)
7. [Re-run the notebooks](#7-re-run-the-notebooks)
8. [Design decisions worth knowing](#8-design-decisions-worth-knowing)
9. [Troubleshooting](#9-troubleshooting)
10. [Credits](#10-credits)

---

## 1. Quick facts

| | |
|---|---|
| **Task** | Binary classification: will the customer subscribe (`y` = yes/no)? |
| **Dataset** | UCI Bank Marketing, `bank-full.csv` (in `DataSet/`) |
| **Final model** | XGBoost (tuned) |
| **Decision threshold** | 0.203, tuned on training data instead of the default 0.5 |
| **Key output** | Ranked calling queue with a CALL / lower-cost-channel recommendation and a priority decile (1 = call first) |
| **App** | Streamlit, in `Member 4/ui/app/` |
| **Python for the app** | 3.13 |

---

## 2. The problem and the data

A bank phones customers to sell term deposits. Only a small share say yes (about 11.7% in this data), so calling everyone is expensive and mostly fruitless. The question is: **who should the call centre phone first?**

- Each row is one customer contact, with 15 usable input columns (age, job, marital status, education, credit default, balance, housing loan, personal loan, contact type, day, month, number of contacts in this campaign, days since the previous campaign, number of previous contacts, outcome of the previous campaign) and the target `y`.
- **`duration` (call length) is deliberately never used.** It is only known after the call ends, so using it would be data leakage and the model would be useless for deciding whom to call.
- The classes are imbalanced, so accuracy is misleading. The project evaluates with **precision, recall, F1 and PR-AUC**.

---

## 3. Team and responsibilities

| Member | Student ID | Responsibility | Main outputs |
|---|---|---|---|
| **Member 1** | IT24102060 | Data cleaning and preprocessing; train/test split | `ds52-preprocessor.joblib`, `ds52-train_test_split.joblib` |
| **Member 2** | IT24101461 | Feature engineering and feature selection | engineered data, fitted feature pipeline, 30 selected columns |
| **Member 3** | IT24102228 | Modelling: baseline models, tuned models, final test evaluation, F1-based comparison | baseline model files; tuned models |
| **Member 4** | IT24100001 | Evaluation and explainability (Part 1), decision threshold and priority deciles (Part 2), packaging and Streamlit app (Part 3) | evaluation results, decision rules, `best_pipeline.joblib`, the app |

Each member works only in their own folder. Nothing in one member's folder is ever modified by another member's notebook; later stages only **read** earlier members' saved files.

---

## 4. Repository structure

```
Bank-and-Marketing-ML-Model/
├── README.md                      ← you are here
├── DataSet/
│   └── bank-full.csv              raw dataset
├── Report/                        written reports
│
├── Member 1/                      Data cleaning and preprocessing
│   ├── colab notebook/            IT24102060_DatacleaningPreprocessing.ipynb
│   └── .joblib files/             preprocessor and train/test split
│
├── Member 2/                      Feature engineering
│   ├── colab notebook/            IT24101461_FeatureEngineering.ipynb
│   └── .joblib files/             engineered data, feature pipeline, selected data
│
├── Member 3 files/                Modelling
│   ├── colab notebook/            IT24102228_Member3.ipynb
│   └── .joblib files/             baseline models (best model, LightGBM, logistic regression),
│                                  plus folders for tuned models and the final model write-up
│
└── Member 4/                      Evaluation, decision rules and the app
    ├── colab notebook/            Part 1 (evaluation), Part 2 (threshold and deciles), Part 3 (export and app)
    ├── .joblib files/             saved evaluation results and Part 2 decision rules
    ├── results/                   CSV outputs and figures used in the report
    └── ui/app/                    the Streamlit app (see section 6)
        ├── app.py                 the web app
        ├── features.py            feature rules and scoring helpers (must stay next to app.py)
        ├── best_pipeline.joblib   feature pipeline + Part 2 rules + app information
        ├── xgb_model.ubj          the trained XGBoost model (must stay next to app.py)
        ├── demo_new_customers.csv 200 unseen test customers for the demo
        ├── demo_answer_key.csv    what those customers really did
        ├── requirements.txt       exact library versions
        └── README.md              app-specific guide
```

> Large raw files (`DataSet/`) and model files are kept in the repo so the project can be inspected end to end. Virtual environments, caches and logs are ignored (see `.gitignore`).

---

## 5. Results

Final model: **XGBoost (tuned)**, evaluated on a held-out test set that no model was trained or tuned on, at the Part 2 threshold of 0.203.

| Metric | Test set |
|---|---|
| Precision | 0.450 |
| Recall | 0.480 |
| F1 | 0.465 |
| PR-AUC | 0.432 |
| ROC-AUC | 0.789 |

Why the threshold matters: at the default 0.5, F1 on the test set is only 0.329. Moving the threshold to 0.203 (chosen to maximise F1 on out-of-fold **training** predictions, never on test data) lifts it to 0.465.

The model also ranks customers into **10 priority deciles**. Decile 1 has the highest real subscription rate, so a team with a limited call budget should work down the deciles in order. The app shows each decile's expected rate, its real rate on the test set, and its lift versus random calling.

*(Figures above are from the Part 3 export. See `Member 4/results/` for the full tables.)*

---

## 6. Run the app

The app lives in `Member 4/ui/app/`. It does **not** train anything: it only loads the saved pipeline and model.

### Requirements
- **Python 3.13.x** (the pinned library versions match this)
- The four files `app.py`, `features.py`, `best_pipeline.joblib` and `xgb_model.ubj` must stay **in the same folder**

### Windows (PowerShell)
```powershell
cd "Member 4\ui\app"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux
```bash
cd "Member 4/ui/app"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at **http://localhost:8501**. Press `Ctrl + C` in the terminal to stop it. If Streamlit asks for an email address on the first run, just press Enter.

### What the app does

| Tab | What it does |
|---|---|
| **Single customer** | Enter one customer (or load a real test example) and get CALL or lower-cost channel, a priority decile and an expected subscription rate |
| **Campaign queue** | Upload a CSV of customers (or press *Use the demo customers*), set how many calls the team can make, and download the ranked queue |
| **How it works** | The rules behind every decision, including the decile table |

For a quick demo: choose **Start from → Real test customer: strongest prospect** and press **Score this customer**.

Using your own customer CSV? It needs the 15 input columns listed in section 2 (any order; extra columns such as `customer_id` are kept). See `Member 4/ui/app/README.md` for allowed values.

---

## 7. Re-run the notebooks

The notebooks are written for **Google Colab**, not for a local machine. They mount Google Drive and expect this layout in the Drive of whoever runs them:

```
My Drive/
└── ML & DWBI Assignment/
    └── ML Assignment/
        ├── IT24102060/   (Member 1)
        ├── IT24101461/   (Member 2)
        ├── IT24102228/   (Member 3)
        └── IT24100001/   (Member 4)
```

Run them in this order, each in Colab:

1. Member 1: cleaning and preprocessing
2. Member 2: feature engineering
3. Member 3: modelling
4. Member 4: Part 1 → Part 2 → Part 3

Running a notebook locally will fail on `from google.colab import drive` (`No module named 'google.colab'`). That is expected: open the notebook in Colab instead.

Some intermediate files (for example Member 3's tuned models) live in the team's Google Drive rather than in this repo.

---

## 8. Design decisions worth knowing

- **No data leakage.** `duration` is excluded. Thresholds and decile cut points are decided on training data only; the test set is used once, for reporting.
- **Threshold tuning.** The decision threshold is tuned on out-of-fold training predictions, not left at 0.5, because the classes are imbalanced.
- **Deciles instead of raw scores.** Class weighting inflates raw model probabilities, so the app shows each decile's real historical subscription rate instead.
- **Nothing is refitted in the app.** Scaling, encoding, feature selection and the model are all pre-fitted. The same customer always gets the same answer, whoever else is in the uploaded file.
- **Model stored as `xgb_model.ubj`.** The XGBoost model is saved in XGBoost's own format rather than pickled inside the joblib bundle. On some Windows setups, loading the pickled model failed with `XGBoostError: input stream corrupted`. The separate file avoids that. `app.py` loads it and plugs it back into the pipeline.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'google.colab'` | You ran a notebook locally. Open it in Google Colab |
| `streamlit` is not recognised | The virtual environment is not active. Run the activate line again |
| `ModuleNotFoundError: No module named 'features'` | Start the app from inside `Member 4/ui/app/`, with `features.py` next to `app.py` |
| `best_pipeline.joblib was not found` | Same as above: run `streamlit run app.py` from the folder that contains it |
| Error about `xgb_model.ubj` | The file is missing or not next to `app.py` |
| `pip` cannot find a version in `requirements.txt` | Use Python 3.13.x |
| PowerShell blocks `Activate.ps1` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |

---

## 10. Credits

- **Dataset:** S. Moro, P. Cortez and P. Rita, *A Data-Driven Approach to Predict the Success of Bank Telemarketing*, Decision Support Systems, 2014 (UCI Machine Learning Repository, Bank Marketing).
- **Team:** Members 1 to 4 as listed in section 3.
- **Course:** IT3091 Machine Learning and Data Warehousing, Sri Lanka Institute of Information Technology (SLIIT).
