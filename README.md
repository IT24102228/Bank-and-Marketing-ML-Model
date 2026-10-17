# Member 2: Feature Engineering and Feature Selection

**Student ID:** IT24101461 · **Project:** IT3091 Assignment 52, Bank Marketing (term deposit prediction)

Member 2 takes the cleaned train/test split from Member 1 and prepares the features the models learn from. The work has four steps: create new features, build a preprocessing pipeline, select the most useful columns, and save everything for Members 3 and 4.

```
Member 1: train/test split (15 columns)
   → 8 engineered features (23 columns)
   → scaling + one-hot encoding (68 columns)
   → feature selection (30 columns)
   → saved for Member 3 (models) and Member 4 (app)
```

---

## 1. What this member did

### Step 1: Engineered features

Eight new features, built only from information available **before** the marketing decision. The target `y` is never used to create them, and the same rules are applied separately to the training and test sets.

| New feature | Rule | Why |
|---|---|---|
| `previously_contacted` | 1 if `pdays` is not -1, else 0 | `pdays = -1` means the customer was never contacted before; this makes that explicit |
| `pdays_clean` | `pdays` with -1 replaced by 0 | Removes the misleading -1, while `previously_contacted` keeps the "never contacted" information |
| `loan_count` | 1 point each for a housing loan and a personal loan (0 to 2) | Summarises a customer's loan burden |
| `has_credit_default` | 1 if the customer has credit in default | Kept separate because it describes repayment status, not another loan |
| `previous_campaign_success` | 1 if `poutcome` is "success" | Flags customers who responded well before |
| `age_group` | 18-30, 31-40, 41-50, 51-60, 60+ | Captures non-linear differences between age groups |
| `balance_group` | negative, zero, low (up to 1,000), medium (up to 5,000), high | Groups balances into financial categories; the original `balance` is kept too |
| `campaign_contact_group` | first contact, second contact, three to four, five or more | Represents how much contact effort was spent |

**Checks run:** the training set grows from 15 to 23 columns and the test set the same; the 8 new features are exactly the expected ones; there are no missing values; training and test columns match.

### Step 2: Preprocessing pipeline

A `ColumnTransformer` that is **fitted on the training data only** and then applied to the test data:

| Columns | Steps |
|---|---|
| 10 numeric columns (`age`, `balance`, `day`, `campaign`, `previous`, `previously_contacted`, `pdays_clean`, `loan_count`, `has_credit_default`, `previous_campaign_success`) | median imputation, then standard scaling |
| 12 categorical columns (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`, `age_group`, `balance_group`, `campaign_contact_group`) | most-frequent imputation, then one-hot encoding (first category dropped as the reference; unseen categories ignored) |

The raw `pdays` column is dropped, because `pdays_clean` replaces it. The result is **68 columns** with no missing values.

### Step 3: Feature selection

`SelectKBest` with `mutual_info_classif`, keeping the **30** best columns. It is fitted on the **training data only**, so the test set never influences which features are chosen.

### Step 4: Save outputs

Three files for the next members:

| File | Contents | Used by |
|---|---|---|
| `IT24101461_engineered_data.joblib` | `X_train`, `X_test` (23 columns, before scaling), `y_train`, `y_test` | Members 3 and 4 |
| `IT24101461_selected_data.joblib` | the 30 selected columns for train and test, the labels, and the selected feature names | Member 3 (training the models) |
| `IT24101461_feature_pipeline.joblib` | the fitted pipeline: preprocessor + selector, ready to transform new customers | Member 4 (the app scores new customers with it) |

---

## 2. Data flow

| | Shape |
|---|---|
| Input from Member 1 (`X_train` / `X_test`) | 36,168 / 9,043 rows × 15 columns |
| After feature engineering | × 23 columns |
| After preprocessing | × 68 columns |
| After feature selection | × 30 columns |

---

## 3. Folder contents

```
Member 2/
├── README.md                              this file
├── colab notebook/
│   └── IT24101461_FeatureEngineering.ipynb
└── .joblib files/
    ├── IT24101461_engineered_data.joblib
    ├── IT24101461_selected_data.joblib
    └── IT24101461_feature_pipeline.joblib
```

---

## 4. How to run it

1. Get Member 1's `ds52-train_test_split.joblib` (in `Member 1/.joblib files/`).
2. Put it in the same folder the notebook runs from. The notebook loads it by file name, and the three output files are saved to that folder too.
3. Run the notebook from top to bottom.

---

## 5. Notes

- **No leakage:** the target is not used to create features, and the scaler, encoder and feature selector are fitted on the training set only.
- **Library versions:** the saved pipeline is a pickled scikit-learn object. Later members load it with a possibly different scikit-learn version, and Member 4's notebook applies a small compatibility fix when loading it. If you re-run this notebook, note the `scikit-learn` version you used so others can match it.