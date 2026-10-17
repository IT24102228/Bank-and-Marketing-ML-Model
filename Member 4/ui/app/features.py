"""
features.py  (Member 4, IT24100001)

Shared by the Part 3 notebook and app.py. Keep this file in the same folder as
app.py: the saved pipeline (best_pipeline.joblib) refers to
features.engineer_features by name, so Python must be able to import it.

  engineer_features(df)   Member 2's feature rules, copied exactly (sections 1.3 to 1.9)
  prepare_input(df, b)    cleans uploaded or typed customer data before scoring
  assign_decile(s, edges) places scores into priority deciles using the saved cut points
  score_customers(df, b)  score -> CALL decision -> decile -> expected rate, ranked
"""

import numpy as np
import pandas as pd

RAW_COLUMNS = ["age", "job", "marital", "education", "default", "balance", "housing",
               "loan", "contact", "day", "month", "campaign", "pdays", "previous", "poutcome"]
NUMERIC_COLUMNS = ["age", "balance", "day", "campaign", "pdays", "previous"]
NOT_FEATURES = ["duration", "y"]   # duration is only known after the call (leakage)


def engineer_features(df):
    """Member 2's 8 engineered features. Fixed rules only: nothing is learned from data."""
    out = df.copy()
    out = out.drop(columns=[c for c in NOT_FEATURES if c in out.columns])

    # 1.3  pdays == -1 means the customer was never contacted before
    out["previously_contacted"] = (out["pdays"] != -1).astype(int)
    # 1.4  -1 replaced by 0; previously_contacted keeps the "never" information
    out["pdays_clean"] = out["pdays"].replace(-1, 0)
    # 1.5  loans and credit default
    out["loan_count"] = (out["housing"] == "yes").astype(int) + (out["loan"] == "yes").astype(int)
    out["has_credit_default"] = (out["default"] == "yes").astype(int)
    # 1.6  previous campaign success
    out["previous_campaign_success"] = (out["poutcome"] == "success").astype(int)
    # 1.7  age group
    out["age_group"] = pd.cut(out["age"], bins=[17, 30, 40, 50, 60, np.inf],
                              labels=["18-30", "31-40", "41-50", "51-60", "60+"])
    # 1.8  balance group
    out["balance_group"] = pd.cut(out["balance"], bins=[-np.inf, -1, 0, 1000, 5000, np.inf],
                                  labels=["negative", "zero", "low", "medium", "high"])
    # 1.9  campaign contact group
    out["campaign_contact_group"] = pd.cut(out["campaign"], bins=[0, 1, 2, 4, np.inf],
                                           labels=["first_contact", "second_contact",
                                                   "three_to_four", "five_or_more"])
    return out


def prepare_input(df, bundle=None):
    """Clean customer rows so they match the training format.

    Returns (clean_df, warnings). Raises ValueError if required columns are missing.
    Extra columns (for example a customer_id) are kept and passed through untouched.
    """
    df = df.copy()
    df.columns = [str(c).strip().strip('"').lower() for c in df.columns]
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    warnings = []
    for c in NUMERIC_COLUMNS:
        before = df[c].isna().sum()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        bad = int(df[c].isna().sum() - before)
        if bad:
            warnings.append(f"{bad} row(s) had a non-numeric '{c}'; the training median is used instead.")
    for c in [c for c in RAW_COLUMNS if c not in NUMERIC_COLUMNS]:
        df[c] = df[c].astype(str).str.strip().str.strip('"').str.lower()
        if bundle is not None and c in bundle.get("categories", {}):
            unknown = sorted(set(df[c]) - set(bundle["categories"][c]))
            if unknown:
                warnings.append(f"'{c}' has values never seen in training {unknown}; "
                                f"they are treated as 'no information'.")
    return df, warnings


def assign_decile(scores, edges):
    """Decile 1 = highest scores, 10 = lowest. `edges` are the 9 saved cut points (ascending)."""
    return 10 - np.searchsorted(np.asarray(edges), np.asarray(scores), side="right")


def score_customers(df, bundle):
    """Score raw customers and return them as a ranked calling queue.

    Adds: rank, score, recommendation, priority_decile, expected_rate.
    Uses only saved objects: nothing is fitted here.
    """
    clean, _ = prepare_input(df, bundle)
    score = bundle["pipeline"].predict_proba(clean[RAW_COLUMNS])[:, 1]   # Member 3 model score
    decile = assign_decile(score, bundle["decile_edges"])                 # Part 2 cut points
    rates = bundle["decile_expected_rate"]                                # Part 2 expected rates

    out = clean.copy()
    out["score"] = score
    out["recommendation"] = np.where(score >= bundle["threshold"], "CALL", "Lower-cost channel")
    out["priority_decile"] = decile
    out["expected_rate"] = [rates[int(d)] for d in decile]
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def decile_ranges(edges):
    """Readable score range for each decile, for display."""
    e = list(edges)
    rows = []
    for d in range(1, 11):
        if d == 1:
            rng = f"score >= {e[8]:.3f}"
        elif d == 10:
            rng = f"score < {e[0]:.3f}"
        else:
            rng = f"{e[9 - d]:.3f} <= score < {e[10 - d]:.3f}"
        rows.append({"decile": d, "score range": rng})
    return pd.DataFrame(rows)
