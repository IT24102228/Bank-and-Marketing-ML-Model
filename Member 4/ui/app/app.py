"""
Term Deposit Call Planner  (Member 4, IT24100001)
Dual-lens decision interface for the Bank Marketing project.

  Tab 1  Single customer   (ML lens)          score one customer and explain the decision
  Tab 2  Campaign queue    (Analytical lens)  rank a CSV of customers into a calling queue
  Tab 3  How it works                         the rules behind every decision

Run from this folder:   streamlit run app.py
Needs in the same folder: best_pipeline.joblib, features.py
Optional (for the demo): demo_new_customers.csv, demo_answer_key.csv

Nothing is trained in this app. It only loads the saved pipeline
(Member 2 feature steps + Member 3 model) and the Part 2 decision rules.
"""
import os

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

import features  # must sit next to app.py: the saved pipeline refers to it by name

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE_PATH = os.path.join(HERE, "best_pipeline.joblib")
DEMO_CUSTOMERS = os.path.join(HERE, "demo_new_customers.csv")
DEMO_ANSWERS = os.path.join(HERE, "demo_answer_key.csv")

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
FIELDS = {   # label and help text for every raw input column
    "age": ("Age", "Customer age in years"),
    "job": ("Job", "Type of job"),
    "marital": ("Marital status", None),
    "education": ("Education", None),
    "default": ("Has credit in default?", None),
    "balance": ("Average yearly balance (EUR)", "Can be negative"),
    "housing": ("Has housing loan?", None),
    "loan": ("Has personal loan?", None),
    "contact": ("Contact type", "How the customer is contacted"),
    "day": ("Last contact day of month", "1 to 31"),
    "month": ("Last contact month", None),
    "campaign": ("Contacts in this campaign", "Including the last contact"),
    "pdays": ("Days since last contact in a previous campaign", "-1 means never contacted before"),
    "previous": ("Contacts before this campaign", None),
    "poutcome": ("Outcome of the previous campaign", "'unknown' if never contacted"),
}
GROUPS = [("Customer profile", ["age", "job", "marital", "education"]),
          ("Finances", ["default", "balance", "housing", "loan"]),
          ("This campaign", ["contact", "day", "month", "campaign"]),
          ("Previous campaigns", ["pdays", "previous", "poutcome"])]

st.set_page_config(page_title="Term Deposit Call Planner", layout="wide")


# ------------------------------------------------------------------ load the saved bundle
@st.cache_resource
def load_bundle(path):
    from xgboost import XGBClassifier
    b = joblib.load(path)
    # The XGBoost model is stored in its own file (xgb_model.ubj), not inside the pickle
    model_path = os.path.join(os.path.dirname(path), b.get("model_file", "xgb_model.ubj"))
    if "xgb_params" in b:
        model = XGBClassifier(**b["xgb_params"])
        model.load_model(model_path)
        b["pipeline"].set_params(model=model)
    return b


if not os.path.exists(BUNDLE_PATH):
    st.error("best_pipeline.joblib was not found next to app.py. Run the Part 3 notebook, "
             "download the app folder, and start the app from inside that folder.")
    st.stop()

bundle = load_bundle(BUNDLE_PATH)
THRESHOLD = bundle["threshold"]
BASE_RATE = bundle["base_rate"]
RATES = bundle["decile_expected_rate"]


def as_percent(df, cols):
    # display copy only: 0.286 -> "28.6%"
    out = df.copy()
    for c in cols:
        out[c] = out[c].map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
    return out


def options_for(col):
    opts = list(bundle["categories"][col])
    if col == "month":
        opts = [m for m in MONTHS if m in opts] + [m for m in opts if m not in MONTHS]
    return opts


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Model in use")
    st.write(f"**{bundle['model_name']}** (Member 3, tuned)")
    m = bundle["test_metrics"]
    c1, c2 = st.columns(2)
    c1.metric("Test F1", f"{m['F1']:.3f}")
    c2.metric("Test PR-AUC", f"{m['PR-AUC']:.3f}")
    c1.metric("Precision", f"{m['Precision']:.3f}")
    c2.metric("Recall", f"{m['Recall']:.3f}")
    st.caption(f"Decision threshold: **{THRESHOLD:.3f}** (tuned in Part 2; default would be 0.5).")
    st.caption(f"Average subscription rate: **{BASE_RATE:.1%}**.")
    st.caption("`duration` is never used: it is only known after the call.")
    trained_with = bundle.get("versions", {}).get("scikit-learn")
    if trained_with and trained_with != sklearn.__version__:
        st.warning(f"scikit-learn {sklearn.__version__} is installed but the pipeline was saved "
                   f"with {trained_with}. Install the version in requirements.txt.")

st.title("Term Deposit Call Planner")
st.caption("Who should the call centre phone first? Scores come from Member 3's tuned model; "
           "the call rule and priority groups come from Part 2.")
tab1, tab2, tab3 = st.tabs(["Single customer (ML lens)",
                            "Campaign queue (Analytical lens)",
                            "How it works"])

# ------------------------------------------------------------------ tab 1: single customer
with tab1:
    examples = bundle.get("examples", {})
    for c, v in bundle["defaults"].items():            # first run: start from a typical customer
        st.session_state.setdefault(f"in_{c}", v)

    def load_example():
        choice = st.session_state["example_choice"]
        row = bundle["defaults"] if choice == "Typical customer (training averages)" else examples[choice]
        for c, v in row.items():
            st.session_state[f"in_{c}"] = v

    st.selectbox("Start from", ["Typical customer (training averages)"] + list(examples),
                 key="example_choice", on_change=load_example,
                 help="The examples are real customers from the test set, which no model was trained on.")

    with st.form("customer_form"):
        cols = st.columns(4)
        for col, (title, fields) in zip(cols, GROUPS):
            with col:
                st.markdown(f"**{title}**")
                for f in fields:
                    label, help_text = FIELDS[f]
                    if f in bundle["categories"]:
                        st.selectbox(label, options_for(f), key=f"in_{f}", help=help_text)
                    else:
                        st.number_input(label, step=1, key=f"in_{f}", help=help_text)
        st.form_submit_button("Score this customer", type="primary")

    customer = pd.DataFrame([{c: st.session_state[f"in_{c}"] for c in features.RAW_COLUMNS}])
    result = features.score_customers(customer, bundle).iloc[0]
    d = int(result["priority_decile"])
    rate = result["expected_rate"]

    st.divider()
    r1, r2, r3 = st.columns(3)
    r1.metric("Recommendation", result["recommendation"],
              "score is above the call threshold" if result["recommendation"] == "CALL"
              else "score is below the call threshold", delta_color="off")
    r2.metric("Priority decile", f"{d} of 10", "1 = call first", delta_color="off")
    r3.metric("Expected subscription rate", f"{rate:.0%}",
              f"{rate / BASE_RATE:.1f}x the average customer", delta_color="off")
    st.caption(f"Model score {result['score']:.3f} against threshold {THRESHOLD:.3f}. "
               f"The expected rate is the share of training customers in decile {d} who really "
               f"subscribed; it is shown instead of the raw score, which class weighting inflates.")

    with st.expander("Why this result?"):
        if bundle["model_name"] == "Logistic Regression":
            clf = bundle["pipeline"].named_steps["model"]
            clean, _ = features.prepare_input(customer, bundle)
            x = np.asarray(bundle["pipeline"][:-1].transform(clean[features.RAW_COLUMNS]))[0]
            contrib = (pd.DataFrame({"feature": bundle["model_columns"], "effect": x * clf.coef_[0]})
                       .assign(size=lambda t: t.effect.abs())
                       .sort_values("size", ascending=False).head(8))
            contrib["direction"] = np.where(contrib.effect > 0, "raises the score", "lowers the score")
            st.write("Biggest influences on **this customer's** score:")
            st.dataframe(contrib[["feature", "direction", "effect"]].round(3), hide_index=True)
        elif bundle.get("top_drivers"):
            st.write(f"Features the {bundle['model_name']} relies on most "
                     "(permutation importance on the test set):")
            st.dataframe(pd.Series(bundle["top_drivers"], name="drop in PR-AUC when shuffled")
                         .rename_axis("feature").reset_index().round(4), hide_index=True)
        else:
            st.write("See the explainability section of the report (Part 1).")
        st.write(f"Decile {d} covers {features.decile_ranges(bundle['decile_edges']).loc[d - 1, 'score range']}.")

# ------------------------------------------------------------------ tab 2: campaign queue
with tab2:
    st.subheader("Build a calling queue")
    st.caption("Upload customers in the bank-full.csv format (comma or semicolon separated). "
               "Extra columns such as a customer ID are kept. `y` and `duration` are ignored.")

    left, right = st.columns([3, 1])
    upload = left.file_uploader("Customer file (CSV)", type=["csv"])
    if os.path.exists(DEMO_CUSTOMERS) and right.button("Use the demo customers"):
        st.session_state["use_demo"] = True
    if upload is not None:
        st.session_state["use_demo"] = False

    raw = None
    if upload is not None:
        raw = pd.read_csv(upload, sep=None, engine="python")
        source = upload.name
    elif st.session_state.get("use_demo"):
        raw = pd.read_csv(DEMO_CUSTOMERS)
        source = "demo_new_customers.csv (test-set customers, never seen in training)"

    queue = None
    if raw is None:
        st.info("Upload a CSV, or press **Use the demo customers**.")
    else:
        try:
            _, notes = features.prepare_input(raw, bundle)
            queue = features.score_customers(raw, bundle)
        except ValueError as err:
            st.error(str(err))
    if queue is not None:
        for n in notes:
            st.warning(n)
        st.success(f"Scored {len(queue):,} customers from {source}.")

        n_calls = st.slider("How many calls can the team make?", 1, len(queue),
                            max(1, int(round(len(queue) * 0.2))))
        top = queue.head(n_calls)
        expected = top["expected_rate"].sum()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Customers", f"{len(queue):,}")
        k2.metric("Marked CALL by the model", f"{(queue['recommendation'] == 'CALL').sum():,}")
        k3.metric(f"Expected subscribers in top {n_calls}", f"{expected:,.0f}",
                  f"random calling: {n_calls * BASE_RATE:,.0f}", delta_color="off")
        k4.metric("Expected hit rate", f"{expected / n_calls:.1%}",
                  f"random: {BASE_RATE:.1%}", delta_color="off")

        counts = (queue["priority_decile"].value_counts().reindex(range(1, 11), fill_value=0)
                  .rename("customers").rename_axis("priority decile"))
        st.bar_chart(counts, x_label="Priority decile (1 = call first)", y_label="Customers")

        show = (["rank"] + [c for c in queue.columns if c.lower() in ("customer_id", "id")] +
                ["recommendation", "priority_decile", "expected_rate", "score"] + features.RAW_COLUMNS)
        st.dataframe(as_percent(top[show].round(3), ["expected_rate"]), hide_index=True)
        st.download_button("Download the full queue (CSV)", queue[show].to_csv(index=False).encode(),
                           file_name="calling_queue.csv", mime="text/csv")

        with st.expander("Demo only: check the queue against what really happened"):
            st.caption("For a demonstration with test-set customers, whose real outcome is known. "
                       "The answer key must have customer_id and actually_subscribed columns.")
            key_file = st.file_uploader("Answer key (CSV)", type=["csv"], key="answer_key")
            if os.path.exists(DEMO_ANSWERS) and st.button("Use the demo answer key"):
                st.session_state["use_demo_answers"] = True
            answers = None
            if key_file is not None:
                answers = pd.read_csv(key_file)
            elif st.session_state.get("use_demo_answers"):
                answers = pd.read_csv(DEMO_ANSWERS)
            if answers is not None and "customer_id" in queue.columns:
                checked = queue.merge(answers[["customer_id", "actually_subscribed"]], on="customer_id")
                hits = int(checked.head(n_calls)["actually_subscribed"].sum())
                total = int(checked["actually_subscribed"].sum())
                a1, a2, a3 = st.columns(3)
                a1.metric(f"Real subscribers in top {n_calls}", f"{hits}",
                          f"expected {expected:.0f}", delta_color="off")
                a2.metric("Real hit rate", f"{hits / n_calls:.1%}",
                          f"random: {checked['actually_subscribed'].mean():.1%}", delta_color="off")
                a3.metric("Share of all subscribers reached", f"{hits / max(total, 1):.0%}",
                          f"with {n_calls / len(checked):.0%} of calls", delta_color="off")
                by_decile = (checked.groupby("priority_decile")
                             .agg(customers=("rank", "size"), real_rate=("actually_subscribed", "mean"),
                                  expected_rate=("expected_rate", "first")))
                st.dataframe(as_percent(by_decile, ["real_rate", "expected_rate"]))
            elif answers is not None:
                st.error("The uploaded customers have no customer_id column to match on.")

# ------------------------------------------------------------------ tab 3: how it works
with tab3:
    st.subheader("How every decision is made")
    st.markdown(f"""
1. **Prepare** the 15 raw columns with Member 2's feature rules and fitted pipeline
   (scaling, one-hot encoding and the same 30 selected columns). Nothing is refitted.
2. **Score** with Member 3's tuned **{bundle['model_name']}**: the "yes" column of `predict_proba`.
3. **Decide**: score at or above **{THRESHOLD:.3f}** means **CALL**. The threshold maximised F1
   on out-of-fold training predictions (Part 2).
4. **Prioritise**: the score is placed into one of 10 deciles using 9 cut points fixed in Part 2,
   and shown with that decile's real historical subscription rate.
""")
    table = features.decile_ranges(bundle["decile_edges"])
    table["expected rate (training)"] = table["decile"].map(RATES)
    if bundle.get("decile_table") is not None:
        test = bundle["decile_table"]
        table["real rate (test set)"] = table["decile"].map(test["conversion_rate"])
        table["lift vs random"] = table["decile"].map(test["lift"])
    if "lift vs random" in table.columns:
        table["lift vs random"] = table["lift vs random"].round(2)
    st.dataframe(as_percent(table, [c for c in table.columns if "rate" in c]), hide_index=True)
    st.caption("Lift = decile rate / overall rate. Lift 3 means three times better than calling at random.")