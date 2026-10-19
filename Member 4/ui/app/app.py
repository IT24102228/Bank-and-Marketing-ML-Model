"""
Term Deposit Call Planner  (Member 4, IT24100001)
Dual-lens decision interface for the Bank Marketing project.

  Tab 1  Single customer   (ML lens)          score one customer and explain the decision
  Tab 2  Campaign queue    (Analytical lens)  rank a CSV of customers into a calling queue
  Tab 3  How it works                         the rules behind every decision
  Tab 4  How we built it                      the data, training and results behind the model

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
    st.write(f"**{bundle['model_name']}**")
    st.caption("A machine learning model that scores how likely a customer is to subscribe to a "
               "term deposit after a phone call. Results on customers it had never seen:")
    m = bundle["test_metrics"]
    c1, c2 = st.columns(2)
    c1.metric("Test F1", f"{m['F1']:.3f}",
              help="One score from 0 to 1 that balances precision and recall. Higher is better.")
    c2.metric("Test PR-AUC", f"{m['PR-AUC']:.3f}",
              help=f"How well the model ranks real subscribers above everyone else. Higher is better. "
                   f"Random guessing would score about {BASE_RATE:.2f}.")
    c1.metric("Precision", f"{m['Precision']:.3f}",
              help="Of the customers marked CALL, the share who really subscribed.")
    c2.metric("Recall", f"{m['Recall']:.3f}",
              help="Of all customers who would subscribe, the share the model marked CALL.")
    st.caption(f"Call threshold: **{THRESHOLD:.3f}**. A score at or above it means CALL "
               f"(the usual default would be 0.5).")
    st.caption(f"Average subscription rate: **{BASE_RATE:.1%}**, the share of all customers who said yes.")

    with st.expander("What data was it trained on?"):
        st.write(f"The public **UCI Bank Marketing** dataset: 45,211 records of a bank phoning "
                 f"customers to sell a **term deposit** (a savings product where the money is locked "
                 f"for a fixed time). Each row is one contact, with the customer's details and whether "
                 f"they subscribed. About {BASE_RATE:.0%} did. 80% of the rows were used to train the "
                 f"model and 20% were held back to test it.")
    with st.expander("What does the model use?"):
        st.write("Fifteen details about each customer:")
        for title, cols in GROUPS:
            st.markdown(f"**{title}:** " + ", ".join(FIELDS[c][0].lower() for c in cols))
        st.caption("It also builds 8 extra features from these, such as whether the customer was "
                   "contacted in an earlier campaign.")
    with st.expander("What is `duration`, and why isn't it used?"):
        st.write("`duration` is how long the last phone call lasted. It is in the original data, but "
                 "this model never uses it. It is only known **after** the call ends, and longer calls "
                 "usually end in a sale, so using it would make the model look better than it is while "
                 "being useless for deciding whom to call. Using information that would not be "
                 "available at decision time is called *data leakage*.")
    with st.expander("What do the scores mean?"):
        st.markdown("- **Precision:** how many CALL customers really subscribe.\n"
                    "- **Recall:** how many real subscribers the model finds.\n"
                    "- **F1:** balances the two.\n"
                    "- **PR-AUC:** how well it ranks likely subscribers first.\n"
                    "- **Threshold:** the score at which the answer changes to CALL.")
    trained_with = bundle.get("versions", {}).get("scikit-learn")
    if trained_with and trained_with != sklearn.__version__:
        st.warning(f"scikit-learn {sklearn.__version__} is installed but the pipeline was saved "
                   f"with {trained_with}. Install the version in requirements.txt.")

st.title("Term Deposit Call Planner")
st.caption("Who should the call centre phone first? Scores come from Member 3's tuned model; "
           "the call rule and priority groups come from Part 2.")
# ------------------------------------------------------------------ intro for new visitors
with st.expander("New here? What is this app for?", expanded=True):
    st.markdown(
        f"A bank's call centre cannot phone every customer. Each call costs agent time, and only "
        f"about **{BASE_RATE:.0%}** of customers say yes to a term deposit. This app scores each "
        f"customer on how likely they are to subscribe and says **CALL** or **Lower-cost channel**, so agents "
        f"phone the most promising people first."
    )
    w1, w2, w3 = st.columns(3)
    with w1:
        st.markdown("**The question**")
        st.write("Should we call this customer about a term deposit, and how high up the list "
                 "should they be?")
    with w2:
        st.markdown("**How it decides**")
        st.write("A machine learning model reads the customer's profile, finances and past "
                 "campaign history, and gives a score from 0 to 1. Call length is never used, "
                 "because it is only known after the call.")
    with w3:
        st.markdown("**What you get**")
        st.write(f"**CALL** if the score is at least {THRESHOLD:.3f}, otherwise **Lower-cost channel** (a cheaper way to reach them than a phone call), plus a "
                 "priority group from 1 (best 10% of customers) to 10.")
    st.markdown(
        "**Try it:** open *Single customer* and pick an example, or use *Campaign queue* to rank "
        "a whole list. *How it works* explains every step."
    )

tab1, tab2, tab3, tab4 = st.tabs(["Single customer (ML lens)",
                                  "Campaign queue (Analytical lens)",
                                  "How it works",
                                  "How we built it"])

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
    call = result["recommendation"] == "CALL"
    times = rate / BASE_RATE
    in100 = round(rate * 100)
    st.subheader("Result")
    if call:
        st.success(f"**Call this customer.** In past campaigns, about {in100} in every 100 customers "
                   f"with a profile like this subscribed to a term deposit. That is {times:.1f}x the "
                   f"average ({BASE_RATE:.0%}).")
    else:
        st.warning(f"**Use a lower-cost channel for this customer.** In past campaigns, about {in100} in every 100 "
                   f"customers with a profile like this subscribed ({times:.1f}x the average of "
                   f"{BASE_RATE:.0%}). The model rates this customer below the level worth a phone call, "
                   f"so phone time is better spent on higher-priority customers.")

    r1, r2, r3 = st.columns(3)
    r1.metric("Recommendation", result["recommendation"],
              "score is high enough to call" if call else "score is too low to call",
              delta_color="off",
              help="CALL means the model's score is at or above the call threshold. "
                   "Lower-cost channel means it is below it.")
    r2.metric("Priority group (decile)", f"{d} of 10", "1 = call first", delta_color="off",
              help="Customers are ranked into 10 equal groups. Group 1 is the most likely to "
                   "subscribe and group 10 the least. If you cannot call everyone, work from "
                   "group 1 downwards.")
    r3.metric("Chance of subscribing", f"{rate:.0%}", f"{times:.1f}x the average customer",
              delta_color="off",
              help=f"The share of past customers in priority group {d} who really subscribed. "
                   "It is a group average, not a promise about this one person.")

    with st.expander("Technical details: score and threshold"):
        st.write(f"Model score **{result['score']:.3f}** against call threshold **{THRESHOLD:.3f}**. "
                 f"The chance of subscribing is the share of training customers in decile {d} who "
                 f"really subscribed. It is shown instead of the raw score, because a raw model score "
                 f"is not a true probability.")

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
    st.write("Rank a whole list of customers into a calling queue. Upload your own file, or try the "
             "demo customers if you have no data.")

    RAW = list(features.RAW_COLUMNS)

    def template_csv():
        # one typical customer, plus one real demo customer when the demo file is available
        rows = [{"customer_id": "example_1", **{c: bundle["defaults"][c] for c in RAW}}]
        if os.path.exists(DEMO_CUSTOMERS):
            demo = pd.read_csv(DEMO_CUSTOMERS)
            if all(c in demo.columns for c in RAW):
                rows.append({"customer_id": "example_2", **{c: demo.iloc[0][c] for c in RAW}})
        return pd.DataFrame(rows).to_csv(index=False).encode()

    with st.expander("What does my CSV need?", expanded=True):
        st.markdown(
            f"Your file needs **all {len(RAW)} columns below**, with these exact names (lowercase, as in "
            "the original `bank-full.csv`). Commas or semicolons both work. Extra columns such as "
            "`customer_id` are kept and shown in the queue. `y` and `duration` are ignored.")
        rows = []
        for c in RAW:
            label, hint = FIELDS.get(c, (c, None))
            allowed = (", ".join(str(o) for o in options_for(c))
                       if c in bundle["categories"] else "a number")
            rows.append({"Column": c, "Meaning": label + (f" ({hint})" if hint else ""),
                         "Allowed values": allowed})
        st.table(pd.DataFrame(rows).set_index("Column"))
        st.download_button("Download a template CSV", template_csv(),
                           file_name="customer_template.csv", mime="text/csv",
                           help="A ready-made file with the right column names and two example "
                                "customers. Replace the examples with your own.")

    n_demo = len(pd.read_csv(DEMO_CUSTOMERS)) if os.path.exists(DEMO_CUSTOMERS) else 0
    left, right = st.columns([2, 1])
    upload = left.file_uploader("Customer file (CSV)", type=["csv"])
    if n_demo:
        if right.button(f"No data? Use the {n_demo} demo customers"):
            st.session_state["use_demo"] = True
        right.caption("Real customers from the test set, which the model never saw during training.")
        right.download_button("Download the demo CSV", open(DEMO_CUSTOMERS, "rb").read(),
                              file_name="demo_new_customers.csv", mime="text/csv")
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
        st.info("Upload a CSV with the columns listed above, or press **No data? Use the demo customers**.")
    else:
        present = {str(c).strip().lower() for c in raw.columns}
        missing = [c for c in RAW if c.lower() not in present]
        if missing:
            st.error(f"Your file is missing {len(missing)} required column(s): "
                     + ", ".join(f"`{c}`" + ("" if FIELDS.get(c, (c,))[0].lower() == c
                                      else f" ({FIELDS[c][0].lower()})") for c in missing)
                     + ". Add them (see the box above), or use the demo customers instead.")
        else:
            if upload is not None:
                st.success(f"Your file has all {len(RAW)} required columns.")
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
    st.write("Four quick steps happen every time a customer is scored.")
    h1, h2, h3, h4 = st.columns(4)
    h1.markdown("**1. Read**")
    h1.write("The model reads the customer's 15 details.")
    h2.markdown("**2. Score**")
    h2.write("It gives a score from 0 to 1. Higher means more likely to subscribe.")
    h3.markdown("**3. Decide**")
    h3.write(f"A score of {THRESHOLD:.3f} or more means **CALL**. Below that, use a **lower-cost channel**.")
    h4.markdown("**4. Prioritise**")
    h4.write("The score puts the customer in one of 10 groups. Group 1 is called first.")
    st.caption("Nothing is retrained when you score a customer, so the same customer always gets "
               "the same answer.")

    st.markdown("#### The 10 priority groups")
    st.write("Each group shows how often its customers really subscribed.")
    table = features.decile_ranges(bundle["decile_edges"])
    table["expected rate (training)"] = table["decile"].map(RATES)
    if bundle.get("decile_table") is not None:
        test = bundle["decile_table"]
        table["real rate (test set)"] = table["decile"].map(test["conversion_rate"])
        table["lift vs random"] = table["decile"].map(test["lift"])
    if "lift vs random" in table.columns:
        table["lift vs random"] = table["lift vs random"].round(2)
    st.dataframe(as_percent(table, [c for c in table.columns if "rate" in c]), hide_index=True)
    st.caption("**Expected rate:** from the training data. **Real rate:** on the test set, which the model "
               "never saw. **Lift:** the real rate compared with the overall average, so 3 means three "
               "times better than calling at random.")

# ------------------------------------------------------------------ tab 4: how we built it
with tab4:
    m = bundle["test_metrics"]
    st.subheader("How this model was built")
    st.write("This page explains the data, the steps, the training and the results behind the "
             "recommendations, so you can judge how far to trust them.")

    st.markdown("#### 1. The data")
    st.write("The data is the public **UCI Bank Marketing** dataset: records of a bank phoning customers "
             "to sell a term deposit, with the outcome of each call. Each row is one customer contact.")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Customer contacts", "45,211")
    d2.metric("Input columns", "15")
    d3.metric("Subscribed", f"{BASE_RATE:.1%}")
    d4.metric("Training / test rows", "36,168 / 9,043")
    st.bar_chart(pd.DataFrame({"Share of customers": [1 - BASE_RATE, BASE_RATE]},
                              index=["Did not subscribe", "Subscribed"]),
                 y_label="Share of customers")
    st.caption("Only a small share say yes, so plain accuracy would be misleading: a model that always "
               "answers 'no' would be right most of the time. We judge the model with precision, recall, "
               "F1 and PR-AUC instead. Call length (`duration`) is never used, because it is only "
               "known after the call has happened. Answers recorded as 'unknown' were kept as their own "
               "category, because not knowing can itself be informative.")

    st.markdown("#### 2. From raw data to a score")
    st.markdown("""
| Step | What happens | Done by |
|---|---|---|
| 1. Clean and split | Audit the data, remove call length, encode the answer, split 80% training / 20% test (same mix of yes and no in both) | Member 1 |
| 2. Create features | Build 8 new features from the raw columns | Member 2 |
| 3. Prepare | Scale the numbers and one-hot encode the categories | Member 2 |
| 4. Select | Keep the 30 most informative columns (mutual information) | Member 2 |
| 5. Train and tune | Train four models, tune them with 5-fold cross-validation | Member 3 |
| 6. Decide | Pick the final model, set the call threshold and the 10 priority groups | Member 4 |
| 7. Serve | Package everything into this app | Member 4 |
""")
    st.bar_chart(pd.DataFrame({"Columns": [15, 23, 68, 30]},
                              index=["1 Raw", "2 With new features", "3 Scaled and encoded",
                                     "4 Selected"]),
                 x_label="Stage", y_label="Number of columns")
    with st.expander("The 8 features we created"):
        st.dataframe(pd.DataFrame([
            ("previously_contacted", "1 if the customer was contacted in an earlier campaign", "Makes 'never contacted' explicit"),
            ("pdays_clean", "days since last contact, with -1 replaced by 0", "Removes the misleading -1"),
            ("loan_count", "1 point each for a housing loan and a personal loan (0 to 2)", "Summarises loan burden"),
            ("has_credit_default", "1 if the customer has credit in default", "Repayment status"),
            ("previous_campaign_success", "1 if the previous campaign outcome was 'success'", "Flags past responders"),
            ("age_group", "18-30, 31-40, 41-50, 51-60, 60+", "Captures non-linear age effects"),
            ("balance_group", "negative, zero, low, medium, high", "Groups balances into categories"),
            ("campaign_contact_group", "first, second, three to four, five or more contacts", "How much contact effort was spent"),
        ], columns=["Feature", "Rule", "Why"]), hide_index=True)

    st.markdown("#### 3. How the model was trained")
    st.markdown(f"""
- **Four candidates:** logistic regression, random forest, XGBoost and LightGBM. Each was scored with stratified 5-fold cross-validation on the training data.
- **Tuning:** each model was tuned with a randomized search that tried different settings and kept the ones with the best F1. XGBoost, for example, tried 15 combinations of trees, depth, learning rate and sampling.
- **Fair testing:** the scaler, encoder, feature selector and models were fitted on training data only. The test set was used once, to report results. A fixed random seed (42) makes every step repeatable.
- **Nothing is retrained here.** This app only loads the saved pipeline and model.
""")
    st.markdown("**Why XGBoost?** Judged at the usual 0.5 cutoff, random forest scored best on F1 and XGBoost "
                "worst. But F1 at a fixed cutoff mostly shows how well that cutoff happens to fit. XGBoost "
                "had the best ROC-AUC on the test set (0.789) and a PR-AUC (0.432) within 0.001 of the "
                "best, so it ranks customers well. In Part 2, Member 4 tuned the cutoff on out-of-fold "
                "training predictions and picked the model with the highest F1 at its tuned cutoff: "
                f"**{bundle['model_name']}**.")
    st.bar_chart(pd.DataFrame({"Cross-validated F1": [0.4195, 0.3664, 0.4224, 0.3297]},
                              index=["LightGBM", "Logistic Regression", "Random Forest", "XGBoost"]),
                 y_label="F1 at the default 0.5 cutoff")
    st.caption("Tuned models, 5-fold cross-validation on the training data, before the cutoff was tuned (Member 3).")
    params = {k: str(v) for k, v in bundle.get("xgb_params", {}).items() if v is not None}
    if params:
        with st.expander("Final model settings"):
            st.dataframe(pd.DataFrame({"Setting": list(params), "Value": list(params.values())}),
                         hide_index=True)

    st.markdown("#### 4. Choosing the call rule")
    st.write(f"A model gives each customer a score. The **threshold** decides which scores mean CALL. "
             f"The usual default of 0.5 suits balanced data, but here it misses too many buyers. We "
             f"tuned the threshold to **{THRESHOLD:.3f}** using training predictions only.")
    st.bar_chart(pd.DataFrame({"F1 on the test set": [0.329, m["F1"]]},
                              index=["Default threshold 0.5", f"Tuned threshold {THRESHOLD:.3f}"]),
                 y_label="F1 on the test set")

    st.markdown("#### 5. How well it works")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Precision", f"{m['Precision']:.3f}", help="Of the customers marked CALL, the share who really subscribed.")
    q2.metric("Recall", f"{m['Recall']:.3f}", help="Of all customers who would subscribe, the share the model marked CALL.")
    q3.metric("F1", f"{m['F1']:.3f}", help="One number balancing precision and recall.")
    q4.metric("PR-AUC", f"{m['PR-AUC']:.3f}", help="How well the model ranks real subscribers above the rest.")
    dt = bundle.get("decile_table")
    if dt is not None:
        st.bar_chart(dt["conversion_rate"].rename("Real subscription rate"),
                     x_label="Priority decile (1 = call first)", y_label="Share who subscribed")
        st.caption(f"On the test set, decile 1 customers subscribed at {dt['conversion_rate'].max():.0%}, "
                   f"{dt['lift'].max():.1f}x the overall average. That ranking is why a team with a limited "
                   f"call budget should work from decile 1 downwards.")

    drivers = bundle.get("top_drivers")
    if drivers:
        st.markdown("#### 6. What the model pays attention to")
        top = pd.Series(drivers).sort_values(ascending=False).head(8)
        top.index = [f"{i + 1}. {name}" for i, name in enumerate(top.index)]
        st.bar_chart(top.rename("Importance"), horizontal=True,
                     x_label="Drop in PR-AUC when the feature is shuffled")
        st.caption("This shows which inputs the model relies on, not what causes a customer to subscribe.")

    st.markdown("#### 7. Limits to keep in mind")
    st.markdown(f"""
- Precision is about **{m['Precision']:.0%}**, so more than half of the CALL customers still say no. Recall is about **{m['Recall']:.0%}**, so the model also misses about half of the people who would subscribe.
- The data comes from one bank's past phone campaigns. It may not carry over to other banks, markets or years.
- The model supports a decision. It does not replace the judgement of the people running the campaign.
""")
    st.caption("Full notebooks, reports and code: github.com/IT24102228/Bank-and-Marketing-ML-Model")