# Term Deposit Call Planner: implementation guide

Member 4 (IT24100001), IT3091 Bank Marketing project.

This folder is the working app. It scores bank customers with **Member 3's tuned model**, applies the **decision rules from Part 2** (call threshold and priority deciles), and shows the result in a web page with two lenses:

| Tab | Lens | What it does |
|---|---|---|
| Single customer | ML lens | Enter one customer, get CALL or not, a priority decile and an expected subscription rate |
| Campaign queue | Analytical lens | Upload a CSV of customers, get a ranked calling queue and the effect of a call budget |
| How it works | Both | The rules behind every decision, for the presentation |

Nothing is trained in the app. It only loads saved objects.

---

## 1. What is in this folder

| File | What it is | Created by |
|---|---|---|
| `app.py` | The web app (Streamlit) | Part 3 notebook |
| `features.py` | Member 2's feature rules and the scoring helpers. **Must stay next to `app.py`** | Part 3 notebook |
| `best_pipeline.joblib` | Member 2's fitted feature pipeline + Member 3's model + Part 2 rules, in one file | Part 3 notebook |
| `xgb_model.ubj` | Model file exported from the Part 3 notebook; **must stay next to `app.py` and `best_pipeline.joblib`** | Part 3 notebook
| `requirements.txt` | Exact library versions used in Colab | Part 3 notebook |
| `demo_new_customers.csv` | 200 real test-set customers, outcome removed (never seen by any model) | Part 3 notebook |
| `demo_answer_key.csv` | What those 200 customers really did, to reveal after the demo | Part 3 notebook |
| `README.md` | This guide | Part 3 notebook |

---

## 2. Before you start

You need these in your `IT24100001` Drive folder, from earlier parts:

- `IT24100001_part2_threshold_deciles.joblib`, saved by the last cell of your Part 2

Plus the Member 1, 2 and 3 files that Part 2 already used.

All notebooks expect this layout in **your My Drive**:

```
My Drive/
└── ML & DWBI Assignment/
    └── ML Assignment/
        ├── IT24102060/   (Member 1)
        ├── IT24101461/   (Member 2)
        ├── IT24102228/   (Member 3)
        └── IT24100001/   (you: Part 1, 2, 3 outputs and the app folder)
```

---

## 3. Step 1: Run the Part 3 notebook in Colab

1. Upload `IT24100001_Part3.ipynb` to Colab.
2. **Runtime → Run all.** Allow Google Drive access when asked.
3. Check these outputs as it runs:

| Section | What you must see |
|---|---|
| 1 | Every required file listed as `OK` or `OK (copy)` |
| 3 | `PASS` for: features.py reproduces Member 2, chained pipeline equals Member 3's model, deciles match Part 2 |
| 6 | `best_pipeline.joblib` saved, and the reload test says `PASS` |
| 7 | The `IT24100001/app` folder listing shows all 7 files |
| 9 | `CONFIRMED: nothing was written to the Member 1, 2 or 3 folders` |

If a check fails, nothing is saved and the message says why.

4. Note the **Python version** printed at the top of `requirements.txt` (section 7 shows it). You need the same version on your laptop.

---

## 4. Step 2: Download the app folder

1. In Google Drive, open `ML & DWBI Assignment / ML Assignment / IT24100001`.
2. Right-click the **`app`** folder → **Download**. Drive gives you a `.zip`.
3. Unzip it somewhere simple, for example `Documents/call_planner`.

Check that all 7 files from section 1 are inside the same folder.

---

## 5. Step 3: Install Python on your laptop

1. Check whether Python is installed. Open a terminal (Windows: **Command Prompt**; macOS: **Terminal**) and type:
   ```
   python --version
   ```
   On macOS use `python3 --version`.
2. You need the **same major.minor version as Colab** (for example 3.12), shown at the top of `requirements.txt`. If it is missing or different, install it from python.org. On Windows, tick **"Add python.exe to PATH"** during installation.

Why the same version: `best_pipeline.joblib` is a saved Python object. Loading it with very different library versions can fail.

---

## 6. Step 4: Create a virtual environment and install the libraries

A virtual environment keeps these exact versions separate from anything else on your laptop.

**Windows (Command Prompt):**
```
cd C:\Users\<you>\Documents\call_planner
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux (Terminal):**
```
cd ~/Documents/call_planner
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

When the environment is active, the prompt starts with `(.venv)`. Every time you open a new terminal, `cd` into the folder and run the `activate` line again before starting the app.

---

## 7. Step 5: Start the app

With the environment active, inside the folder:
```
streamlit run app.py
```

Your browser opens at **http://localhost:8501**. If it does not, open that address yourself.

To stop the app, go back to the terminal and press **Ctrl + C**.

The first time, Streamlit may ask for an email address. Just press Enter to skip.

---

## 8. Step 6: Demonstration script

### Tab 1: Single customer (ML lens)

1. In **Start from**, choose **"Real test customer: strongest prospect"**. These are real customers from the test set, which no model was trained on.
2. Press **Score this customer**. Point out:
   - **Recommendation: CALL**, because the score is above the tuned threshold.
   - **Priority decile 1 or 2**: call first.
   - **Expected subscription rate**: the share of similar training customers who really subscribed.
3. Now change some inputs by hand to make a weak prospect, for example: `contact` = unknown, `month` = may, `campaign` = 8, `housing` = yes, `poutcome` = unknown, `pdays` = -1. Press **Score this customer** again. The decile drops and the recommendation becomes **Lower-cost channel**.
4. Open **Why this result?** to show what drives the model.

What to say: *"The score comes from Member 3's tuned model. The call line and the priority groups come from our Part 2 rules, which were chosen on training data only."*

### Tab 2: Campaign queue (Analytical lens)

1. Press **Use the demo customers** (200 real test customers, outcome hidden).
2. Move the **"How many calls can the team make?"** slider. Show **expected subscribers** against **random calling**.
3. Show the decile bar chart and the ranked table. Press **Download the full queue** to show the deliverable a call centre would receive.
4. Open **"Demo only: check the queue against what really happened"** and press **Use the demo answer key**. This reveals how many of the top-ranked customers **really** subscribed, which proves the ranking works on unseen customers.

What to say: *"With the same number of calls, following the queue reaches far more subscribers than calling at random."*

### Tab 3: How it works

Walk through the four steps and the decile table: score range, expected rate from training, real rate on the test set, and lift.

---

## 9. Screenshots for the report

1. Tab 1 with a strong prospect (CALL, decile 1 or 2).
2. Tab 1 with a weak prospect (lower-cost channel, low decile).
3. Tab 2 with the demo customers loaded and the slider set (metrics and ranked table visible).
4. Tab 2 with the answer key revealed.
5. Tab 3 decile table.

Windows: **Win + Shift + S**. macOS: **Cmd + Shift + 4**.

---

## 10. Using your own new customer data

The CSV needs a header row with these 15 columns (any order; extra columns such as `customer_id` are kept in the output):

```
age,job,marital,education,default,balance,housing,loan,contact,day,month,campaign,pdays,previous,poutcome
35,management,married,tertiary,no,1500,yes,no,cellular,15,may,2,-1,0,unknown
```

Comma or semicolon separated both work, so the original `bank-full.csv` format is accepted. `y` and `duration` are ignored if present.

Allowed values:

| Column | Values |
|---|---|
| `job` | admin., blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown |
| `marital` | married, single, divorced |
| `education` | primary, secondary, tertiary, unknown |
| `default`, `housing`, `loan` | yes, no |
| `contact` | cellular, telephone, unknown |
| `month` | jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec |
| `poutcome` | success, failure, other, unknown |
| `pdays` | days since last contact in a previous campaign; **-1 means never contacted** |

Text is not case sensitive. A value never seen in training is accepted but carries no information, and the app shows a warning.

**Do not use `bank.csv` from the UCI website as "new" data.** It is a random 10% sample of `bank-full.csv`, so most of its rows were used for training. The demo file in this folder is the honest choice.

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'features'` | Start the app from inside the folder, and keep `features.py` next to `app.py` |
| `best_pipeline.joblib was not found` | Same as above: run `streamlit run app.py` from the folder that contains it |
| Errors mentioning `_RemainderColsList`, `force_int_remainder_cols` or unpickling | Library versions differ from Colab. Activate the virtual environment and run `pip install -r requirements.txt` again |
| Sidebar warning about the scikit-learn version | Same fix as above |
| `pip` cannot find a version in `requirements.txt` | Your Python version differs from Colab's. Install the version at the top of `requirements.txt` |
| macOS: `lightgbm` fails with `libomp` | Run `brew install libomp`, then reinstall |
| Windows PowerShell blocks `activate` | Use Command Prompt instead, or run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `streamlit` is not recognised | The virtual environment is not active. Run the `activate` line again |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| "Missing required columns" after upload | The CSV header must contain all 15 column names in section 10 |

---

## 12. How it works (for the viva)

For every customer, the app does four things. None of them learns from the new data.

1. **Prepare**: Member 2's 8 feature rules, then Member 2's fitted pipeline (`transform` only: scaling with training averages, one-hot encoding with training categories, the same 30 selected columns).
2. **Score**: Member 3's tuned model, `predict_proba`, "yes" column.
3. **Decide**: score at or above the tuned threshold means **CALL**. The threshold maximised F1 on out-of-fold training predictions.
4. **Prioritise**: the score is compared with 9 fixed cut points to give a decile from 1 to 10, shown with that decile's real subscription rate from training.

The threshold, cut points and expected rates were fixed in Part 2, so the same customer always gets the same answer, whoever else is uploaded with them.
