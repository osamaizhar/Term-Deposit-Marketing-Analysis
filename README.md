# Term-Deposit-Marketing-Analysis (7v2rUeFH7VqYe0Fb
)
Analysis for Term Deposit Marketing of a Client
\
````markdown
# Term-Deposit-Marketing-Analysis

> **Dial smarter, save budget, win customers.**  
> This repo shows—end-to-end—how we turn raw call-center logs into two production-ready ML pipelines that tell you **who to ring first** *and* **who to keep ringing**.

---

## 🚀 Project at a Glance
| | |
|---|---|
| **Problem** | Out of 41 k+ calls, only **≈7 %** end in a term-deposit sale. Every wrong number wastes money; every missed prospect wastes revenue. |
| **Solution** | Two stage models:<br>1. **Pre-Call Targeting** – predict the best prospects *before* anyone dials.<br>2. **Post-Call Prioritisation** – decide whether to chase, chill, or cut after the first conversation. |
| **Impact** | Lift recall 4× with SMOTE, slash futile calls, and keep your agents (and CFO) smiling. |

---

## 🏢 Background — Why We Built This
We’re a lean startup focused on *machine-learning magic* for European banks—fraud sniffers, sentiment sleuths, customer-intent whisperers, you name it.  
Our north-star: **boost call-center success rates** while keeping models interpretable enough for regulators and executives alike.

---

## 🗄️ Data Description — The Raw Material
* **Source:** A European bank’s direct-marketing phone campaign for term deposits.  
* **Call pattern:** One customer may get multiple calls until they say *yes*, *no*, or stop answering.  
* **Privacy:** All PII stripped; we keep only business-relevant attributes (age band, job type, previous outcome, etc.).  
* **Term deposits 101:** Think “short-term timed savings”—money locked for a month to a few years, withdrawn only when the term ends.  
* **Class imbalance:** Positive class ≈ 7 % → we attack it with **SMOTE** to level the playing field.

---

## 🗂️ Repo Structure
| Path | Contents |
|------|----------|
| `exploratory_data_analysis.ipynb` | Visual, comment-rich EDA notebook |
| `prediction_model_training_final.ipynb` | All feature-engineering + Model 1 & Model 2 training |
| `result_imgs/` | Confusion matrices, ROC curves, feature-importances—ready for slides |
| `term-deposit-marketing-2020.csv` | Trimmed dataset so you can hack offline |
| `requirements.txt` | Pinned deps (Python 3.10-friendly) |

---

## 🏗️ Pipeline in Mini-Episodes

1. **EDA & Feature Prep**  
   * One-hot categoricals, drop high-cardinality junk  
   * Correlation heat-map finds gold nuggets («month», «contact_type», «poutcome»)

2. **Imbalance Buster**  
   * `imblearn.SMOTE` = recall hero, accuracy guardian

3. **Model 1 — “Should We Call At All?”**  
   * Only pre-campaign features (no leakage)  
   * Logistic Reg baseline → k-NN → XGBoost (best overall F1)

4. **Model 2 — “Should We Call Again?”**  
   * Adds call metrics (duration, day, campaign)  
   * Extra context bumps recall > 0.35 with modest precision dip

5. **Explainability**  
   * SHAP plots & plain-English feature rankings—because auditors love receipts

---

## 🔧 Quick Install

```bash
# one-liner—clone, enter, install
git clone https://github.com/osamaizhar/Term-Deposit-Marketing-Analysis.git && \
cd Term-Deposit-Marketing-Analysis && \
pip install -r requirements.txt
````

---

## ▶️ Reproduce Results

```bash
# 1. Fire up Jupyter
jupyter notebook exploratory_data_analysis.ipynb

# 2. Train & evaluate
jupyter notebook prediction_model_training_final.ipynb

# 3. Admire the plots
xdg-open result_imgs/
```

Prefer scripts?

```bash
jupyter nbconvert --to script *.ipynb && python exploratory_data_analysis.py
```

---

## 📈 Key Takeaways

* **Previous outcome is king** – a past “success” boosts win-rate \~7×.
* **Seasonal sweet spots** – May and November spike conversions.
* **Balanced beats raw accuracy** – SMOTE quadruples true-positive capture without wrecking precision.
* **ROI metric matters** – we optimise “calls saved per extra subscriber,” not just F1 score.

---

## 🤝 Contributing

1. Fork → `git checkout -b feat/<your-idea>`
2. Commit (Conventional Commits welcome)
3. PR to **main**—CI will lint & test automatically.
4. Grab coffee; we’ll review quickly. ☕

---

## 🪪 License

MIT—do great things, just give credit.

---

## 👤 Author & Links

**Osama Izhar**

* GitHub: [https://github.com/osamaizhar](https://github.com/osamaizhar)
* LinkedIn: [https://www.linkedin.com/in/osamaizhar-b4727116a/](https://www.linkedin.com/in/osamaizhar-b4727116a/)
* Apziva AI Residency: [https://www.apziva.com/](https://www.apziva.com/)

```

