# 7v2rUeFH7VqYe0Fb

# Term-Deposit-Marketing-Analysis

> **Dial smarter, save budget, win customers.**  
> This repo shows—end-to-end—how we turn raw call-center logs into two production-ready ML pipelines that tell you **who to ring first** *and* **who to keep ringing**.

## 💡 Executive Summary

**The Bottom Line:** We built a two-stage ML system that captures 91% of potential term deposit subscribers while focusing your marketing resources on the highest-value segments: **retirees, high-balance customers, and the 60+ age group**. By prioritizing call quality over quantity (longer conversations = higher conversions) and limiting contacts to 2-3 per customer, you can maximize ROI and minimize wasted effort.

---

## 🚀 Project at a Glance

| | |
|---|---|
| **Problem** | Out of 41k+ calls, only **≈7.2%** end in a term-deposit sale. Every wrong number wastes money; every missed prospect wastes revenue. |
| **Solution** | Two stage models:<br>1. **Pre-Call Targeting** – predict the best prospects *before* anyone dials (62% recall).<br>2. **Post-Call Prioritisation** – decide whether to chase, chill, or cut after the first conversation (91% recall). |
| **Impact** | Lift recall 4× with SMOTE, slash futile calls by targeting retirees & high-balance customers first, and keep your agents (and CFO) smiling. |
| **Key Insight** | **Call duration is king** – longer conversations = higher conversions. Focus on quality engagement over quantity. |

---

## 🏢 Background — Why We Built This

We're a lean startup focused on *machine-learning magic* for European banks—fraud sniffers, sentiment sleuths, customer-intent whisperers, you name it.  
Our north-star: **boost call-center success rates** while keeping models interpretable enough for regulators and executives alike.

---

## 🗄️ Data Description — The Raw Material

* **Source:** A European bank's direct-marketing phone campaign for term deposits.  
* **Call pattern:** One customer may get multiple calls until they say *yes*, *no*, or stop answering.  
* **Privacy:** All PII stripped; we keep only business-relevant attributes (age band, job type, previous outcome, etc.).  
* **Term deposits 101:** Think "short-term timed savings"—money locked for a month to a few years, withdrawn only when the term ends.  
* **Class imbalance:** Positive class ≈ 7 % → we attack it with **SMOTE** to level the playing field.

---

## 🗂️ Repo Structure

| Path | Contents |
|------|----------|
| `exploratory_data_analysis.ipynb` | Visual, comment-rich EDA notebook |
| `prediction_model_training_final.ipynb` | All feature-engineering + Model 1 & Model 2 training |
| `term-deposit-marketing-2020.csv` | Dataset so you can hack offline |
| `requirements.txt` | Pinned deps (Python 3.10-friendly) |

---

## 🏗️ Pipeline in Mini-Episodes

1. **EDA & Feature Prep**  
   * One-hot categoricals, drop high-cardinality junk  
   * Correlation heat-map finds gold nuggets («month», «contact_type», «poutcome»)
   * Discovered key segments: retirees, high-balance, 60+ age group show highest conversion rates

2. **Imbalance Buster**  
   * `imblearn.SMOTE` = recall hero, accuracy guardian
   * Tackles the 7.2% subscription rate challenge

3. **Model 1 — "Should We Call At All?"**  
   * Only pre-campaign features (no leakage)  
   * Logistic Reg baseline → k-NN → XGBoost (best overall F1)
   * Achieves 62% recall, capturing most potential subscribers before any contact

4. **Model 2 — "Should We Call Again?"**  
   * Adds call metrics (duration, day, campaign)  
   * Extra context bumps recall to 91% with F1 score up to 0.48
   * Call duration emerges as the strongest predictor

5. **Customer Segmentation**
   * KMeans clustering reveals 94% typical subscribers + 6% high-value niche segment
   * Enables targeted, personalized marketing strategies

6. **Explainability**  
   * SHAP plots & plain-English feature rankings—because auditors love receipts
   * Clear business recommendations backed by data

---

## 🔧 Quick Install

```bash
# one-liner—clone, enter, install
git clone https://github.com/osamaizhar/Term-Deposit-Marketing-Analysis.git && \
cd Term-Deposit-Marketing-Analysis && \
pip install -r requirements.txt
```

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

* **Previous outcome is king** – a past "success" boosts win-rate ~7×.
* **Seasonal sweet spots** – May and November spike conversions.
* **Balanced beats raw accuracy** – SMOTE quadruples true-positive capture without wrecking precision.
* **ROI metric matters** – we optimise "calls saved per extra subscriber," not just F1 score.

---

## 📊 Final Results & Business Recommendations

### Summary of Achievements

- **Comprehensive ML Pipeline:** Built a robust two-stage prediction system (Pre-Call and Post-Call models) for term deposit marketing, with deep data exploration, class imbalance handling, and customer segmentation.
- **Data Quality:** 40,000 customers, 14 features, no missing values.
- **Class Imbalance Addressed:** Only ~7.2% subscribe, so advanced resampling and recall-focused optimization were used.

### Model Achievements & What They Predict

#### 1️⃣ Pre-Call Model (Model 1) — *Who to Call Before Any Campaign Contact*

- **What it predicts:** Uses only demographic and financial data (age, job, balance, marital status, etc.) to predict which customers are most likely to subscribe before any campaign contact is made.
- **Performance:**  
  - **Accuracy:** ~62%
  - **Recall:** ~62% (captures most actual subscribers)
  - **F1 Score:** ~0.17
- **Who gets flagged:**  
  - **Retirees, high-balance customers, and those aged 60+** are most likely to be flagged as high-potential subscribers
  - **Younger customers (≤29)** also show above-average interest and are included
  - **Middle-aged (30–59)**, while the largest group, are less likely to be flagged

#### 2️⃣ Post-Call Model (Model 2) — *Who to Focus On After Initial Contact*

- **What it predicts:** Uses all available features, including campaign data (call duration, number of contacts, month, etc.) to predict which customers are most likely to subscribe after being contacted.
- **Performance:**  
  - **Recall:** up to 91% (captures nearly all actual subscribers after contact)
  - **F1 Score:** up to 0.48
- **Key predictors:**
  - **Call duration** is the strongest predictor—longer, quality conversations matter most
  - **Mobile contacts** (vs. landline) and those contacted 2–3 times (not more) are more likely to convert
  - **Retirees, high-balance, and 60+ customers** remain top targets

### Customer Segmentation Insights

- **KMeans clustering** on subscribers revealed:
  - **Majority Segment (94%)**: Typical subscribers
  - **Minority Segment (6%)**: Distinct, potentially high-value or niche group—should be targeted with specialized offers

### Business Recommendations

1. **Focus on High-Value Segments:**
   - Prioritize **retirees, high-balance customers, and the 60+ age group** for marketing campaigns
   - Use the minority cluster for targeted, personalized offers

2. **Adopt the Two-Stage ML Approach:**
   - **Pre-Call Model:** Use to select initial call targets, maximizing subscriber capture
   - **Post-Call Model:** Use after first contact to focus follow-up on the most promising leads

3. **Optimize Call Strategy:**
   - Limit campaign contacts to 2–3 per customer
   - Emphasize mobile contact over landline
   - Train agents to engage longer with promising leads

4. **Continuous Model Improvement:**
   - Regularly retrain models with new campaign data
   - Monitor performance and adjust thresholds to maintain high recall

5. **Address Root Causes of Low Subscription:**
   - Improve targeting to reach interested segments
   - Consider product adjustments for better market fit
   - Address trust and timing issues in campaign messaging

### 🎯 Business Impact

- **Maximized Subscriber Capture:** By focusing on recall, the company will reach most potential subscribers, directly increasing revenue
- **Efficient Resource Allocation:** Reduces wasted calls and human effort, improving ROI
- **Actionable Segmentation:** Enables differentiated marketing strategies for unique customer groups

---

## 🤝 Contributing

1. Fork → `git checkout -b feat/<your-idea>`
2. Commit (Conventional Commits welcome)
3. PR to **main**—CI will lint & test automatically.
4. Grab coffee; we'll review quickly. ☕

---

## 🪪 License

MIT—do great things, just give credit.

---

## 👤 Author & Links

**Osama Izhar**

* GitHub: [https://github.com/osamaizhar](https://github.com/osamaizhar)
* LinkedIn: [https://www.linkedin.com/in/osamaizhar-b4727116a/](https://www.linkedin.com/in/osamaizhar-b4727116a/)
* Apziva AI Residency: [https://www.apziva.com/](https://www.apziva.com/)
