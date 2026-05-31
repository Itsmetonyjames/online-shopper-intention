# Online Shopper Purchase Intention — ML Analysis

Predicting whether an e-commerce browsing session will result in a purchase, combined with K-Means clustering to identify natural shopper segments.

---

## Business Questions

> 1. *Can we predict purchase intent from browsing behaviour — enabling real-time intervention?*
> 2. *What natural shopper profiles exist within session data?*

---

## Dataset

| Property | Detail |
|---|---|
| Source | UCI Machine Learning Repository — Sakar et al. (2019) |
| Size | 12,330 browsing sessions |
| Features | 17 (page counts, durations, bounce/exit rates, traffic type, visitor type, month) |
| Target | `Revenue` — True if session ended in a purchase |
| Class split | ~84% non-purchase / ~16% purchase |

---

## Approach

| Stage | Method |
|---|---|
| Supervised learning | Logistic Regression with class balancing |
| Unsupervised learning | K-Means Clustering |
| Preprocessing | StandardScaler + OneHotEncoder via sklearn Pipeline |
| Evaluation (supervised) | Accuracy, Precision, Recall, F1, ROC-AUC |
| Evaluation (unsupervised) | Elbow Method + Silhouette Score |

---

## Results

### Supervised — Logistic Regression

| Metric | Score |
|---|---|
| Accuracy | 0.85 |
| Precision | 0.51 |
| Recall | 0.75 |
| F1-Score | 0.61 |
| ROC-AUC | **0.90** |

### Unsupervised — K-Means

| Metric | Value |
|---|---|
| Optimal clusters (k) | 2 |
| Max Silhouette Score | 0.29 |
| Cluster 0 purchase rate | 27.9% |
| Cluster 1 purchase rate | 13.2% |

---

## Key Findings

- **Page value** is the strongest predictor of purchase intent — sessions with higher page values convert at significantly higher rates
- **Cluster 0** (high-intent browsers): lower bounce rates, higher page values, purchase rate ~28% — prime candidates for real-time discount triggers
- **Cluster 1** (passive browsers): longer sessions but low engagement quality, purchase rate ~13% — better targeted through longer-term re-engagement campaigns
- **Class imbalance** is a critical challenge — accuracy alone is misleading; ROC-AUC of 0.90 confirms the model genuinely discriminates between buyers and non-buyers
- Combining supervised + unsupervised methods gives a fuller picture than either approach alone

---

## Real-World Applications

- **Real-time**: Flag high-intent sessions (Cluster 0) for live chat or discount triggers
- **Segmentation**: Use cluster profiles to personalise email campaigns and landing pages
- **Product teams**: Use feature importance to improve page layout and reduce bounce rates

---

## Files

| File | Description |
|---|---|
| `Online_Shopper_Portfolio.ipynb` | Full analysis notebook |
| `Online_Shopper_Report_Clean.pdf` | Project report |
| `AIML_Artefact_1_Dataset.csv` | Dataset (UCI Online Shoppers Intention) |

---

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook Online_Shopper_Portfolio.ipynb
```

---

## Tools Used

`Python` `Pandas` `NumPy` `Scikit-learn` `Matplotlib` `Seaborn`

---

*Tony James — MSc Data Science, York St John University, London*  
[LinkedIn](https://linkedin.com/in/tony-james-4bba63195) | [GitHub](https://github.com/itsmetonyjames)
