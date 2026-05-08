# E-Commerce Customer Intelligence Platform
### End-to-End Retail Analytics + Forecasting + Segmentation Pipeline
**Python · Prophet · Scikit-learn · Streamlit · Plotly**

---

## One-Line Summary

> Built an end-to-end customer analytics platform on 1.06M retail transactions — engineering RFM segmentation, K-Means clustering validation, cohort retention analysis, 12-week Prophet revenue forecasting, and Apriori market basket analysis, delivered through an interactive Streamlit dashboard that surfaces actionable intelligence across 5,862 customers and 43 countries.

---

## Dashboard Preview

Run locally with:

```bash
streamlit run app/streamlit_app.py
```

---

## Problem Statement

A UK-based wholesale gift-ware retailer operating across **43 countries and 5,862 customers** had no unified system to understand customer behaviour, forecast future revenue, or identify product bundling opportunities. Raw transactional data spanning **two full fiscal years** existed as a flat CSV with no customer-level intelligence, no cohort tracking, and no predictive capability.

**Core business problems:**
- No visibility into which customers were at risk of churning
- No system to identify high-value customers worth protecting
- No revenue forecast to support inventory and staffing decisions
- No understanding of which products were frequently bought together
- No geographic breakdown of international revenue performance

---

## Repository Structure

```
E-Commerce Customer Intelligence Platform/
│
├── data/
│   ├── raw/
│   │   └── online_retail_II.csv          # Source dataset (1.06M rows)
│   └── processed/
│       ├── retail_clean.parquet          # Cleaned dataset (776K rows)
│       ├── rfm_segments.parquet          # RFM scores and segment labels
│       ├── cohort_matrix.parquet         # 25x25 retention matrix
│       ├── forecast.parquet              # Prophet forecast (105 history + 12 future weeks)
│       ├── association_rules.parquet     # 268 product association rules
│       └── geo_summary.parquet           # Country-level revenue metrics
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb            # Cleaning, EDA, feature engineering
│   ├── 02_rfm_segmentation.ipynb         # RFM scoring and K-Means clustering
│   ├── 03_cohort_retention.ipynb         # Cohort construction and retention matrix
│   ├── 04_forecasting.ipynb              # STL decomposition and Prophet forecasting
│   ├── 05_market_basket.ipynb            # Apriori association rule mining
│   └── 06_geo_analysis.ipynb             # Geographic revenue breakdown
│
├── app/
│   └── streamlit_app.py                  # Interactive multi-page dashboard
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

**Online Retail II — UCI Machine Learning Repository**
Created by Dr. Daqing Chen, London South Bank University

| Source | Link |
|---|---|
| UCI Repository | https://archive.uci.edu/dataset/502/online+retail+ii |
| Kaggle Mirror | https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci |

| Column | Description |
|---|---|
| Invoice | Unique invoice number (prefix C = cancellation) |
| StockCode | Product identifier |
| Description | Product name |
| Quantity | Units per transaction |
| InvoiceDate | Transaction timestamp |
| Price | Unit price in GBP |
| Customer ID | Unique customer identifier |
| Country | Customer country |

---

## Methodology

### Stage 1 — Data Cleaning (`01_data_cleaning.ipynb`)

Raw data required systematic cleaning before any analysis could begin. Each step was logged with before/after row counts to ensure full transparency over data loss decisions.

| Step | Rows Removed | Reason |
|---|---|---|
| Drop null Customer ID | 243,007 | Guest checkouts cannot be segmented |
| Drop null Description | 0 | No additional loss after prior step |
| Remove cancellations | 18,744 | Invoices prefixed with C |
| Remove Quantity/Price ≤ 0 | 71 | Returns, adjustments, data errors |
| Remove non-product StockCodes | 2,617 | Postage, bank charges, test entries |
| Drop duplicates | 26,060 | Exact duplicate rows |
| **Total removed** | **290,499 (27.2%)** | |
| **Clean dataset** | **776,872 rows** | |

Feature engineering added `Revenue`, `Year`, `Month`, `DayOfWeek`, `Hour`, and `YearMonth` columns. Clean data saved as Parquet (6.2 MB vs ~80 MB CSV) for fast downstream loading.

**EDA findings:**
- Revenue spikes every October/November — strong Christmas buying seasonality
- Thursday is the highest-revenue day; Saturday is near zero — confirms B2B wholesale operation
- All activity between 06:00 and 20:00, peaking at noon — no consumer-style late-night orders
- UK accounts for 83.7% of total revenue; Eire and Netherlands lead internationally

### Stage 2 — RFM Segmentation (`02_rfm_segmentation.ipynb`)

Each of the 5,862 customers was reduced to three behavioural signals:

- **Recency** — days since last purchase (lower = better)
- **Frequency** — number of distinct invoices (higher = better)
- **Monetary** — total revenue generated (higher = better)

Each dimension was scored 1–5 using quintile binning. Recency was scored inversely (recent buyers score 5). Scores were combined into an RFM string and mapped to eight segment labels using R and F score thresholds.

| Segment | Customers | Avg Recency | Avg Frequency | Avg Monetary |
|---|---|---|---|---|
| Champions | 1,477 | 20.4 days | 15.6 orders | £8,011 |
| Loyal Customers | 1,212 | 78.6 days | 5.4 orders | £2,002 |
| Lost | 1,084 | 547.9 days | 1.5 orders | £502 |
| Hibernating | 618 | 315.9 days | 1.3 orders | £422 |
| At Risk | 551 | 302.1 days | 5.3 orders | £2,088 |
| Recent Customers | 451 | 28.1 days | 1.5 orders | £924 |
| Potential Loyalists | 380 | 106.6 days | 1.4 orders | £530 |
| Cannot Lose Them | 89 | 493.8 days | 7.5 orders | £2,860 |

**K-Means clustering** was used to validate that the rule-based segments reflect genuine data structure rather than arbitrary cutoffs. The Elbow method and Silhouette score both identified **k=5** as optimal (Silhouette = 0.60). Five data-driven clusters emerged:

| Cluster | Label | Customers | Avg Monetary |
|---|---|---|---|
| 2 | Mega Accounts | 4 | £423,256 |
| 4 | High Value Regulars | 24 | £98,003 |
| 3 | Mid Tier Actives | 365 | £13,677 |
| 0 | Low Spend Majority | 3,564 | £1,878 |
| 1 | Dormant | 1,905 | £711 |

### Stage 3 — Cohort Retention Analysis (`03_cohort_retention.ipynb`)

Each customer was assigned to a cohort based on their first purchase month. The 25×25 retention matrix tracks the percentage of each cohort still active in each subsequent month.

**Key findings:**
- December 2009 cohort (n=952) is the strongest, retaining ~40% at month 23
- Most cohorts stabilise at 15–25% retention after month 1 — typical for B2B wholesale
- Bright cells along Q4 columns reflect the seasonal buying spike
- Average long-term retention flattens around 20% from month 2 onwards

### Stage 4 — Revenue Forecasting (`04_forecasting.ipynb`)

**STL decomposition** separated the weekly time series into trend, seasonality, and residual components. The trend component was nearly flat (£155K–£157K range), confirming that year-over-year revenue growth is driven almost entirely by seasonality, not structural business expansion.

**Prophet** was trained on 105 weeks of history with yearly seasonality in multiplicative mode (revenue swings scale with trend level). The learned seasonality component peaks at +60% in October/November and troughs at -70% in late December/early January — precisely matching the wholesale gift-ware buying cycle.

12-week forecast (Dec 2011 – Feb 2012):

| Week | Forecast | 95% Lower | 95% Upper |
|---|---|---|---|
| 2011-12-11 | £228,001 | £172,484 | £285,031 |
| 2011-12-25 | £81,068 | £26,328 | £135,626 |
| 2012-01-01 | £45,468 | £0 | £101,287 |
| 2012-01-22 | £158,169 | £100,970 | £212,711 |
| 2012-02-26 | £142,819 | £87,138 | £197,425 |

The model correctly predicts the Christmas shutdown dip and gradual January recovery.

### Stage 5 — Market Basket Analysis (`05_market_basket.ipynb`)

The Apriori algorithm was applied to UK transactions only. With 5,242 unique products, the full product matrix (33,384 invoices × 5,242 products) caused a 69.5 GB memory allocation failure at 1% support. The product space was restricted to the **top 200 products by purchase frequency** before mining.

| Parameter | Value | Rationale |
|---|---|---|
| min_support | 0.01 | ~334 baskets — statistically meaningful |
| max_len | 2 | Pairs only — triplets rarely actionable with sparse data |
| min_lift | 3.0 | 3× more likely than chance |

**268 rules generated. Average lift: 11.7×. Maximum lift: 33.8×.**

Top findings:
- Feltcraft Cushion Rabbit ↔ Feltcraft Cushion Butterfly — lift 33.8, confidence 66%
- Regency Teacup set (pink, green, roses) — consistently bought as a trio
- Gardeners Kneeling Pad variants — near-certain co-purchase with lift 32.9

### Stage 6 — Geographic Analysis (`06_geo_analysis.ipynb`)

| Country | Revenue | Orders | Customers | AOV |
|---|---|---|---|---|
| United Kingdom | £14,294,085 | 33,384 | 5,336 | £428 |
| Eire | £596,997 | 542 | 5 | £1,101 |
| Netherlands | £549,773 | 216 | 22 | £2,545 |
| Germany | £383,419 | 756 | 107 | £507 |
| France | £309,856 | 594 | 95 | £522 |
| Australia | £167,800 | 89 | 15 | £1,885 |

Netherlands stands out — only 22 customers but the highest AOV at £2,545, indicating a small number of very large wholesale accounts. Australia and Denmark show similar patterns of low customer count with bulk ordering behaviour.

---

## Key Results

### Headline Metrics

| Metric | Value |
|---|---|
| Total Revenue | £17,078,546 |
| Clean Transactions | 776,872 |
| Unique Customers | 5,862 |
| Unique Products | 4,625 |
| Average Order Value | £466 |
| Countries Served | 41 |
| Data Period | Dec 2009 – Dec 2011 |

### Customer Intelligence

- **Champions (1,477 customers)** average £8,011 revenue and purchase every 20 days — primary retention target
- **Cannot Lose Them (89 customers)** — high historical spend (£2,860 avg) but 494 days since last purchase — urgent re-engagement required
- **Lost + Hibernating = 1,702 customers (29%)** — significant win-back opportunity
- **4 Mega Accounts** averaging £423K each — extreme revenue concentration risk

### Forecasting

- Prophet correctly identifies Christmas-period revenue collapse (-70% seasonality effect)
- January 2012 forecast: £45K minimum (post-Christmas trough)
- Recovery to £158K by late January as wholesale buying resumes

### Market Basket

- 268 actionable product association rules identified
- Average lift of 11.7× means co-purchased products appear together far more than chance
- Clear product set structure: teacup collections, cushion ranges, kneeling pad variants — strong bundling and cross-sell opportunity

---

## Streamlit Dashboard

Six interactive pages:

| Page | Contents |
|---|---|
| Overview | 5 KPI cards, monthly revenue trend, top products and countries |
| RFM Segmentation | Segment bar chart, K-Means donut, segment profile table, customer lookup |
| Cohort Retention | Interactive heatmap (25×25), average retention curve |
| Forecasting | Prophet forecast chart with 95% interval, 12-week forecast table |
| Market Basket | Top rules by lift, support vs confidence scatter, filterable rules table |
| Geography | Choropleth world map, revenue and AOV bar charts, country summary table |

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.14 |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn (K-Means, StandardScaler, Silhouette) |
| Time Series | Prophet, Statsmodels (STL) |
| Market Basket | MLxtend (Apriori, Association Rules) |
| Storage | Parquet (PyArrow) |
| Dashboard | Streamlit |
| Environment | Antigravity IDE, Python 3.14 |

---

## Setup & Installation

```bash
# 1. Clone the repository
git clone <your-repository-url>
cd "E-Commerce Customer Intelligence Platform"

# 2. Install dependencies into your environment
pip install -r requirements.txt

# 3. Place the dataset
# Download online_retail_II.csv from UCI or Kaggle and place it in:
# data/raw/online_retail_II.csv

# 4. Run notebooks in order
# Open and run each notebook sequentially:
# notebooks/01_data_cleaning.ipynb      → generates retail_clean.parquet
# notebooks/02_rfm_segmentation.ipynb   → generates rfm_segments.parquet
# notebooks/03_cohort_retention.ipynb   → generates cohort_matrix.parquet
# notebooks/04_forecasting.ipynb        → generates forecast.parquet
# notebooks/05_market_basket.ipynb      → generates association_rules.parquet
# notebooks/06_geo_analysis.ipynb       → generates geo_summary.parquet

# 5. Launch the dashboard
streamlit run app/streamlit_app.py
```

---

## Key Architectural Decisions

**Parquet over CSV for processed data** — 776K clean rows compress from ~80 MB CSV to 6.2 MB Parquet. All downstream notebooks and the dashboard load in milliseconds rather than seconds.

**Quintile scoring for RFM over fixed thresholds** — Fixed thresholds assume prior knowledge of the data distribution. Quintile-based scoring is self-calibrating and works correctly regardless of the revenue scale or customer behaviour patterns of a given dataset.

**K-Means as validation, not replacement** — Rule-based segments are interpretable and business-friendly. K-Means serves as an independent check that the rules reflect genuine data structure. Both outputs are retained in the final dataset.

**Multiplicative seasonality in Prophet** — Revenue swings scale with the underlying trend level. Additive seasonality would imply a fixed £X swing regardless of trend — incorrect for a business with growing baseline revenue. Multiplicative mode correctly models percentage-based swings.

**Top-200 product filter for Apriori** — The full 5,242-product matrix at 1% support requires 69.5 GB RAM — not feasible on standard hardware. Restricting to the 200 most-purchased products covers the commercially relevant combinations while keeping memory usage manageable.

**Proportional cohort indexing** — Cohort index is computed as `(OrderMonth - CohortMonth).days // 30` rather than a simple month difference. This handles varying month lengths correctly and avoids off-by-one errors at month boundaries.

---

## Author

**Sujal Deb**

Python · Forecasting · Segmentation · Market Basket Analysis · Streamlit
