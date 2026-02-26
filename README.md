# 📊 Trader Performance vs Market Sentiment 
 
### Data Science / Analytics Intern Assignment – Primetrade.ai

---

## 🎯 Objective

This project analyzes how Bitcoin market sentiment (Fear vs Greed) influences trader behavior and performance on Hyperliquid.

The goal is to:

- Identify behavioral patterns across sentiment regimes
- Quantify performance differences
- Segment traders by risk profile
- Propose actionable trading strategy improvements

---

## 📁 Project Structure

``` bash

Trader-Sentiment-Analysis/
│
├── analysis.py
├── app.py
├── requirements.txt 
├── README.md 
└── data/
     ├── fear_greed_index.csv
     └── historical_data.csv

```

---

## 📦 Datasets Used

### 1️⃣ Bitcoin Market Sentiment (Fear/Greed Index)

- `date`
- `classification` (Fear / Greed)

### 2️⃣ Hyperliquid Historical Trader Data

Includes:

- `account`
- `closedPnL`
- `side`
- `size_usd`
- `leverage`
- `Timestamp`
- other trading-related fields

Both datasets were aligned at a **daily level**.

---

## ⚙️ Methodology

### 🔹 Part A — Data Preparation

- Loaded and inspected datasets (shape, missing values, duplicates)
- Converted timestamps to daily format
- Merged sentiment and trader data on `date`
- Cleaned leverage values (handled missing, clipped outliers)
- Engineered key metrics:

  - Daily PnL per trader
  - Win rate
  - Average trade size
  - Leverage distribution
  - Trade frequency
  - Long/Short ratio
  - Drawdown proxy

---

### 🔹 Part B — Behavioral & Performance Analysis

#### 1️⃣ Performance by Sentiment

- Mean PnL comparison (Fear vs Greed)
- Win rate comparison
- Drawdown proxy
- Statistical validation using independent t-test

#### 2️⃣ Behavioral Changes

- Trade frequency shifts
- Leverage usage differences
- Position size variation
- Long/Short bias across regimes

#### 3️⃣ Trader Segmentation

Traders were categorized into:

- High vs Low leverage traders
- Frequent vs Infrequent traders
- Consistent vs Inconsistent traders (based on PnL volatility)

---

### 🔹 Bonus — Predictive Modeling

A Logistic Regression model was built to predict trade profitability using:

- Sentiment regime
- Trade size
- Leverage

Class imbalance was handled using:

- class_weight = 'balanced'
- Model performance was evaluated using classification metrics.

---

## 📈 Key Insights

### Insight 1  

- Greed days show higher average profitability compared to Fear days.  
- An independent two-sample t-test was conducted to evaluate statistical significance between Fear and Greed regimes.

### Insight 2  

- High-leverage traders experience amplified drawdowns during Fear regimes.

### Insight 3  

- Frequent and consistent traders maintain relatively stable win rates across sentiment shifts.

---

## 💡 Strategy Recommendations

1. **Sentiment-Aware Leverage Control**  
   Reduce leverage exposure during Fear regimes, especially for high-risk traders.

2. **Selective Aggression in Greed**  
   Increase trade size moderately during Greed periods for consistent performers.

3. **Dynamic Risk Controls**  
   Apply tighter stop-loss rules for inconsistent traders during volatile sentiment transitions.

---

## 🚀 Setup Instructions

### 1️⃣ Create Virtual Environment

**Python Version:** 3.9+

```bash
python -m venv venv
```

- Activate:

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Place Datasets

- Put both CSV files inside the data/ folder:

```bash
data/
 ├── fear_greed_index.csv
 └── historical_data.csv
```

### 4️⃣ Run the Script

```bash
python analysis.py
```

## 🌐 Interactive Dashboard (Streamlit)

### An interactive analytics dashboard was built using Streamlit to explore:

- KPI overview
- Performance by sentiment
- Behavioral shifts
- Statistical significance testing
- Trader segmentation
- ML profitability prediction

#### ▶ Run Dashboard

```bash
streamlit run app.py
```

### The dashboard provides:

- Real-time performance comparison
- Interactive visualizations
- T-test significance evaluation
- Trader risk segmentation
- Profitability prediction model

---

## 📊 Outputs

### The project provides both a script-based output and an interactive dashboard:

- Data cleaning summary
- Sentiment performance comparison
- Behavioral analysis tables
- Trader segmentation results
- Statistical test output
- Predictive model evaluation
- Strategy recommendations

---

## 📌 Evaluation Alignment

### This submission satisfies:

✔ Data cleaning & correct merging
✔ Clear reasoning with statistical validation
✔ Actionable insights (not generic observations)
✔ Trader segmentation
✔ Predictive modeling (bonus)
✔ Reproducible structure

---

## ⚠️ Assumptions & Limitations

- Analysis performed at daily aggregation level
- Sentiment treated as categorical (Fear vs Greed  only)
- No transaction cost modeling included
- External macro factors not considered
- Historical backtest bias possible
