import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

SENTIMENT_FILE = "data/fear_greed_index.csv"
TRADER_FILE = "data/historical_data.csv"

st.set_page_config(page_title="Advanced Trader Sentiment Dashboard", layout="wide")

st.title("📊 Advanced Trader Sentiment Intelligence Dashboard")
st.markdown("Performance • Behavior • Segmentation • Statistical Testing • ML Prediction")


@st.cache_data
def load_and_prepare():

    sentiment = pd.read_csv(SENTIMENT_FILE)
    trader = pd.read_csv(TRADER_FILE)

    sentiment['date'] = pd.to_datetime(sentiment['date']).dt.normalize()
    sentiment['classification'] = sentiment['classification'].str.capitalize()

    trader['Timestamp'] = pd.to_datetime(trader['Timestamp'], unit='ms')
    trader['date'] = trader['Timestamp'].dt.normalize()

    trader = trader.rename(columns={
        'Account': 'account',
        'Closed PnL': 'closedPnL',
        'Side': 'side',
        'Size USD': 'size_usd',
        'Leverage': 'leverage'
    })

    if 'leverage' not in trader.columns:
        trader['leverage'] = 1

    df = trader.merge(
        sentiment[['date', 'classification']],
        on='date',
        how='left'
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    df['leverage'] = df['leverage'].fillna(df['leverage'].median())
    df = df.dropna(subset=['classification'])

    df['win'] = df['closedPnL'] > 0

    return df


df = load_and_prepare()

# KPI SECTION

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Trades", len(df))
col2.metric("Total PnL", round(df['closedPnL'].sum(), 2))
col3.metric("Win Rate", round(df['win'].mean()*100, 2))
col4.metric("Avg Leverage", round(df['leverage'].mean(), 2))

st.divider()

# PERFORMANCE BY SENTIMENT

st.subheader("📈 Performance by Sentiment")

perf = df.groupby('classification').agg({
    'closedPnL': ['mean', 'std'],
    'win': 'mean'
})

st.dataframe(perf)

fig1, ax1 = plt.subplots()
df.groupby('classification')['closedPnL'].mean().plot(kind='bar', ax=ax1)
ax1.set_ylabel("Average PnL")
st.pyplot(fig1)

# BEHAVIOR ANALYSIS

st.subheader("📊 Trader Behavior by Sentiment")

behavior = df.groupby('classification').agg({
    'size_usd': 'mean',
    'leverage': 'mean',
    'account': 'count'
}).rename(columns={'account': 'trade_count'})

st.dataframe(behavior)

fig2, ax2 = plt.subplots()
behavior['trade_count'].plot(kind='bar', ax=ax2)
ax2.set_ylabel("Trade Count")
st.pyplot(fig2)

# SCATTER PLOT

st.subheader("💰 Trade Size vs PnL")

fig3, ax3 = plt.subplots()
ax3.scatter(df['size_usd'], df['closedPnL'], alpha=0.3)
ax3.set_xlabel("Trade Size (USD)")
ax3.set_ylabel("PnL")
st.pyplot(fig3)

# STATISTICAL TEST

st.subheader("🧪 Statistical Test (Fear vs Greed)")

fear = df[df['classification'] == 'Fear']['closedPnL']
greed = df[df['classification'] == 'Greed']['closedPnL']

if len(fear) > 0 and len(greed) > 0:
    stat, p_value = ttest_ind(fear, greed, equal_var=False)

    st.write("T-statistic:", round(stat, 4))
    st.write("P-value:", round(p_value, 6))

    if p_value < 0.05:
        st.success("Statistically Significant Difference ✅")
    else:
        st.warning("No Significant Difference ❌")

# SEGMENTATION

st.subheader("👥 Trader Segmentation")

summary = df.groupby('account').agg({
    'closedPnL': 'mean',
    'leverage': 'mean',
    'size_usd': 'mean'
}).reset_index()

summary['leverage_segment'] = np.where(
    summary['leverage'] > summary['leverage'].median(),
    'High Leverage',
    'Low Leverage'
)

st.dataframe(summary.head(20))

# ML PREDICTION MODEL

st.subheader("🤖 Profitability Prediction Model")

df_model = df.copy()
df_model['sentiment_encoded'] = df_model['classification'].astype('category').cat.codes
df_model['profitable'] = (df_model['closedPnL'] > 0).astype(int)

X = df_model[['sentiment_encoded', 'size_usd', 'leverage']]
y = df_model['profitable']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

preds = model.predict(X_test)

report = classification_report(y_test, preds, output_dict=True)
st.write("Model Accuracy:", round(report['accuracy'], 3))

# STRATEGY INSIGHTS

st.subheader("🧠 Strategy Insights")

summary_sentiment = df.groupby('classification')['closedPnL'].mean()

best = summary_sentiment.idxmax()
worst = summary_sentiment.idxmin()

st.success(f"Highest Profitability During: {best}")
st.error(f"Weakest Performance During: {worst}")

high_lev = df[df['leverage'] > df['leverage'].median()]['closedPnL'].mean()
low_lev = df[df['leverage'] <= df['leverage'].median()]['closedPnL'].mean()

st.write("High Leverage Avg PnL:", round(high_lev,2))
st.write("Low Leverage Avg PnL:", round(low_lev,2))

st.write("Analysis Complete 🚀")