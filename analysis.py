import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.stats import ttest_ind

# CONFIG

SENTIMENT_FILE = "data/fear_greed_index.csv"
TRADER_FILE = "data/historical_data.csv"

# LOAD DATA

def load_data():
    sentiment = pd.read_csv(SENTIMENT_FILE)
    trader = pd.read_csv(TRADER_FILE)
    return sentiment, trader


# DATA PREPARATION

def prepare_data(sentiment, trader):

    print("Sentiment Shape:", sentiment.shape)
    print("Trader Shape:", trader.shape)

    print("\nMissing Values:")
    print(sentiment.isnull().sum())
    print(trader.isnull().sum())

    print("\nDuplicates:")
    print("Sentiment:", sentiment.duplicated().sum())
    print("Trader:", trader.duplicated().sum())

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
        trader['leverage'] = trader['Size Tokens'] / trader['Start Position']

    df = trader.merge(
        sentiment[['date', 'classification']],
        on='date',
        how='left'
    )
    df = df.replace([np.inf, -np.inf], np.nan)
    df['leverage'] = df['leverage'].fillna(df['leverage'].median())
    df['leverage'] = np.clip(
    df['leverage'],
    0,
    df['leverage'].quantile(0.99)
    )
    df = df.dropna(subset=['classification'])

    print("\nMerged Shape:", df.shape)

    return df

# FEATURE ENGINEERING

def create_features(df):

    df['win'] = df['closedPnL'] > 0

    daily = df.groupby(['account', 'date']).agg({
        'closedPnL': 'sum',
        'win': 'mean',
        'size_usd': 'mean',
        'leverage': 'mean',
        'side': 'count'
    }).reset_index()

    daily.rename(columns={
        'closedPnL': 'daily_pnl',
        'win': 'win_rate',
        'size_usd': 'avg_trade_size',
        'side': 'trade_count'
    }, inplace=True)

    daily['rolling_max'] = daily.groupby('account')['daily_pnl'].cummax()
    daily['drawdown'] = daily['daily_pnl'] - daily['rolling_max']

    return df, daily

#– ANALYSIS

def performance_by_sentiment(df):
    perf = df.groupby('classification').agg({
        'closedPnL': ['mean', 'std'],
        'win': 'mean'
    })
    print("\nPerformance by Sentiment")
    print(perf)
    return perf


def behavior_by_sentiment(df):
    behavior = df.groupby('classification').agg({
        'size_usd': 'mean',
        'leverage': 'mean',
        'account': 'count'
    }).rename(columns={'account': 'trade_count'})

    print("\nBehavior by Sentiment")
    print(behavior)
    ls = df.groupby(['classification', 'side']).size().unstack()
    print("\nLong/Short Ratio")
    print(ls)

    return behavior

# SEGMENTATION

def segment_traders(daily):

    summary = daily.groupby('account').agg({
        'daily_pnl': 'mean',
        'leverage': 'mean',
        'trade_count': 'mean'
    }).reset_index()

    summary['leverage_segment'] = np.where(
        summary['leverage'] > summary['leverage'].median(),
        'High Leverage',
        'Low Leverage'
    )

    summary['frequency_segment'] = np.where(
        summary['trade_count'] > summary['trade_count'].median(),
        'Frequent',
        'Infrequent'
    )

    vol = daily.groupby('account')['daily_pnl'].std().reset_index()
    vol.rename(columns={'daily_pnl': 'volatility'}, inplace=True)

    summary = summary.merge(vol, on='account', how='left')

    summary['volatility'] = summary['volatility'].fillna(0)

    summary['consistency'] = np.where(
        summary['volatility'] < summary['volatility'].median(),
        'Consistent',
        'Inconsistent'
    )
    
    print("\nTrader Segments Created")
    print(summary.head())

    return summary

# VISUALS

def create_plots(df):

    plt.figure()

    df.groupby('classification')['closedPnL'].mean().plot(kind='bar')
    plt.title("Average PnL by Sentiment")
    plt.ylabel("Average PnL")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure()
    df.groupby('classification').size().plot(kind='bar')
    plt.title("Trade Count by Sentiment")
    plt.ylabel("Number of Trades")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.scatter(df['size_usd'], df['closedPnL'], alpha=0.2)
    plt.title("Trade Size vs PnL")
    plt.xlabel("Trade Size (USD)")
    plt.ylabel("PnL")
    plt.tight_layout()
    plt.show()

# STRATEGY RULES

def strategy_rules(df):

    summary = df.groupby('classification')['closedPnL'].mean()

    best = summary.idxmax()
    worst = summary.idxmin()

    print("\nSTRATEGY INSIGHTS")
    print(f"1️⃣ Highest profitability during: {best}")
    print("   → Increase trade size moderately.")

    print(f"2️⃣ Weakest performance during: {worst}")
    print("   → Reduce leverage and tighten stops.")

    high_lev = df[df['leverage'] > df['leverage'].median()]['closedPnL'].mean()
    low_lev = df[df['leverage'] <= df['leverage'].median()]['closedPnL'].mean()

    print("\n3️⃣ High leverage vs Low leverage performance:")
    print("High Leverage Avg PnL:", high_lev)
    print("Low Leverage Avg PnL:", low_lev)


def statistical_test(df):

    fear = df[df['classification'] == 'Fear']['closedPnL']
    greed = df[df['classification'] == 'Greed']['closedPnL']

    stat, p_value = ttest_ind(fear, greed, equal_var=False)

    print("\nStatistical Test: Fear vs Greed PnL")
    print("T-statistic:", stat)
    print("P-value:", p_value)

    if p_value < 0.05:
        print("Result: Statistically significant difference ✅")
    else:
        print("Result: No significant difference ❌")
        
        
# SIMPLE MODEL

def simple_model(df):

    df_model = df.copy()

    # Clean infinity values (safety)
    df_model = df_model.replace([np.inf, -np.inf], np.nan)
    df_model['leverage'] = df_model['leverage'].fillna(df_model['leverage'].median())

    # Encode sentiment
    df_model['sentiment_encoded'] = df_model['classification'].astype('category').cat.codes
    df_model['profitable'] = (df_model['closedPnL'] > 0).astype(int)

    X = df_model[['sentiment_encoded', 'size_usd', 'leverage']]
    y = df_model['profitable']

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\nImproved Predictive Model Report")
    print(classification_report(y_test, preds))

# MAIN

if __name__ == "__main__":

    sentiment, trader = load_data()
    df = prepare_data(sentiment, trader)

    df, daily = create_features(df)

    performance_by_sentiment(df)
    behavior_by_sentiment(df)
    statistical_test(df)
    segments = segment_traders(daily)

    create_plots(df)

    strategy_rules(df)

    simple_model(df)

    print("\nAnalysis Complete 🚀")