# Gold Price Prediction with REAL-TIME prices (INR)
# Uses GoldAPI.io — free API, no credit card needed
# Libraries: pandas, numpy, scikit-learn, matplotlib, requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import json
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ══════════════════════════════════════════════════════
#  CONFIGURATION — paste your API key here
# ══════════════════════════════════════════════════════
API_KEY = "goldapi-elm7hlsmmqdqfgu-io"   # <-- replace with your key from goldapi.io
# ══════════════════════════════════════════════════════


def fetch_live_gold_price(api_key):
    """Fetch current gold price in INR from GoldAPI.io"""
    url = "https://www.goldapi.io/api/XAU/INR"
    headers = {
        "x-access-token": api_key,
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        # price_gram_24k = price per gram of 24K gold in INR
        price_per_gram = data.get("price_gram_24k", None)
        price_per_10g  = price_per_gram * 10 if price_per_gram else None
        price_oz       = data.get("price", None)           # price per troy oz
        timestamp      = data.get("timestamp", None)

        return {
            "price_per_gram": round(price_per_gram, 2) if price_per_gram else None,
            "price_per_10g":  round(price_per_10g, 2)  if price_per_10g  else None,
            "price_per_oz":   round(price_oz, 2)        if price_oz       else None,
            "timestamp":      datetime.fromtimestamp(timestamp) if timestamp else datetime.now(),
            "currency":       "INR"
        }
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        return None


def generate_historical_data(current_price_10g, days=500):
    """
    Generate historical data anchored to the real current price.
    In a production app, you'd fetch this from the API's historical endpoint.
    """
    np.random.seed(42)
    dates = pd.date_range(end=datetime.today(), periods=days, freq="D")

    # Work backwards from today's real price
    t        = np.arange(days)
    end_val  = current_price_10g
    start_val = end_val * 0.80           # assume ~20% lower 500 days ago
    trend    = np.linspace(start_val, end_val, days)
    seasonal = (end_val * 0.03) * np.sin(2 * np.pi * t / 365)
    noise    = np.random.normal(0, end_val * 0.003, days)
    prices   = trend + seasonal + noise

    df = pd.DataFrame({"Date": dates, "Price": prices})
    return df


def engineer_features(df):
    """Add all feature columns needed by the model"""
    df = df.copy()
    df["Day"]       = df["Date"].dt.day
    df["Month"]     = df["Date"].dt.month
    df["Year"]      = df["Date"].dt.year
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Quarter"]   = df["Date"].dt.quarter

    for lag in [1, 3, 7, 14, 30]:
        df[f"Lag_{lag}"] = df["Price"].shift(lag)

    for window in [7, 14, 30]:
        df[f"MA_{window}"]  = df["Price"].rolling(window).mean()
        df[f"STD_{window}"] = df["Price"].rolling(window).std()

    df["Momentum_7"]  = df["Price"] - df["Price"].shift(7)
    df["Momentum_30"] = df["Price"] - df["Price"].shift(30)

    df.dropna(inplace=True)
    return df


FEATURES = [
    "Day", "Month", "Year", "DayOfWeek", "DayOfYear", "Quarter",
    "Lag_1", "Lag_3", "Lag_7", "Lag_14", "Lag_30",
    "MA_7", "MA_14", "MA_30",
    "STD_7", "STD_14", "STD_30",
    "Momentum_7", "Momentum_30"
]


# ── MAIN ──────────────────────────────────────────────
print("=" * 55)
print("   GOLD PRICE PREDICTION — REAL TIME (INR)")
print("=" * 55)

# 1. Fetch live price
if API_KEY == "YOUR_API_KEY_HERE":
    print("\nNo API key set — using demo price (Rs.87,500 / 10g)")
    live = {
        "price_per_10g":  87500.00,
        "price_per_gram": 8750.00,
        "price_per_oz":   271_900.00,
        "timestamp":      datetime.now(),
        "currency":       "INR"
    }
else:
    print("\nFetching live gold price from GoldAPI.io ...")
    live = fetch_live_gold_price(API_KEY)
    if not live:
        print("Could not fetch live data. Using fallback price.")
        live = {"price_per_10g": 87500.00, "price_per_gram": 8750.00,
                "price_per_oz": 271900.00, "timestamp": datetime.now(), "currency": "INR"}

print(f"\nLive Gold Price  ({live['timestamp'].strftime('%d %b %Y, %I:%M %p')})")
print(f"   Per gram (24K) : Rs. {live['price_per_gram']:>10,.2f}")
print(f"   Per 10 grams   : Rs. {live['price_per_10g']:>10,.2f}  ← standard Indian unit")
print(f"   Per troy oz    : Rs. {live['price_per_oz']:>10,.2f}")

# 2. Build historical dataset anchored to real price
df = generate_historical_data(live["price_per_10g"], days=500)
df.to_csv("gold_prices.csv", index=False)
print(f"\nHistorical data  : {len(df)} days generated")
print(f"Price range      : Rs.{df['Price'].min():,.0f} — Rs.{df['Price'].max():,.0f}")

# 3. Feature engineering
df = engineer_features(df)

# 4. Train / test split
X = df[FEATURES].values
y = df["Price"].values
split   = int(len(X) * 0.85)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Train model
print("\nTraining Random Forest model ...")
model = RandomForestRegressor(
    n_estimators=300, max_depth=None,
    min_samples_split=2, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\nModel Performance")
print(f"   MAE  : Rs.{mae:,.2f}")
print(f"   RMSE : Rs.{rmse:,.2f}")
print(f"   R²   : {r2:.4f}")

# 7. Predict TOMORROW's price
tomorrow_date = datetime.today() + timedelta(days=1)
last_row      = X[[-1]]
next_price    = model.predict(last_row)[0]
change        = next_price - live["price_per_10g"]
direction     = "UP" if change > 0 else "DOWN"
pct_change    = abs(change / live["price_per_10g"]) * 100

print(f"\nPrediction for {tomorrow_date.strftime('%d %b %Y')}")
print(f"   Today's price    : Rs.{live['price_per_10g']:>10,.2f}")
print(f"   Predicted price  : Rs.{next_price:>10,.2f}")
print(f"   Expected change  : {direction} by Rs.{abs(change):,.2f}  ({pct_change:.2f}%)")

# 8. Feature Importance
importances  = pd.Series(model.feature_importances_, index=FEATURES)
top_features = importances.sort_values(ascending=False).head(10)

# 9. Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"Gold Price Prediction (INR per 10g)  —  Live price: Rs.{live['price_per_10g']:,.0f}",
    fontsize=13, fontweight="bold"
)

# Left: Actual vs Predicted
ax1 = axes[0]
test_dates = df["Date"].values[split:]
ax1.plot(test_dates, y_test, label="Actual",    color="gold",      linewidth=2)
ax1.plot(test_dates, y_pred, label="Predicted", color="steelblue", linestyle="--", linewidth=2)
ax1.axhline(live["price_per_10g"], color="red", linestyle=":", linewidth=1.5,
            label=f"Live: Rs.{live['price_per_10g']:,.0f}")
ax1.set_title("Actual vs Predicted")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price (Rs. per 10g)")
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"Rs.{x:,.0f}"))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax1.legend()
ax1.grid(alpha=0.3)

# Right: Feature Importance
ax2 = axes[1]
top_features.plot(kind="barh", ax=ax2, color="goldenrod")
ax2.set_title("Top 10 Feature Importances")
ax2.set_xlabel("Importance Score")
ax2.invert_yaxis()
ax2.grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("gold_price_prediction.png", dpi=150)
plt.show()
print("\nChart saved to gold_price_prediction.png")
print("=" * 55)
