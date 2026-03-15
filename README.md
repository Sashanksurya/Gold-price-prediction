# Gold-price-prediction
Gold Price Prediction  Demo terminal output with real INR prices Step-by-step API key setup instructions Tech stack table, project structure, future improvements
# Gold Price Prediction (INR) — Real Time

A machine learning project that fetches **live gold prices in Indian Rupees** using GoldAPI and predicts tomorrow's gold price using a **Random Forest Regressor**.

---

## Features

- Fetches real-time gold price from [GoldAPI.io](https://www.goldapi.io) in INR
- Shows price per gram, per 10 grams, and per troy ounce
- Generates 500 days of realistic historical data anchored to the live price
- Engineers 19 features: lag prices, moving averages, rolling std, momentum, date parts
- Trains a Random Forest with 300 decision trees
- Evaluates with MAE, RMSE, and R² score
- Predicts tomorrow's price with UP/DOWN direction and percentage change
- Saves a two-panel chart: Actual vs Predicted + Feature Importance

---

## Demo Output

```
=======================================================
   GOLD PRICE PREDICTION — REAL TIME (INR)
=======================================================

Live Gold Price  (15 Mar 2026, 10:30 AM)
   Per gram (24K) : Rs.    8,750.00
   Per 10 grams   : Rs.   87,500.00
   Per troy oz    : Rs.  271,900.00

Model Performance (Random Forest)
   MAE  : Rs.500.00
   RMSE : Rs.640.00
   R²   : 0.52

Prediction for 16 Mar 2026
   Today's price    : Rs.   87,500.00
   Predicted price  : Rs.   87,668.00
   Expected change  : UP by Rs.168.00  (0.19%)
```

---

## Tech Stack

| Library | Purpose |
|---|---|
| `requests` | Fetch live gold price from API |
| `pandas` | Store and manipulate price data |
| `numpy` | Generate historical data and math |
| `scikit-learn` | Random Forest, train/test split, evaluation |
| `matplotlib` | Plot and save chart as PNG |

---

## Installation

```bash
git clone https://github.com/yourusername/gold-price-prediction.git
cd gold-price-prediction
pip install pandas numpy scikit-learn matplotlib requests
```

---

## API Key Setup

1. Go to [https://www.goldapi.io](https://www.goldapi.io)
2. Click **Get Free API Key** — no credit card needed
3. Open `gold_price_prediction.py` and replace:

```python
API_KEY = "YOUR_API_KEY_HERE"
```

> Without an API key the project runs in **demo mode** with a sample price.

---

## Run

```bash
python gold_price_prediction.py
```

Output chart saved as `gold_price_prediction.png`.

---

## How It Works

```
Live API Price → Generate History → Feature Engineering
      → Train/Test Split → Random Forest (300 trees)
      → Evaluate → Predict Tomorrow → Save Chart
```

---

## Project Structure

```
gold_price_prediction/
├── gold_price_prediction.py   ← main script
├── gold_prices.csv            ← generated on first run
├── gold_price_prediction.png  ← chart saved on first run
├── requirements.txt
└── README.md
```

---

## Future Improvements

- Use real historical API data instead of simulated
- Add USD/INR exchange rate and Sensex as features
- Try XGBoost or LSTM for better accuracy
- Build a Flask web dashboard
- Send daily prediction as email/WhatsApp alert
