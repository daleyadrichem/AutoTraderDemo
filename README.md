# **AI Trading Demo – Pattern Detection With Machine Learning**

This repository contains a Jupyter notebook that demonstrates how **AI models can analyze stock market data, detect patterns, and make simple price-direction predictions**.  
It is designed for **presentations, workshops, and educational demos** explaining what happens “behind the scenes” in AI — especially neural networks.

> ⚠️ **Educational Use Only**  
> This is *not* a real trading bot.  
> It should **not** be used for real-money trading or financial decision-making.

---

## ⭐ What This Demo Shows

- How to load real historical market data (using the S&P 500 ETF `SPY`)
- How to visualize price trends and common indicators
- How to engineer features for machine learning
- How to build and compare several AI models:
  - Logistic Regression (linear baseline)
  - Random Forest (non-linear ensemble)
  - Neural Network (MLP)
- How these models can predict whether the price will go **up** or **down** tomorrow
- A tiny, toy backtest demonstrating how predictions could drive a simple strategy
- Markdown explanations that guide the user step-by-step (great for teaching)

---

## 📂 Repository Structure

```
.
├── pyproject.toml        # uv project configuration
├── README.md             # You're reading it!
└── ai_trading_demo.ipynb # Main notebook with full walkthrough
````

---

## 🚀 Getting Started

### **1. Install uv (if you haven't already)**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
````

### **2. Install dependencies**

```bash
uv sync
```

### **3. Launch Jupyter**

```bash
uv run jupyter notebook
```

Open **`ai_trading_demo.ipynb`** and you're ready to go.

---

## 📊 Technologies Used

* **Python 3.10+**
* **Jupyter Notebook**
* **yfinance** for market data
* **pandas**, **numpy** for data processing
* **matplotlib** for plotting
* **scikit-learn** for ML models

---

## 🧠 Key Teaching Concepts

This repo is ideal for explaining:

### **Neural Networks Basics**

* Inputs → weights → hidden layers → activation → output
* Learning by minimizing prediction error
* Why non-linear functions help detect patterns

### **Trading Model Workflow**

1. Collect data
2. Engineer features
3. Train model
4. Evaluate performance
5. Convert predictions into actions

### **What AI Can and Cannot Do**

* Great at identifying historical patterns
* Not magic — does not *understand* markets
* Requires strong evaluation to avoid false confidence

Use this as the foundation for a talk or workshop segment on **how real AI systems process data and learn patterns**.

---

## 📉 Limitations (Intentional!)

This notebook **does not** include:

* Robust backtesting
* Risk management
* Transaction costs
* Market regime detection
* Live trading or broker integration

The goal is to keep the example simple and focused on **core AI concepts**, not real trading automation.

---

## 📜 License

MIT — free to use, modify, and share.

---

## 🙏 Contributing

Issues, suggestions, or improvements are welcome!
Feel free to open a PR or ask for extra features such as:

* LSTM (recurrent neural network) example
* More visualizations
* More datasets or assets
* Extended backtesting modules