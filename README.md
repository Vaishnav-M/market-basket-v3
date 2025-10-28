# 🛒 Instacart Market Basket Analysis

**Discover shopping patterns from 50,000 real customer orders using Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 What Does This Do?

This project analyzes **real shopping data** from Instacart (3.4M+ orders) to discover:

- **🔗 Product Associations**: Which items are bought together?
- **📊 Customer Patterns**: How do people shop?
- **💡 Smart Recommendations**: What should stores suggest?
- **📦 Product Bundles**: Optimal product groupings

## 🚀 Quick Start

### 1️⃣ Clone & Setup

```bash
git clone https://github.com/Vaishnav-M/instacart-market-basket-analysis.git
cd market_basket_analysis
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

### 2️⃣ Convert Instacart Data (First Time Only)

```bash
python convert_instacart_data.py
```

This creates `data/instacart_transactions_50k.csv` from the raw Instacart dataset.

### 3️⃣ Launch Streamlit App

```bash
streamlit run streamlit_app.py
```

Open browser: **http://localhost:8501** 🎉

### 📊 Interactive Visualizations
- Scatter plots (Support vs Confidence vs Lift)
- Top products bar charts
- Customer segment distribution
- Feature importance charts

## 🛠️ Technologies
- Python 3.10
- **Machine Learning:** scikit-learn (K-Means, Random Forest)
- **Data Mining:** mlxtend (Apriori algorithm)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Deployment:** Streamlit (web dashboard)

## 📂 Project Structure
```
market_basket_analysis/
├── data/                     # Transaction datasets
│   └── sample_transactions.csv
├── models/                   # Saved ML models
├── src/
│   ├── analysis.py          # Apriori algorithm & association rules
│   ├── ml_models.py         # K-Means & Random Forest models
│   └── utils.py             # Visualization & helper functions
├── app.py                   # Interactive Streamlit dashboard
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🚀 Getting Started

### 1. Create Virtual Environment
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Analysis
```bash
python src/analysis.py
```

### 4. Launch Streamlit App
```bash
streamlit run app.py
```

## 📊 Metrics Explained
- **Support**: How frequently an itemset appears in transactions
- **Confidence**: Probability of buying item B when item A is purchased
- **Lift**: How much more likely item B is purchased when item A is purchased (compared to random chance)

## � Sample Insights
- **Association Rules:** "Customers who buy bread and butter are 3x more likely to buy milk"
- **Customer Segments:** Premium shoppers spend 2.5x more than quick shoppers
- **Purchase Prediction:** 85% accuracy predicting toothpaste purchases when toothbrush is in basket
- **Classic Pattern:** Beer and diapers frequently bought together!

## 🎯 Business Applications
- **Product Placement:** Optimize shelf layouts based on purchase patterns
- **Customer Segmentation:** Personalized marketing for different shopper types
- **Promotional Bundles:** Design data-driven product bundles
- **Cross-Selling:** Predict and recommend complementary products
- **Inventory Management:** Stock products that sell together
- **Targeted Marketing:** Send personalized offers to customer segments

## 📝 License
MIT License
