# 🎯 Quick Start Guide - Market Basket Analysis

## Your CSV Should Look Like This:

```csv
Transaction,Item
1,Coconut Oil
1,Tomato
1,Onion
2,Milk
2,Bread
3,Coconut Oil
3,Curry Leaves
```

**Rules:**
✅ Two columns: `Transaction` and `Item`
✅ Multiple rows per transaction (one per item bought)
✅ Transaction IDs can be numbers or strings
✅ Item names can be anything (text)

---

## The 3 Algorithms Explained Simply:

### 🔍 **1. Association Rules** (What goes together?)
**Question:** "If someone buys X, what else do they buy?"

**Example:**
- 65% of people who buy Coconut Oil also buy Curry Leaves
- **Business Action:** Put these products near each other!

### 👥 **2. Customer Segmentation** (Who are my customers?)
**Question:** "What types of shoppers do I have?"

**Example:**
- Segment 1: VIP Shoppers (buy 15 items, spend ₹1,500)
- Segment 2: Quick Shoppers (buy 2 items, spend ₹200)
- **Business Action:** Different marketing for each segment!

### 🎯 **3. Purchase Prediction** (What will they buy next?)
**Question:** "Based on their cart, what should I recommend?"

**Example:**
- Customer has: Onion + Curry Leaves
- Model predicts: 85% chance they'll buy Coconut Oil
- **Business Action:** Show Coconut Oil recommendation!

---

## How the App Works:

```
INPUT CSV
    ↓
[Filter to Top 200 Products]  ← Avoid memory issues
    ↓
[Convert to Transactions]  ← Group items by Transaction ID
    ↓
┌─────────────────┬──────────────────┬────────────────────┐
│                 │                  │                    │
│  ASSOCIATION    │   CUSTOMER       │   PURCHASE         │
│  RULES          │   SEGMENTATION   │   PREDICTION       │
│  (Apriori)      │   (K-Means)      │   (Random Forest)  │
│                 │                  │                    │
│  Finds pairs    │  Groups          │  Predicts next     │
│  of products    │  customers       │  purchase from     │
│  bought         │  by behavior     │  current basket    │
│  together       │                  │                    │
│                 │                  │                    │
└─────────────────┴──────────────────┴────────────────────┘
              ↓
       [VISUALIZATIONS & INSIGHTS]
```

---

## Real Example with Your BigBasket Data:

### Input:
```csv
Transaction,Item
1,Coconut Oil
1,Tomato
1,Onion
1,Curry Leaves
2,Milk
2,Bread
```

### Output:

**Tab 1 - Association Rules:**
```
Rule: Coconut Oil → Curry Leaves
- Support: 2.5% (found in 2.5% of transactions)
- Confidence: 65% (65% who buy coconut oil buy curry leaves)
- Lift: 3.2 (3.2x more likely than random)

💡 Recommendation: Place Curry Leaves near Coconut Oil section
```

**Tab 2 - Customer Segmentation:**
```
Segment 0: VIP Shoppers (150 customers)
- Avg Items: 15.2
- Variety: 12.8 types
- Value: ₹1,520 per visit

Segment 1: Quick Shoppers (2,450 customers)
- Avg Items: 2.1
- Variety: 2.0 types
- Value: ₹210 per visit

💡 Recommendation: Focus store layout on Quick Shoppers (82% of customers)
```

**Tab 3 - Purchase Prediction:**
```
Target Product: Coconut Oil
Customer's Cart: [Onion, Curry Leaves, Tomato]

Prediction: 85% chance they'll buy Coconut Oil
Model Accuracy: 87%

💡 Recommendation: Show "Add Coconut Oil?" popup at checkout
```

---

## Parameters Explained:

### **Minimum Support** (0.01% - 5%)
- **Lower** = Find rare patterns (e.g., 0.1% = 20 transactions out of 20,000)
- **Higher** = Only popular patterns (e.g., 5% = 1,000 transactions)
- **Default:** 0.1% (good balance)

### **Minimum Confidence** (1% - 100%)
- **Lower** = Weak relationships OK
- **Higher** = Only strong relationships
- **Default:** 5% (reasonably confident)

### **Minimum Lift** (1.0 - 5.0)
- **1.0** = Random (no correlation)
- **2.0** = 2x more likely than random
- **5.0** = 5x more likely (very strong!)
- **Default:** 1.2 (positive correlation)

---

## File Structure in Your Project:

```
market_basket_analysis/
├── data/
│   └── indian/
│       ├── BigBasket.csv                    ← Product catalog (27K products)
│       └── bigbasket_transactions_5k.csv    ← Transaction data (20K transactions)
├── models/
│   └── indian/
│       ├── apriori_model.pkl               ← Trained Association Rules
│       ├── customer_segmentation.pkl       ← Trained K-Means
│       └── purchase_prediction_*.pkl       ← Trained Random Forests
├── src/
│   ├── analysis.py                         ← Apriori algorithm
│   ├── ml_models.py                        ← K-Means & Random Forest
│   └── utils.py                            ← Visualization & helpers
├── app.py                                  ← Streamlit web interface
└── train_models_indian.py                 ← Training script
```

---

## Quick Commands (Remember: VENV FIRST!):

```powershell
# Activate virtual environment
cd e:\workout_programs\vaishnav_ml\market_basket_analysis
.\venv\Scripts\Activate.ps1

# Retrain models
python train_models_indian.py

# Run the app
streamlit run app.py --server.port 8502

# Test models
python test_models_indian.py
```

---

## Common Questions:

**Q: Why only 200 products?**
A: Memory limit! With 27K products, the algorithm needs 10+ TB RAM. Filtering to top 200 keeps it under 2GB.

**Q: Why transaction_value = items × 100?**
A: Your CSV doesn't have prices, so we estimate ₹100 per item average for Indian retail.

**Q: What if I have 0 rules generated?**
A: Lower the minimum support (try 0.01%) or confidence (try 1%). Your data might not have strong patterns.

**Q: How accurate is purchase prediction?**
A: Depends on data! Your BigBasket models: 84-90% accuracy on popular products.

**Q: Can I use this for my store?**
A: YES! Just format your POS data as: `Transaction,Item` CSV

---

**🚀 Now go to http://localhost:8502 and try it!**
