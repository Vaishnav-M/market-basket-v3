# 🛒 Market Basket Analysis - Complete Logic Explanation

## 📋 **1. INPUT CSV FORMAT**

Your CSV needs **exactly 2 columns**:

```csv
Transaction,Item
1,Coconut Oil
1,Tomato
1,Onion
2,Milk
2,Bread
2,Eggs
3,Coconut Oil
3,Milk
```

**Key Points:**
- **Transaction**: Unique ID for each shopping trip (can be numbers or strings)
- **Item**: Product name purchased in that transaction
- **One row per item** - If someone buys 3 items, you have 3 rows with the same Transaction ID

---

## 🧠 **2. THE THREE ML ALGORITHMS**

### **Algorithm 1: Association Rules (Apriori)**

**What it does:** Finds products that are frequently bought together

**The Logic:**
1. **Support** = How often do items appear together?
   - Example: Milk appears in 40 out of 100 transactions = 40% support
   
2. **Confidence** = If customer buys Item A, what % also buy Item B?
   - Example: 80% of people who buy Bread also buy Milk = 80% confidence
   
3. **Lift** = How much more likely to buy together vs random?
   - Lift > 1 = Positive correlation (buy together more than random)
   - Lift = 1 = No correlation
   - Lift < 1 = Negative correlation (don't buy together)

**Real Example from your data:**
```
Rule: {Coconut Oil} → {Curry Leaves}
Support: 2.5% (appears in 2.5% of all transactions)
Confidence: 65% (65% who buy Coconut Oil also buy Curry Leaves)
Lift: 3.2 (3.2x more likely to buy together than random)
```

**Business Use:**
- **Shelf Placement**: Put Coconut Oil near Curry Leaves
- **Promotions**: "Buy Coconut Oil, get 10% off Curry Leaves"
- **Recommendations**: Show Curry Leaves when customer adds Coconut Oil to cart

---

### **Algorithm 2: Customer Segmentation (K-Means Clustering)**

**What it does:** Groups customers into segments based on shopping behavior

**The Logic:**
1. **Create Features** for each customer (transaction):
   - `num_items`: How many items did they buy?
   - `unique_categories`: How diverse were their purchases?
   - `transaction_value`: Total value (₹100 per item average)

2. **K-Means Clustering**:
   - Finds natural groups of similar customers
   - Uses distance metrics (Euclidean distance)
   - Assigns each customer to a segment

3. **Segment Naming** (based on transaction value ranking):
   - **💎 VIP Shoppers**: High value + high variety
   - **🌟 Premium Shoppers**: Medium-high value + high variety
   - **🛒 Bulk Buyers**: High value + low variety
   - **⚡ Quick Shoppers**: Low value + few items
   - **🔍 Variety Seekers**: Medium value + high variety
   - **💰 Budget Shoppers**: Low value overall

**Example:**
```
Segment 0: VIP Shoppers
- Average Items: 15.2
- Product Variety: 12.8
- Transaction Value: ₹1,520
- Count: 150 customers

Segment 1: Quick Shoppers
- Average Items: 2.1
- Product Variety: 2.0
- Transaction Value: ₹210
- Count: 2,450 customers
```

**Business Use:**
- **Personalized Marketing**: Send VIP customers premium product offers
- **Loyalty Programs**: Different rewards for different segments
- **Store Layout**: Optimize for majority segment (Quick Shoppers)

---

### **Algorithm 3: Purchase Prediction (Random Forest)**

**What it does:** Predicts if a customer will buy a specific product based on their current basket

**The Logic:**
1. **Feature Engineering**:
   - Convert each transaction into a binary feature vector
   - Example: `{Milk: 1, Bread: 1, Eggs: 0, Butter: 0, ...}`
   - 1 = item in basket, 0 = not in basket

2. **Target Variable**:
   - Did they buy the target product? (Yes=1, No=0)

3. **Random Forest**:
   - Trains 100 decision trees
   - Each tree votes on prediction
   - Final prediction = majority vote
   - Also provides probability (confidence %)

**Example:**
```
Target Product: Coconut Oil

Training Data:
Transaction 1: [Tomato=1, Onion=1, Curry Leaves=1] → Coconut Oil=1 ✅
Transaction 2: [Milk=1, Bread=1, Eggs=1] → Coconut Oil=0 ❌
Transaction 3: [Onion=1, Curry Leaves=1, Ginger=1] → Coconut Oil=1 ✅

Model learns: If customer has {Curry Leaves + Onion}, 85% chance they'll buy Coconut Oil
```

**Business Use:**
- **Real-time Recommendations**: "Based on your cart, you might need Coconut Oil"
- **Cross-selling**: Suggest complementary products at checkout
- **Inventory**: Predict demand based on current sales patterns

---

## 🔄 **3. THE COMPLETE WORKFLOW**

### **Step 1: Data Upload**
```python
# Your CSV is uploaded
df = pd.read_csv("bigbasket_transactions.csv")

# Columns: Transaction, Item
# 579,120 rows (20,000 transactions)
```

### **Step 2: Data Filtering**
```python
# To avoid memory issues, filter to top 200 most popular products
product_counts = df['Item'].value_counts()
top_200 = product_counts.head(200)

# Filter dataset
df_filtered = df[df['Item'].isin(top_200)]
# Result: Using 23% of records with most popular items
```

### **Step 3: Transform to Transaction Format**
```python
# Convert to list of lists (one list per transaction)
transactions = [
    ['Coconut Oil', 'Tomato', 'Onion'],        # Transaction 1
    ['Milk', 'Bread', 'Eggs'],                  # Transaction 2
    ['Coconut Oil', 'Curry Leaves', 'Ginger']   # Transaction 3
]
```

### **Step 4: Run Apriori Algorithm**
```python
# Find frequent itemsets
itemsets = apriori(
    encoded_transactions,
    min_support=0.001,  # 0.1% minimum
    use_colnames=True
)

# Generate rules
rules = association_rules(
    itemsets,
    metric='lift',
    min_threshold=1.2
)

# Result: 60 itemsets, 6 rules found
```

### **Step 5: Run K-Means Clustering**
```python
# Create customer features
customer_features = {
    'Transaction 1': {num_items: 3, unique_categories: 3, value: 300},
    'Transaction 2': {num_items: 2, unique_categories: 2, value: 200}
}

# Cluster customers
kmeans = KMeans(n_clusters=4)
clusters = kmeans.fit_predict(customer_features)

# Result: 4 segments identified
```

### **Step 6: Train Random Forest**
```python
# Prepare binary features
X = [[1, 1, 0, 1, 0],  # Transaction 1: has tomato, onion, curry leaves
     [0, 0, 1, 1, 1],  # Transaction 2: has milk, bread, eggs
     ...]

y = [1, 0, ...]  # Did they buy target product?

# Train model
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)

# Predict
probability = rf.predict_proba([1, 0, 1, 0, 0])  # New basket
# Result: 85% chance they'll buy target product
```

---

## 📊 **4. THE MATH BEHIND IT**

### **Support Formula**
```
Support(A) = Count(transactions with A) / Total transactions

Example:
- Coconut Oil appears in 500 out of 20,000 transactions
- Support = 500/20,000 = 0.025 = 2.5%
```

### **Confidence Formula**
```
Confidence(A → B) = Support(A ∪ B) / Support(A)

Example:
- {Coconut Oil + Curry Leaves} appears in 325 transactions
- {Coconut Oil} appears in 500 transactions
- Confidence = 325/500 = 0.65 = 65%
```

### **Lift Formula**
```
Lift(A → B) = Support(A ∪ B) / (Support(A) × Support(B))

Example:
- Support(Coconut Oil + Curry Leaves) = 0.01625 (325/20,000)
- Support(Coconut Oil) = 0.025 (500/20,000)
- Support(Curry Leaves) = 0.02 (400/20,000)
- Lift = 0.01625 / (0.025 × 0.02) = 3.25

Interpretation: 3.25x more likely to buy together than if independent
```

### **K-Means Distance**
```
Euclidean Distance = √[(x₁-y₁)² + (x₂-y₂)² + (x₃-y₃)²]

Example (2 customers):
Customer A: [num_items=5, variety=4, value=500]
Customer B: [num_items=2, variety=2, value=200]

Distance = √[(5-2)² + (4-2)² + (500-200)²]
         = √[9 + 4 + 90000]
         = √90013 = 300.02

→ These customers are very different (far apart)
```

---

## 🎯 **5. BUSINESS VALUE**

### **For Retailers:**
1. **Increase Sales**: Place related products near each other
2. **Optimize Layout**: Understand customer shopping patterns
3. **Targeted Marketing**: Send personalized offers to each segment
4. **Inventory Management**: Stock products that sell together

### **For E-commerce:**
1. **Product Recommendations**: "Customers who bought X also bought Y"
2. **Smart Search**: Show complementary products
3. **Dynamic Pricing**: Bundle deals for high-lift pairs
4. **Cart Abandonment**: Predict and prevent with targeted offers

### **Real ROI Example:**
```
Before: Random product placement
- Average transaction value: ₹250
- Cross-sell rate: 15%

After: Using Market Basket Analysis
- Strategic placement of high-lift pairs
- Average transaction value: ₹320 (+28%)
- Cross-sell rate: 32% (+113%)

For a store with 1,000 daily customers:
Daily revenue increase = 1000 × (320 - 250) = ₹70,000
Monthly revenue increase = ₹2,100,000
```

---

## 🚀 **6. HOW TO USE THE APP**

1. **Upload CSV** (Transaction, Item columns)
2. **Adjust Parameters**:
   - Support: 0.01% - 5% (lower = more results)
   - Confidence: 1% - 100% (higher = stronger rules)
   - Lift: 1.0 - 5.0 (higher = stronger correlation)
3. **Run Analysis**:
   - Tab 1: Association Rules
   - Tab 2: Customer Segmentation
   - Tab 3: Purchase Prediction
4. **Get Insights**:
   - Product placement recommendations
   - Customer segment profiles
   - Purchase probability predictions

---

## 📚 **7. KEY TERMS GLOSSARY**

- **Itemset**: A group of items (e.g., {Milk, Bread})
- **Frequent Itemset**: Itemset appearing in >= min_support transactions
- **Association Rule**: Pattern like A → B (if buy A, then buy B)
- **Clustering**: Grouping similar items/customers together
- **Feature Vector**: Numerical representation of data for ML
- **Silhouette Score**: Measure of cluster quality (0-1, higher is better)
- **Precision**: % of positive predictions that were correct
- **Recall**: % of actual positives that were caught
- **F1-Score**: Harmonic mean of precision and recall

---

**Built with ❤️ for Indian Retail Markets**
*Using real BigBasket data with 27,000 products*
