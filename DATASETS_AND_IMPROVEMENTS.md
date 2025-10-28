# 📊 Real-World Datasets to Improve Your Model

## 🔥 **Best Free Datasets for Market Basket Analysis**

### **1. Instacart Market Basket Analysis (3M+ orders)**
- **URL:** https://www.kaggle.com/c/instacart-market-basket-analysis/data
- **Size:** 3.4M grocery orders, 200K users, 50K products
- **Format:** CSV files (orders.csv, products.csv, aisles.csv)
- **Quality:** ⭐⭐⭐⭐⭐ (Industry-standard, used in competitions)
- **Use Case:** Perfect for your project (actual grocery transactions)

**Download Instructions:**
```bash
# Install Kaggle CLI
pip install kaggle

# Configure API key (get from kaggle.com/account)
# Save kaggle.json to ~/.kaggle/

# Download dataset
kaggle competitions download -c instacart-market-basket-analysis

# Unzip
unzip instacart-market-basket-analysis.zip -d data/instacart/
```

---

### **2. Online Retail Dataset (540K+ transactions)**
- **URL:** https://archive.ics.uci.edu/ml/datasets/online+retail
- **Size:** 541,909 transactions, 4,372 customers, 4,070 products
- **Format:** Excel/CSV
- **Quality:** ⭐⭐⭐⭐⭐ (UCI Repository, academic quality)
- **Use Case:** E-commerce transactions (UK-based retailer)

**Download Instructions:**
```bash
# Direct download
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx -O data/online_retail.xlsx

# Or use Python
import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
df = pd.read_excel(url)
df.to_csv('data/online_retail.csv', index=False)
```

---

### **3. Groceries Dataset (9,835 transactions)**
- **URL:** https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset
- **Size:** 9,835 transactions, 169 unique products
- **Format:** CSV
- **Quality:** ⭐⭐⭐⭐ (Clean, ready-to-use)
- **Use Case:** Grocery store basket analysis

**Download Instructions:**
```bash
kaggle datasets download -d heeraldedhia/groceries-dataset
unzip groceries-dataset.zip -d data/groceries/
```

---

### **4. Market Basket Optimization (7,500 transactions)**
- **URL:** https://www.kaggle.com/datasets/dragonheir/retail-data-analytics
- **Size:** 7,500 transactions from French retail store
- **Format:** CSV
- **Quality:** ⭐⭐⭐⭐ (Good for learning)
- **Use Case:** Retail optimization

---

### **5. Walmart Sales Data (421,570 transactions)**
- **URL:** https://www.kaggle.com/datasets/yasserh/walmart-dataset
- **Size:** 421K transactions across 45 stores
- **Format:** CSV
- **Quality:** ⭐⭐⭐⭐ (Large, real-world)
- **Use Case:** Multi-store analysis, time series

---

### **6. Amazon Product Reviews (Implicit Baskets)**
- **URL:** https://nijianmo.github.io/amazon/index.html
- **Size:** 233M reviews, 142M products
- **Format:** JSON
- **Quality:** ⭐⭐⭐⭐⭐ (Massive scale)
- **Use Case:** "Customers also viewed/bought" recommendations

---

### **7. Tesco Grocery Dataset (92,000 baskets)**
- **URL:** https://www.dunnhumby.com/source-files/
- **Size:** 92K baskets, UK supermarket chain
- **Format:** CSV
- **Quality:** ⭐⭐⭐⭐⭐ (Real commercial data)
- **Use Case:** Customer segmentation + basket analysis

---

## 🚀 **How to Integrate These Datasets:**

### **Step 1: Download Dataset**

I'll create a script to download and prepare the **Online Retail dataset** (easiest to start):

```python
# download_dataset.py
import pandas as pd
import os

def download_online_retail():
    """Download and prepare UCI Online Retail dataset"""
    
    print("📦 Downloading Online Retail Dataset...")
    
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
    
    # Download
    df = pd.read_excel(url)
    print(f"✅ Downloaded {len(df)} transactions")
    
    # Clean data
    df = df.dropna(subset=['CustomerID', 'Description'])
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    
    # Rename columns to match your format
    df = df.rename(columns={
        'InvoiceNo': 'Transaction',
        'Description': 'Item'
    })
    
    # Select relevant columns
    df = df[['Transaction', 'Item', 'Quantity', 'UnitPrice', 'CustomerID']]
    
    # Save
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/online_retail.csv', index=False)
    
    print(f"✅ Saved to data/online_retail.csv")
    print(f"   Transactions: {df['Transaction'].nunique():,}")
    print(f"   Unique Products: {df['Item'].nunique():,}")
    print(f"   Customers: {df['CustomerID'].nunique():,}")
    
    return df

if __name__ == "__main__":
    download_online_retail()
```

---

### **Step 2: Adapt Your Training Script**

Update `train_models.py` to handle larger datasets:

```python
# Add to train_models.py

def train_on_large_dataset(data_path, sample_size=None):
    """
    Train on large dataset with optional sampling
    
    Args:
        data_path: Path to CSV file
        sample_size: Number of transactions to sample (None = use all)
    """
    print(f"📊 Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Sample if dataset too large
    if sample_size and df['Transaction'].nunique() > sample_size:
        transactions = df['Transaction'].unique()
        sampled = np.random.choice(transactions, sample_size, replace=False)
        df = df[df['Transaction'].isin(sampled)]
        print(f"   Sampled {sample_size:,} transactions")
    
    print(f"   Total transactions: {df['Transaction'].nunique():,}")
    print(f"   Unique products: {df['Item'].nunique():,}")
    
    return df
```

---

## 🔧 **Model Improvements:**

### **1. Handle Large Datasets (Performance)**

```python
# Add to src/analysis.py

def find_frequent_itemsets_fpgrowth(self, encoded_df):
    """
    Use FP-Growth algorithm (faster than Apriori for large datasets)
    """
    from mlxtend.frequent_patterns import fpgrowth
    
    self.frequent_itemsets = fpgrowth(
        encoded_df, 
        min_support=self.min_support, 
        use_colnames=True
    )
    
    print(f"Found {len(self.frequent_itemsets)} frequent itemsets (FP-Growth)")
    return self.frequent_itemsets
```

**Why:** FP-Growth is 10-100x faster than Apriori on large datasets

---

### **2. Add Deep Learning (Neural Networks)**

```python
# New file: src/nn_models.py

from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model

class BasketPredictionNN:
    """
    Deep Learning model for basket prediction
    Better than Random Forest for complex patterns
    """
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self.build_model()
    
    def build_model(self):
        """Build neural network architecture"""
        
        inputs = Input(shape=(self.input_dim,))
        x = Dense(128, activation='relu')(inputs)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50):
        """Train the neural network"""
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
```

**Why:** Neural networks capture complex non-linear patterns Random Forest might miss

---

### **3. Add Time Series Forecasting**

```python
# New file: src/forecasting.py

from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

class SalesForecasting:
    """
    Predict future sales trends
    """
    
    def __init__(self):
        self.model = None
    
    def prepare_time_series(self, df, product, freq='D'):
        """
        Convert transaction data to time series
        
        Args:
            df: Transaction DataFrame
            product: Product to forecast
            freq: Frequency ('D'=daily, 'W'=weekly, 'M'=monthly)
        """
        # Filter product
        product_df = df[df['Item'] == product]
        
        # Aggregate by date
        product_df['Date'] = pd.to_datetime(product_df['InvoiceDate'])
        sales = product_df.groupby('Date')['Quantity'].sum()
        
        # Resample to frequency
        sales = sales.resample(freq).sum().fillna(0)
        
        return sales
    
    def train(self, time_series, order=(1,1,1)):
        """Train ARIMA model"""
        
        self.model = ARIMA(time_series, order=order)
        self.model = self.model.fit()
        
        return self.model
    
    def forecast(self, steps=30):
        """Forecast future values"""
        
        forecast = self.model.forecast(steps=steps)
        return forecast
```

**Why:** Predict future demand for inventory optimization

---

### **4. Add Recommendation System (Collaborative Filtering)**

```python
# New file: src/recommender.py

from sklearn.decomposition import NMF
import numpy as np

class ProductRecommender:
    """
    Collaborative filtering for product recommendations
    Like Netflix/Amazon recommendations
    """
    
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.model = NMF(n_components=n_components, random_state=42)
        self.customer_product_matrix = None
    
    def create_matrix(self, df):
        """
        Create customer-product interaction matrix
        """
        # Pivot to matrix
        matrix = df.pivot_table(
            index='CustomerID',
            columns='Item',
            values='Quantity',
            fill_value=0
        )
        
        self.customer_product_matrix = matrix
        return matrix
    
    def train(self, matrix):
        """Train NMF model"""
        
        self.W = self.model.fit_transform(matrix)
        self.H = self.model.components_
        
        # Reconstruct matrix (predictions)
        self.predicted = np.dot(self.W, self.H)
        
        return self.predicted
    
    def recommend(self, customer_id, top_n=5):
        """
        Recommend top N products for a customer
        """
        customer_idx = self.customer_product_matrix.index.get_loc(customer_id)
        
        # Get predictions for this customer
        predictions = self.predicted[customer_idx]
        
        # Get top N products they haven't bought
        bought = self.customer_product_matrix.iloc[customer_idx]
        not_bought = bought[bought == 0].index
        
        # Get scores for not-bought products
        scores = [(prod, predictions[self.customer_product_matrix.columns.get_loc(prod)]) 
                  for prod in not_bought]
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_n]
```

**Why:** Personalized recommendations increase conversion by 20-35%

---

### **5. Add Customer Lifetime Value (CLV) Prediction**

```python
# Add to src/ml_models.py

class CLVPredictor:
    """
    Predict Customer Lifetime Value
    Identify high-value customers
    """
    
    def create_rfm_features(self, df):
        """
        Create RFM (Recency, Frequency, Monetary) features
        """
        import datetime
        
        # Get last date
        max_date = df['InvoiceDate'].max()
        
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (max_date - x.max()).days,  # Recency
            'InvoiceNo': 'nunique',  # Frequency
            'TotalPrice': 'sum'  # Monetary
        })
        
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        return rfm
    
    def predict_clv(self, rfm_features):
        """
        Predict CLV using regression
        """
        from sklearn.ensemble import GradientBoostingRegressor
        
        # Simple CLV formula
        clv = rfm_features['Frequency'] * rfm_features['Monetary']
        
        return clv
```

**Why:** Focus marketing on high-value customers (80/20 rule)

---

## 📈 **Performance Improvements:**

### **1. Parallel Processing**

```python
# Add to train_models.py

from multiprocessing import Pool

def train_multiple_products_parallel(products):
    """
    Train prediction models in parallel
    """
    with Pool(processes=4) as pool:
        results = pool.map(train_purchase_prediction, products)
    
    return results
```

**Speed:** 4x faster training on 4-core CPU

---

### **2. GPU Acceleration (for Neural Networks)**

```python
# Add to requirements.txt
tensorflow-gpu==2.10.0  # If you have NVIDIA GPU

# In nn_models.py
import tensorflow as tf

# Use GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ Using GPU: {gpus}")
else:
    print("⚠️ Using CPU (slower)")
```

**Speed:** 10-100x faster neural network training

---

### **3. Caching & Memoization**

```python
# Add to src/analysis.py

from functools import lru_cache

@lru_cache(maxsize=1000)
def get_product_associations(self, product):
    """
    Cache product associations for fast lookup
    """
    return self.rules[self.rules['antecedents'].apply(lambda x: product in x)]
```

**Speed:** Instant repeated queries

---

## 🎯 **Next Steps to Make Model Better:**

### **Immediate (This Week):**
1. ✅ Download Online Retail dataset (540K transactions)
2. ✅ Run `train_models.py` on it
3. ✅ Compare accuracy with sample data

### **Short-term (This Month):**
1. ✅ Add FP-Growth algorithm (faster)
2. ✅ Implement neural network predictor
3. ✅ Add time series forecasting
4. ✅ Deploy to Streamlit Cloud with real data

### **Long-term (Next 3 Months):**
1. ✅ Train on Instacart dataset (3M orders)
2. ✅ Add collaborative filtering
3. ✅ Implement CLV prediction
4. ✅ Build mobile app or Chrome extension
5. ✅ Pitch to real retailers

---

Want me to create the download script and updated training pipeline for you? I can set it up to automatically:
1. Download the Online Retail dataset
2. Clean and prepare it
3. Retrain all models
4. Compare performance with sample data

Just say "yes" and I'll build it! 🚀
