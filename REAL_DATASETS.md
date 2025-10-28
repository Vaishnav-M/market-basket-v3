# 🔍 Real Transaction Datasets for Market Basket Analysis

## ✅ **Free Public Datasets Available:**

### **1. Instacart Market Basket Analysis Dataset** ⭐ RECOMMENDED
- **Source**: Kaggle (Official Instacart)
- **Link**: https://www.kaggle.com/c/instacart-market-basket-analysis/data
- **Size**: 3+ million grocery orders from 200,000+ users
- **Format**: Multiple CSVs with orders, products, departments
- **Country**: USA (but has Indian-style products too)
- **License**: Open source for research/learning
- **Quality**: ⭐⭐⭐⭐⭐ (Real customer data, anonymized)

**Files Included:**
```
orders.csv          - 3.4M orders
order_products.csv  - 32M rows of products purchased
products.csv        - 50K products
aisles.csv          - Product categories
departments.csv     - Store departments
```

**Sample Data:**
```csv
order_id,product_id,add_to_cart_order,reordered
1,49302,1,1
1,11109,2,1
1,10246,3,0
```

**How to Use:**
1. Download from Kaggle (requires free account)
2. Extract to `data/instacart/`
3. Run conversion script (I can create one for you!)
4. Train models on real data!

---

### **2. Online Retail Dataset (UCI)**
- **Source**: UCI Machine Learning Repository
- **Link**: https://archive.ics.uci.edu/ml/datasets/online+retail
- **Size**: 541,909 transactions from UK online retailer (2010-2011)
- **Format**: Single Excel/CSV file
- **Country**: UK
- **License**: Public domain
- **Quality**: ⭐⭐⭐⭐ (Real e-commerce data)

**Columns:**
```
InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
```

**Sample:**
```csv
InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country
536365,85123A,WHITE HANGING HEART T-LIGHT HOLDER,6,2010-12-01 08:26:00,2.55,17850,United Kingdom
536365,71053,WHITE METAL LANTERN,6,2010-12-01 08:26:00,3.39,17850,United Kingdom
```

---

### **3. Groceries Dataset (Kaggle)**
- **Source**: Kaggle
- **Link**: https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset
- **Size**: 38,765 transactions
- **Format**: CSV
- **Quality**: ⭐⭐⭐ (Good for learning)

**Columns:**
```
Member_number, Date, itemDescription
```

**Sample:**
```csv
Member_number,Date,itemDescription
1808,21-07-2015,tropical fruit
1808,21-07-2015,yogurt
1808,21-07-2015,coffee
```

---

### **4. The Bread Basket Dataset**
- **Source**: Kaggle
- **Link**: https://www.kaggle.com/datasets/xvivancos/transactions-from-a-bakery
- **Size**: 21,293 transactions from Edinburgh bakery
- **Format**: CSV
- **Quality**: ⭐⭐⭐ (Small but real)

**Columns:**
```
Transaction,Item
```

**Sample:**
```csv
Transaction,Item
1,Bread
1,Scandinavian
1,Scandinavian
```

---

### **5. Walmart Sales Dataset**
- **Source**: Kaggle
- **Link**: https://www.kaggle.com/datasets/mikhail1681/walmart-sales
- **Size**: 421,570 rows
- **Format**: CSV
- **Quality**: ⭐⭐⭐⭐

---

## 🇮🇳 **India-Specific Datasets:**

### **1. BigBasket Products + JioMart (Hybrid Approach)**
Since BigBasket transaction data isn't public, here's what we can do:

**Option A: Web Scraping (Legal Gray Area)**
- Scrape public BigBasket/JioMart product listings
- Generate realistic Indian baskets based on:
  - Regional preferences (North/South/East/West India)
  - Festival shopping patterns
  - Weekly grocery patterns

**Option B: Request from Startups**
- Reach out to Indian grocery startups
- Many share anonymized data for research
- Try: Dunzo, Grofers, Zepto (for academic projects)

### **2. Indian E-commerce Datasets**
- **Flipkart/Amazon India**: Sometimes release anonymized data
- **Government Open Data**: https://data.gov.in (check for retail data)

---

## 🎯 **BEST OPTION FOR YOU:**

### **Use Instacart Dataset!** ⭐

**Why:**
1. ✅ **Largest**: 3M+ real transactions
2. ✅ **High Quality**: Real customer data
3. ✅ **Well Structured**: Perfect for market basket analysis
4. ✅ **Has Indian Products**: Includes spices, rice, lentils, etc.
5. ✅ **Free & Legal**: Open for research/learning

**How to Get It:**

```bash
# 1. Create Kaggle account (free)
# 2. Go to: https://www.kaggle.com/c/instacart-market-basket-analysis
# 3. Click "Download All"
# 4. Extract to: data/instacart/
```

---

## 🚀 **I Can Create a Converter Script!**

Would you like me to create a script that:
1. Downloads Instacart dataset
2. Converts it to our format (Transaction, Item)
3. Filters for Indian-style products
4. Trains models on REAL data

**Just say "yes" and I'll create it for you!**

---

## 📊 **Dataset Comparison:**

| Dataset | Size | Quality | Indian Products | Free | Format |
|---------|------|---------|----------------|------|--------|
| **Instacart** | 3M+ | ⭐⭐⭐⭐⭐ | Some | ✅ | CSV |
| **Online Retail** | 541K | ⭐⭐⭐⭐ | No | ✅ | CSV |
| **Groceries** | 38K | ⭐⭐⭐ | Some | ✅ | CSV |
| **Bread Basket** | 21K | ⭐⭐⭐ | No | ✅ | CSV |
| **Our Synthetic** | 20K | ⭐⭐⭐ | ✅ All | ✅ | CSV |

---

## 💡 **My Recommendation:**

### **Short Term (Learning):**
✅ Use our **synthetic BigBasket data** (already working!)

### **Medium Term (Better Learning):**
✅ Download **Instacart dataset** (3M real transactions)
- I'll create converter script
- Train on real customer behavior
- Compare results with synthetic data

### **Long Term (Production):**
✅ Partner with an Indian startup for real anonymized data
- OR use your own store's POS data
- OR combine multiple public datasets

---

**Want me to create the Instacart converter script now?** 🚀
