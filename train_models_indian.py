"""
Training pipeline for Indian Supermarket Market Basket Analysis
Trains models on Indian shopping behavior patterns
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from analysis import MarketBasketAnalyzer
from ml_models import CustomerSegmentation, PurchasePrediction

# Configuration
DATA_PATH = 'data/indian/bigbasket_transactions_5k.csv'  # Use 5K dataset for good balance
MODELS_DIR = 'models/indian/'

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Indian-specific thresholds (filtered to top 1000 products)
# Note: Synthetic data has random product combinations, so support must be very low
MIN_SUPPORT = 0.001  # 0.1% = appears in at least 14 transactions (out of 13,725)
MIN_CONFIDENCE = 0.05  # 5% confidence  
MIN_LIFT = 1.2  # 1.2x lift threshold

def main():
    print("=" * 80)
    print("🇮🇳 INDIAN SUPERMARKET MARKET BASKET ANALYSIS - TRAINING PIPELINE")
    print("=" * 80)
    
    # Load data
    print(f"\n📁 Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    print(f"   Total Records: {len(df):,}")
    print(f"   Unique Transactions: {df['Transaction'].nunique():,}")
    print(f"   Unique Products: {df['Item'].nunique():,}")
    
    # **FILTER TO TOP 200 PRODUCTS** (balance between diversity and performance)
    # Focus on most popular products that appear frequently in transactions
    product_counts = df['Item'].value_counts()
    top_n_products = 200  # Top 200 most popular Indian products
    popular_products = product_counts.head(top_n_products).index
    
    print(f"\n🎯 Filtering to top {top_n_products} most popular Indian products:")
    print(f"   Before: {df['Item'].nunique():,} products")
    original_records = len(df)
    df = df[df['Item'].isin(popular_products)]
    print(f"   After: {df['Item'].nunique():,} products")
    print(f"   Kept {len(df):,} / {original_records:,} records ({len(df)/original_records*100:.1f}%)")
    print(f"\n   These top {top_n_products} products account for the majority of sales!")
    
    # Show Indian-specific stats (handle optional columns)
    if 'ShoppingPattern' in df.columns:
        print(f"\n📊 Shopping Pattern Distribution:")
        for pattern, count in df.groupby('ShoppingPattern')['Transaction'].nunique().items():
            pct = (count / df['Transaction'].nunique()) * 100
            print(f"      {pattern}: {count:,} ({pct:.1f}%)")
    
    if 'Region' in df.columns:
        print(f"\n🌍 Regional Distribution:")
        for region, count in df.groupby('Region')['Transaction'].nunique().items():
            pct = (count / df['Transaction'].nunique()) * 100
            print(f"      {region}: {count:,} ({pct:.1f}%)")
    
    # Get top products for prediction
    top_products = df['Item'].value_counts().head(5).index.tolist()
    print(f"\n🎯 Top Indian Products (for prediction):")
    for i, product in enumerate(top_products[:10], 1):
        count = df[df['Item'] == product]['Transaction'].nunique()
        print(f"      {i}. {product}: {count:,} transactions")
    
    print(f"\n🎯 Training models on top 3: {top_products[:3]}")
    
    # ============================================================================
    # 1. MARKET BASKET ANALYSIS (APRIORI)
    # ============================================================================
    print("\n" + "=" * 80)
    print("🛒 TRAINING MARKET BASKET ANALYSIS MODEL (APRIORI)")
    print("=" * 80)
    
    analyzer = MarketBasketAnalyzer(min_support=MIN_SUPPORT, min_confidence=MIN_CONFIDENCE)
    
    # Process transactions
    print(f"✅ Loading {len(df)} transaction records...")
    transactions = df.groupby('Transaction')['Item'].apply(list).values.tolist()
    analyzer.transactions = transactions
    print(f"✅ Processed {len(transactions)} unique transactions")
    
    # Find patterns
    encoded_df = analyzer.encode_transactions()
    frequent_itemsets = analyzer.find_frequent_itemsets(encoded_df)
    rules = analyzer.generate_rules(metric='lift')
    
    # Show results
    print(f"\n📊 Results:")
    print(f"   - Frequent Itemsets: {len(analyzer.frequent_itemsets)}")
    print(f"   - Association Rules: {len(rules)}")
    
    if len(rules) > 0:
        print(f"\n🔝 Top 5 Indian Product Association Rules (by Lift):")
        top_rules = rules.nlargest(5, 'lift')
        for idx, rule in top_rules.iterrows():
            antecedents = ', '.join(list(rule['antecedents']))
            consequents = ', '.join(list(rule['consequents']))
            print(f"   {idx}. {antecedents} → {consequents}")
            print(f"      Confidence: {rule['confidence']*100:.2f}%, Lift: {rule['lift']:.2f}")
    else:
        print(f"\n⚠️  No association rules found. Try lowering MIN_CONFIDENCE or MIN_SUPPORT")
    
    # Save results (rules and frequent itemsets)
    import pickle
    with open(f'{MODELS_DIR}market_basket_analysis.pkl', 'wb') as f:
        pickle.dump({
            'rules': rules,
            'frequent_itemsets': analyzer.frequent_itemsets,
            'min_support': MIN_SUPPORT,
            'min_confidence': MIN_CONFIDENCE
        }, f)
    print(f"Model saved to {MODELS_DIR}market_basket_analysis.pkl")
    
    # ============================================================================
    # 2. CUSTOMER SEGMENTATION (K-MEANS)
    # ============================================================================
    print("\n" + "=" * 80)
    print("👥 TRAINING CUSTOMER SEGMENTATION MODEL (K-MEANS)")
    print("=" * 80)
    
    segmentation = CustomerSegmentation(n_clusters=5)  # 5 segments for Indian customers
    
    # Create features
    customer_features = segmentation.create_customer_features(df, 'Transaction', 'Item')
    print(f"✅ Created features for {len(customer_features)} customers")
    
    # Train model
    segments = segmentation.fit(customer_features)
    print(f"✅ Customer segmentation complete!")
    
    # Show segments
    print(f"\n📊 Indian Customer Segments:\n")
    profiles = segmentation.get_segment_profiles()
    for _, segment in profiles.iterrows():
        print(f"   Segment {segment['Segment_ID']}: {segment['Segment_Name']}")
        print(f"      Avg Items: {segment['num_items']:.1f}")
        print(f"      Avg Variety: {segment['unique_categories']:.1f}")
        print(f"      Avg Value: ${segment['transaction_value']:.2f}\n")
    
    # Save model
    segmentation.save_model(f'{MODELS_DIR}customer_segmentation.pkl')
    
    # ============================================================================
    # 3. PURCHASE PREDICTION (RANDOM FOREST) - Top 3 Indian Products
    # ============================================================================
    
    for i, product in enumerate(top_products[:3]):
        print("\n" + "=" * 80)
        print(f"🎯 TRAINING PURCHASE PREDICTION MODEL (RANDOM FOREST)")
        print(f"   Target Product: {product}")
        print("=" * 80)
        
        predictor = PurchasePrediction()
        
        # Prepare training data
        print(f"\n✅ Preparing training data...")
        X_train, X_test, y_train, y_test = predictor.prepare_training_data(
            df, product, transaction_col='Transaction', item_col='Item'
        )
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Positive class ratio: {y_train.mean()*100:.2f}%")
        
        # Train model
        print(f"\n🚀 Training Random Forest...")
        predictor.train(X_train, y_train)
        
        # Evaluate (this method prints metrics internally)
        print(f"\n📊 Evaluating model...")
        metrics = predictor.evaluate(X_test, y_test)
        
        # Feature importance
        feature_importance_df = predictor.get_feature_importance(X_train.columns.tolist())
        top_features = feature_importance_df.head(5)
        print(f"\n🔍 Top 5 Indian Products Predicting '{product}':")
        for idx, row in top_features.iterrows():
            print(f"   {idx+1}. {row['Product']}: {row['Importance']:.4f}")
        
        # Save model
        safe_filename = product.lower().replace(' ', '_').replace('/', '_')
        predictor.save_model(f'{MODELS_DIR}purchase_prediction_{safe_filename}.pkl')
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    
    print(f"\n📦 Saved Indian Market Models:")
    print(f"   - {MODELS_DIR}market_basket_analysis.pkl")
    print(f"   - {MODELS_DIR}customer_segmentation.pkl")
    for product in top_products[:3]:
        safe_filename = product.lower().replace(' ', '_').replace('/', '_')
        print(f"   - {MODELS_DIR}purchase_prediction_{safe_filename}.pkl")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Run 'python test_models_indian.py' to test the trained models")
    print(f"   2. Or launch the Indian Streamlit app: 'streamlit run app_indian.py'")


if __name__ == '__main__':
    main()
