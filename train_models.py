"""
Train and save ML models for Market Basket Analysis
Run this script to train models once, then load them in the app for fast predictions
"""

import pandas as pd
import sys
from pathlib import Path
import os

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from analysis import MarketBasketAnalyzer
from ml_models import CustomerSegmentation, PurchasePrediction


def train_market_basket_model(data_path, min_support=0.01, min_confidence=0.3, min_lift=1.0):
    """
    Train Apriori algorithm for market basket analysis
    
    Args:
        data_path: Path to transaction CSV file
        min_support: Minimum support threshold
        min_confidence: Minimum confidence threshold
        min_lift: Minimum lift threshold
    
    Returns:
        Trained analyzer with rules
    """
    print("\n" + "="*80)
    print("🛒 TRAINING MARKET BASKET ANALYSIS MODEL (APRIORI)")
    print("="*80)
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} transaction records")
    
    # Initialize analyzer
    analyzer = MarketBasketAnalyzer(
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift
    )
    
    # Process transactions
    transactions = df.groupby('Transaction')['Item'].apply(list).values.tolist()
    analyzer.transactions = transactions
    print(f"✅ Processed {len(transactions)} unique transactions")
    
    # Find patterns
    encoded_df = analyzer.encode_transactions()
    frequent_itemsets = analyzer.find_frequent_itemsets(encoded_df)
    rules = analyzer.generate_rules(metric='lift')
    
    print(f"\n📊 Results:")
    print(f"   - Frequent Itemsets: {len(frequent_itemsets)}")
    print(f"   - Association Rules: {len(rules)}")
    
    # Show top 3 rules
    print(f"\n🔝 Top 3 Rules (by Lift):")
    top_rules = analyzer.get_top_rules(n=3)
    for idx, row in top_rules.iterrows():
        ant = ', '.join(list(row['antecedents']))
        cons = ', '.join(list(row['consequents']))
        print(f"   {idx+1}. {ant} → {cons}")
        print(f"      Confidence: {row['confidence']:.2%}, Lift: {row['lift']:.2f}")
    
    return analyzer


def train_customer_segmentation(data_path, n_clusters=4):
    """
    Train K-Means clustering for customer segmentation
    
    Args:
        data_path: Path to transaction CSV file
        n_clusters: Number of customer segments
    
    Returns:
        Trained segmentation model
    """
    print("\n" + "="*80)
    print("👥 TRAINING CUSTOMER SEGMENTATION MODEL (K-MEANS)")
    print("="*80)
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Initialize model
    segmenter = CustomerSegmentation(n_clusters=n_clusters)
    
    # Create features
    customer_features = segmenter.create_customer_features(df, 'Transaction', 'Item')
    print(f"✅ Created features for {len(customer_features)} customers")
    
    # Train model
    segmenter.fit(customer_features)
    
    # Get segment profiles
    profiles = segmenter.get_segment_profiles()
    
    print(f"\n📊 Customer Segments:")
    for idx, row in profiles.iterrows():
        print(f"\n   Segment {idx}: {row['Segment_Name']}")
        print(f"      Avg Items: {row['num_items']:.1f}")
        print(f"      Avg Variety: {row['unique_categories']:.1f}")
        print(f"      Avg Value: ${row['transaction_value']:.2f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    segmenter.save_model('models/customer_segmentation.pkl')
    
    return segmenter


def train_purchase_prediction(data_path, target_product):
    """
    Train Random Forest for purchase prediction
    
    Args:
        data_path: Path to transaction CSV file
        target_product: Product to predict
    
    Returns:
        Trained prediction model
    """
    print("\n" + "="*80)
    print(f"🎯 TRAINING PURCHASE PREDICTION MODEL (RANDOM FOREST)")
    print(f"   Target Product: {target_product}")
    print("="*80)
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Initialize model
    predictor = PurchasePrediction()
    
    # Prepare data
    print("✅ Preparing training data...")
    X_train, X_test, y_train, y_test = predictor.prepare_training_data(
        df, target_product, 'Transaction', 'Item'
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Positive class ratio: {y_train.sum() / len(y_train):.2%}")
    
    # Train model
    print("\n🚀 Training Random Forest...")
    predictor.train(X_train, y_train)
    
    # Evaluate
    print("\n📊 Evaluating model...")
    report = predictor.evaluate(X_test, y_test)
    
    # Feature importance
    feature_importance = predictor.get_feature_importance(X_train.columns.tolist())
    print(f"\n🔍 Top 5 Features Predicting '{target_product}':")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"   {idx+1}. {row['Product']}: {row['Importance']:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    predictor.save_model(f'models/purchase_prediction_{target_product.lower().replace(" ", "_")}.pkl')
    
    return predictor


def main():
    """
    Main training pipeline
    """
    print("\n" + "="*80)
    print("🚀 MARKET BASKET ANALYSIS - MODEL TRAINING PIPELINE")
    print("="*80)
    
    # Configuration - Use real data if available!
    if os.path.exists('data/online_retail_1k.csv'):
        DATA_PATH = 'data/online_retail_1k.csv'
        print("\n✅ Using REAL DATASET: Online Retail (1K sample)")
    elif os.path.exists('data/online_retail.csv'):
        DATA_PATH = 'data/online_retail.csv'
        print("\n✅ Using REAL DATASET: Online Retail (FULL)")
    else:
        DATA_PATH = 'data/sample_transactions.csv'
        print("\n⚠️  Using sample data (download real data with: python download_datasets.py)")
    
    # Load data to determine top products
    df = pd.read_csv(DATA_PATH)
    top_products = df['Item'].value_counts().head(5).index.tolist()
    TARGET_PRODUCTS = top_products[:3]  # Train on top 3 products
    
    print(f"🎯 Target products for prediction: {TARGET_PRODUCTS}")
    
    # Check if data exists
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ Error: Data file not found at {DATA_PATH}")
        print("   Please ensure you have transaction data in the data/ folder")
        return
    
    print(f"\n📁 Data Source: {DATA_PATH}")
    print(f"   Transactions: {df['Transaction'].nunique():,}")
    print(f"   Unique Products: {df['Item'].nunique():,}")
    
    # 1. Train Market Basket Analysis
    try:
        # Adjust thresholds for real data (lower support needed)
        min_support = 0.005 if 'online_retail' in DATA_PATH else 0.05
        min_confidence = 0.2 if 'online_retail' in DATA_PATH else 0.3
        
        analyzer = train_market_basket_model(
            DATA_PATH,
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=1.2
        )
    except Exception as e:
        print(f"\n❌ Market Basket Analysis failed: {str(e)}")
    
    # 2. Train Customer Segmentation
    try:
        segmenter = train_customer_segmentation(DATA_PATH, n_clusters=4)
    except Exception as e:
        print(f"\n❌ Customer Segmentation failed: {str(e)}")
    
    # 3. Train Purchase Prediction Models
    for product in TARGET_PRODUCTS:
        try:
            predictor = train_purchase_prediction(DATA_PATH, product)
        except Exception as e:
            print(f"\n❌ Purchase Prediction for '{product}' failed: {str(e)}")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print("\n📦 Saved Models:")
    print("   - models/customer_segmentation.pkl")
    for product in TARGET_PRODUCTS:
        model_name = f"purchase_prediction_{product.lower().replace(' ', '_')}.pkl"
        print(f"   - models/{model_name}")
    
    print("\n🎯 Next Steps:")
    print("   1. Run 'python test_models.py' to test the trained models")
    print("   2. Or use the models in your Streamlit app for instant predictions!")
    print("\n")


if __name__ == "__main__":
    main()
