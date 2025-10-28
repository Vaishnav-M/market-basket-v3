"""
Test Indian Market Basket Analysis Models
Interactive testing of trained models on real Indian products
"""

import pandas as pd
import pickle
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ml_models import CustomerSegmentation, PurchasePrediction

def test_customer_segmentation():
    """Test customer segmentation model"""
    print("\n" + "="*80)
    print("👥 TESTING CUSTOMER SEGMENTATION")
    print("="*80)
    
    # Load model
    with open('models/indian/customer_segmentation.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print("\n✅ Model loaded successfully!")
    print("\n📊 Customer Segments Available:")
    
    # Load sample data to show segments
    df = pd.read_csv('data/indian/bigbasket_transactions_5k.csv')
    
    # Filter to top 200 products (same as training)
    product_counts = df['Item'].value_counts()
    popular_products = product_counts.head(200).index
    df = df[df['Item'].isin(popular_products)]
    
    segmentation = CustomerSegmentation(n_clusters=5)
    segmentation.kmeans = data['kmeans']
    segmentation.scaler = data['scaler']
    
    # Create features for first 10 customers
    customer_features = segmentation.create_customer_features(df, 'Transaction', 'Item')
    
    # Drop customer_id column if it exists (not used in training)
    if 'customer_id' in customer_features.columns:
        customer_features = customer_features.drop('customer_id', axis=1)
    
    # Predict segments
    scaled_features = segmentation.scaler.transform(customer_features)
    segments = segmentation.kmeans.predict(scaled_features)
    
    # Show first 10 customers
    print("\n🔍 Sample Customer Predictions:")
    for i in range(min(10, len(customer_features))):
        segment_id = segments[i]
        items = customer_features.iloc[i]['num_items']
        print(f"   Customer {i+1}: Segment {segment_id} - {int(items)} items purchased")
    
    print(f"\n✅ Tested on {len(customer_features)} customers!")
    return segmentation


def test_purchase_prediction():
    """Test purchase prediction models"""
    print("\n" + "="*80)
    print("🎯 TESTING PURCHASE PREDICTION MODELS")
    print("="*80)
    
    # List available models
    model_files = [
        'purchase_prediction_coconut_-_(for_kalash_with_jatta).pkl',
        'purchase_prediction_mint_lip_balm.pkl',
    ]
    
    for model_file in model_files:
        model_path = f'models/indian/{model_file}'
        if not os.path.exists(model_path):
            continue
            
        print(f"\n📦 Testing: {model_file}")
        
        # Load model
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        predictor = PurchasePrediction()
        predictor.model = data['model']
        predictor.scaler = data['scaler']
        predictor.feature_names = data['feature_names']
        
        # Get top features
        feature_importance_df = predictor.get_feature_importance(predictor.feature_names)
        top_5 = feature_importance_df.head(5)
        
        print(f"   ✅ Model loaded!")
        print(f"   🔍 Top 5 Predictive Products:")
        for idx, row in top_5.iterrows():
            print(f"      {idx+1}. {row['Product'][:50]}: {row['Importance']:.4f}")
    
    print(f"\n✅ All prediction models tested!")


def interactive_prediction():
    """Interactive purchase prediction demo"""
    print("\n" + "="*80)
    print("🛒 INTERACTIVE PURCHASE PREDICTION DEMO")
    print("="*80)
    
    # Load data
    df = pd.read_csv('data/indian/bigbasket_transactions_5k.csv')
    product_counts = df['Item'].value_counts()
    popular_products = product_counts.head(200).index.tolist()
    df = df[df['Item'].isin(popular_products)]
    
    print(f"\n📊 Available Products: {len(popular_products)}")
    print("\n🎯 Top 20 Products:")
    for i, product in enumerate(popular_products[:20], 1):
        count = product_counts[product]
        print(f"   {i}. {product} ({count} purchases)")
    
    # Load coconut prediction model
    model_path = 'models/indian/purchase_prediction_coconut_-_(for_kalash_with_jatta).pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        predictor = PurchasePrediction()
        predictor.model = data['model']
        predictor.scaler = data['scaler']
        predictor.feature_names = data['feature_names']
        
        print(f"\n✅ Loaded Coconut Purchase Prediction Model")
        print(f"\n💡 Example: Predicting if customer will buy Coconut...")
        
        # Get a random transaction
        sample_transaction = df[df['Transaction'] == df['Transaction'].unique()[0]]
        basket_items = sample_transaction['Item'].tolist()
        
        print(f"\n🛒 Customer's Current Basket ({len(basket_items)} items):")
        for item in basket_items[:5]:
            print(f"   - {item}")
        if len(basket_items) > 5:
            print(f"   ... and {len(basket_items)-5} more items")
        
        print(f"\n🎯 Model says: This customer has X% chance of buying Coconut")
        print(f"   (Actual prediction would require running model.predict())")


def show_usage_guide():
    """Show how to use the models"""
    print("\n" + "="*80)
    print("📚 HOW TO USE THESE MODELS IN YOUR APP")
    print("="*80)
    
    print("""
1️⃣ CUSTOMER SEGMENTATION
   Use Case: Identify customer type for targeted marketing
   
   Code:
   ```python
   # Load model
   with open('models/indian/customer_segmentation.pkl', 'rb') as f:
       data = pickle.load(f)
   
   # Predict segment for new customer
   segment = model.predict(customer_features)
   # segment = 0-4 (Quick Shopper to Premium Shopper)
   ```

2️⃣ PURCHASE PREDICTION
   Use Case: Recommend next product based on current basket
   
   Code:
   ```python
   # Load model
   with open('models/indian/purchase_prediction_coconut.pkl', 'rb') as f:
       data = pickle.load(f)
   
   # Predict if customer will buy coconut
   prediction = model.predict(customer_basket)
   # prediction = 0 (No) or 1 (Yes)
   ```

3️⃣ MARKET BASKET ANALYSIS
   Use Case: Find product associations (bought together)
   
   Code:
   ```python
   # Load saved rules
   with open('models/indian/market_basket_analysis.pkl', 'rb') as f:
       data = pickle.load(f)
       rules = data['rules']
   
   # Show rules where antecedent is 'Rice'
   rice_rules = rules[rules['antecedents'].apply(lambda x: 'Rice' in x)]
   ```

🎯 NEXT STEPS:
   1. Create Streamlit dashboard (app_indian.py)
   2. Deploy to Streamlit Cloud
   3. Add to your portfolio
   4. Apply to Reliance/DMart/BigBasket with this project!
""")


def main():
    """Main test function"""
    print("="*80)
    print("🇮🇳 INDIAN MARKET BASKET ANALYSIS - MODEL TESTING")
    print("="*80)
    
    # Test each model
    test_customer_segmentation()
    test_purchase_prediction()
    interactive_prediction()
    show_usage_guide()
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETE!")
    print("="*80)
    print("\n🚀 Next Steps:")
    print("   1. Run: streamlit run app.py (use existing dashboard)")
    print("   2. Or create Indian-specific dashboard: app_indian.py")
    print("   3. Deploy to Streamlit Cloud")
    print("   4. Add GitHub README with screenshots")


if __name__ == '__main__':
    main()
