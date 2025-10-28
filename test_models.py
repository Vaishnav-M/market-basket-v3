"""
Test trained ML models
Demonstrates how to load and use the trained models for predictions
"""

import pandas as pd
import sys
from pathlib import Path
import os

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from ml_models import CustomerSegmentation, PurchasePrediction


def test_customer_segmentation():
    """
    Test customer segmentation model
    """
    print("\n" + "="*80)
    print("👥 TESTING CUSTOMER SEGMENTATION MODEL")
    print("="*80)
    
    model_path = 'models/customer_segmentation.pkl'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("   Run 'python train_models.py' first!")
        return
    
    # Load model
    print(f"📦 Loading model from {model_path}...")
    segmenter = CustomerSegmentation()
    segmenter.load_model(model_path)
    
    # Test with sample customers
    print("\n🧪 Testing with sample customers:")
    
    test_customers = [
        {'num_items': 10, 'unique_categories': 8, 'transaction_value': 10},  # Premium Shopper
        {'num_items': 15, 'unique_categories': 3, 'transaction_value': 15},  # Bulk Buyer
        {'num_items': 3, 'unique_categories': 3, 'transaction_value': 3},    # Quick Shopper
        {'num_items': 5, 'unique_categories': 5, 'transaction_value': 5},    # Variety Seeker
    ]
    
    for idx, customer in enumerate(test_customers, 1):
        # Prepare features
        features = [[customer['num_items'], customer['unique_categories'], customer['transaction_value']]]
        
        # Predict segment
        segment = segmenter.predict_segment(features)[0]
        
        print(f"\n   Customer {idx}:")
        print(f"      Items: {customer['num_items']}, Variety: {customer['unique_categories']}, Value: ${customer['transaction_value']}")
        print(f"      → Predicted Segment: {segment}")
    
    print("\n✅ Customer Segmentation test complete!")


def test_purchase_prediction(target_product='Toothpaste'):
    """
    Test purchase prediction model
    
    Args:
        target_product: Product to predict
    """
    print("\n" + "="*80)
    print(f"🎯 TESTING PURCHASE PREDICTION MODEL")
    print(f"   Target Product: {target_product}")
    print("="*80)
    
    model_name = f"purchase_prediction_{target_product.lower().replace(' ', '_')}.pkl"
    model_path = f'models/{model_name}'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("   Run 'python train_models.py' first!")
        return
    
    # Load model
    print(f"📦 Loading model from {model_path}...")
    predictor = PurchasePrediction()
    predictor.load_model(model_path)
    
    # Load data to get product names
    df = pd.read_csv('data/sample_transactions.csv')
    all_products = sorted(df['Item'].unique())
    other_products = [p for p in all_products if p != target_product]
    
    # Test scenarios
    print("\n🧪 Testing purchase scenarios:")
    
    test_scenarios = [
        {
            'name': 'Scenario 1: Grooming products in basket',
            'products': ['Toothbrush', 'Mouthwash']
        },
        {
            'name': 'Scenario 2: Dairy products in basket',
            'products': ['Milk', 'Cheese']
        },
        {
            'name': 'Scenario 3: Empty basket',
            'products': []
        },
        {
            'name': 'Scenario 4: Random products',
            'products': ['Beer', 'Chips']
        }
    ]
    
    for scenario in test_scenarios:
        # Create basket
        basket = {product: 1 if product in scenario['products'] else 0 
                 for product in other_products}
        
        # Predict
        try:
            prediction, probability = predictor.predict_next_purchase(basket)
            
            print(f"\n   {scenario['name']}")
            print(f"      Basket: {scenario['products'] if scenario['products'] else 'Empty'}")
            print(f"      Prediction: {'Will buy' if prediction == 1 else 'Will NOT buy'} {target_product}")
            print(f"      Confidence: {probability:.1%}")
            
            if prediction == 1:
                print(f"      💡 Recommendation: Suggest '{target_product}' to customer!")
        except Exception as e:
            print(f"\n   {scenario['name']}")
            print(f"      ❌ Error: {str(e)}")
    
    print("\n✅ Purchase Prediction test complete!")


def test_all_models():
    """
    Test all trained models
    """
    print("\n" + "="*80)
    print("🚀 TESTING ALL TRAINED MODELS")
    print("="*80)
    
    # Test customer segmentation
    test_customer_segmentation()
    
    # Test purchase prediction for multiple products
    target_products = ['Toothpaste', 'Milk', 'Eggs']
    
    for product in target_products:
        test_purchase_prediction(product)
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETE!")
    print("="*80)
    print("\n🎯 Models are ready for deployment in your Streamlit app!")
    print("   Run 'streamlit run app.py' to use them interactively")
    print("\n")


def interactive_prediction():
    """
    Interactive testing - user can input their own baskets
    """
    print("\n" + "="*80)
    print("🛒 INTERACTIVE PURCHASE PREDICTION")
    print("="*80)
    
    # Load data
    df = pd.read_csv('data/sample_transactions.csv')
    all_products = sorted(df['Item'].unique())
    
    print("\n📦 Available Products:")
    for idx, product in enumerate(all_products, 1):
        print(f"   {idx}. {product}")
    
    # Get target product
    print("\n🎯 Which product do you want to predict?")
    target_idx = int(input("   Enter product number: ")) - 1
    target_product = all_products[target_idx]
    
    # Load model
    model_name = f"purchase_prediction_{target_product.lower().replace(' ', '_')}.pkl"
    model_path = f'models/{model_name}'
    
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found for '{target_product}'")
        print("   Run 'python train_models.py' first!")
        return
    
    predictor = PurchasePrediction()
    predictor.load_model(model_path)
    
    # Get basket items
    print(f"\n🛒 What's in the customer's basket?")
    print("   Enter product numbers separated by commas (e.g., 1,3,5)")
    print("   Or press Enter for empty basket")
    
    basket_input = input("   Basket: ").strip()
    
    if basket_input:
        basket_indices = [int(i.strip()) - 1 for i in basket_input.split(',')]
        basket_products = [all_products[i] for i in basket_indices if i < len(all_products)]
    else:
        basket_products = []
    
    # Create basket vector
    other_products = [p for p in all_products if p != target_product]
    basket = {product: 1 if product in basket_products else 0 for product in other_products}
    
    # Predict
    prediction, probability = predictor.predict_next_purchase(basket)
    
    print("\n" + "="*80)
    print("📊 PREDICTION RESULT")
    print("="*80)
    print(f"\n   Target Product: {target_product}")
    print(f"   Basket: {basket_products if basket_products else 'Empty'}")
    print(f"\n   Prediction: {'✅ Will buy' if prediction == 1 else '❌ Will NOT buy'} {target_product}")
    print(f"   Confidence: {probability:.1%}")
    
    if prediction == 1:
        print(f"\n   💡 Recommendation: Show '{target_product}' to this customer!")
    
    print("\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_prediction()
    else:
        test_all_models()
        
        print("\n💡 Tip: Run 'python test_models.py --interactive' for hands-on testing!")
