"""
Process BigBasket Indian grocery dataset for market basket analysis
Converts product catalog into transaction format
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def process_bigbasket_catalog(input_file='data/indian/BigBasket.csv'):
    """
    Process BigBasket product catalog and generate realistic transactions
    """
    print("="*80)
    print("🇮🇳 Processing BigBasket Indian Grocery Dataset")
    print("="*80)
    
    # Load BigBasket product catalog
    print(f"\n📥 Loading dataset: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"✅ Loaded {len(df):,} products")
    print(f"\nColumns: {list(df.columns)}")
    
    # Show product categories
    if 'category' in df.columns:
        print(f"\n📊 Product Categories:")
        print(df['category'].value_counts().head(10))
    
    if 'sub_category' in df.columns:
        print(f"\n📊 Sub-Categories:")
        print(df['sub_category'].value_counts().head(10))
    
    # Show sample products
    print(f"\n🛒 Sample Products:")
    if 'product' in df.columns:
        print(df['product'].head(20))
    
    return df


def generate_transactions_from_catalog(catalog_df, num_transactions=5000):
    """
    Generate realistic Indian shopping transactions from product catalog
    """
    print("\n" + "="*80)
    print(f"🛒 Generating {num_transactions:,} Indian Shopping Transactions")
    print("="*80)
    
    transactions = []
    transaction_id = 1
    
    # Get product names
    product_col = 'product' if 'product' in catalog_df.columns else catalog_df.columns[0]
    category_col = 'category' if 'category' in catalog_df.columns else None
    
    products = catalog_df[product_col].dropna().unique()
    
    # Indian shopping patterns
    basket_sizes = {
        'quick_shop': {'size': (3, 8), 'weight': 0.3},
        'weekly': {'size': (15, 30), 'weight': 0.35},
        'monthly': {'size': (35, 60), 'weight': 0.25},
        'festival': {'size': (50, 100), 'weight': 0.1}
    }
    
    for _ in range(num_transactions):
        # Choose shopping pattern
        pattern = random.choices(
            list(basket_sizes.keys()),
            weights=[p['weight'] for p in basket_sizes.values()]
        )[0]
        
        basket_size = random.randint(*basket_sizes[pattern]['size'])
        
        # Select random products
        basket_products = np.random.choice(products, min(basket_size, len(products)), replace=False)
        
        # Create transaction records
        for product in basket_products:
            transactions.append({
                'Transaction': transaction_id,
                'Item': product,
                'ShoppingPattern': pattern
            })
        
        transaction_id += 1
        
        if transaction_id % 1000 == 0:
            print(f"   Generated {transaction_id-1:,} transactions...")
    
    # Create DataFrame
    trans_df = pd.DataFrame(transactions)
    
    print(f"\n✅ Generated {num_transactions:,} transactions")
    print(f"   Total Records: {len(trans_df):,}")
    print(f"   Unique Products: {trans_df['Item'].nunique():,}")
    
    # Save
    output_file = 'data/indian/bigbasket_transactions_5k.csv'
    trans_df.to_csv(output_file, index=False)
    print(f"   Saved to: {output_file}")
    
    return trans_df


def main():
    """Main processing function"""
    
    import os
    
    # Check if BigBasket data exists
    input_files = [
        'data/indian/BigBasket.csv',
        'data/indian/BigBasket Products.csv',
        'data/indian/bigbasket.csv',
        'data/indian/bigbasket_products.csv'
    ]
    
    found = False
    for input_file in input_files:
        if os.path.exists(input_file):
            print(f"✅ Found dataset: {input_file}")
            
            # Process catalog
            catalog_df = process_bigbasket_catalog(input_file)
            
            # Generate transactions
            generate_transactions_from_catalog(catalog_df, num_transactions=5000)
            
            # Also generate larger dataset
            generate_transactions_from_catalog(catalog_df, num_transactions=20000)
            
            found = True
            break
    
    if not found:
        print("\n" + "="*80)
        print("❌ BigBasket dataset not found!")
        print("="*80)
        print("\n📥 Please download the BigBasket dataset first:")
        print("\n1. Go to: https://www.kaggle.com/datasets/surajjha101/bigbasket-entire-product-list-28k-datapoints")
        print("2. Click 'Download' (requires free Kaggle account)")
        print("3. Extract ZIP and copy 'BigBasket.csv' to:")
        print(f"   {os.path.abspath('data/indian/')}")
        print("\n4. Then run this script again:")
        print("   python process_bigbasket_data.py")
    else:
        print("\n" + "="*80)
        print("✅ SUCCESS! Indian transaction data ready for training!")
        print("="*80)
        print("\n🎯 Next Steps:")
        print("   1. Run: python train_models_indian.py")
        print("   2. Or: streamlit run app_indian.py")


if __name__ == '__main__':
    main()
