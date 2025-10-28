"""
Convert Instacart Market Basket Dataset to Transaction Format
The Instacart dataset is a relational database with foreign keys - we need to join them!

Dataset Structure:
- orders.csv: 3.4M orders (order_id, user_id, eval_set, order_number, ...)
- order_products__prior.csv: 32M rows (order_id, product_id, add_to_cart_order, reordered)
- order_products__train.csv: 1.4M rows (order_id, product_id, add_to_cart_order, reordered)
- products.csv: 49K products (product_id, product_name, aisle_id, department_id)
- aisles.csv: 134 aisles (aisle_id, aisle)
- departments.csv: 21 departments (department_id, department)
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_instacart_data():
    """
    Load all Instacart CSV files
    """
    print("="*80)
    print("🛒 Loading Instacart Market Basket Dataset")
    print("="*80)
    
    base_path = Path('data/instacart-data')
    
    # Load all files
    print("\n📥 Loading files...")
    
    print("   Loading products.csv...")
    products = pd.read_csv(base_path / 'products.csv')
    print(f"   ✅ {len(products):,} products")
    
    print("   Loading aisles.csv...")
    aisles = pd.read_csv(base_path / 'aisles.csv')
    print(f"   ✅ {len(aisles):,} aisles")
    
    print("   Loading departments.csv...")
    departments = pd.read_csv(base_path / 'departments.csv')
    print(f"   ✅ {len(departments):,} departments")
    
    print("   Loading orders.csv (this may take a moment)...")
    orders = pd.read_csv(base_path / 'orders.csv')
    print(f"   ✅ {len(orders):,} orders")
    
    print("   Loading order_products__prior.csv (32M rows - be patient!)...")
    order_products_prior = pd.read_csv(base_path / 'order_products__prior.csv')
    print(f"   ✅ {len(order_products_prior):,} order-product records (prior)")
    
    print("   Loading order_products__train.csv...")
    order_products_train = pd.read_csv(base_path / 'order_products__train.csv')
    print(f"   ✅ {len(order_products_train):,} order-product records (train)")
    
    return {
        'products': products,
        'aisles': aisles,
        'departments': departments,
        'orders': orders,
        'order_products_prior': order_products_prior,
        'order_products_train': order_products_train
    }


def convert_to_transaction_format(data, sample_size=50000, use_prior=True):
    """
    Convert Instacart relational data to Transaction, Item format
    
    Args:
        data: Dictionary of DataFrames
        sample_size: Number of orders to sample (None for all)
        use_prior: Use prior dataset (32M rows) or train (1.4M rows)
    """
    print("\n" + "="*80)
    print("🔄 Converting to Transaction Format")
    print("="*80)
    
    # Choose dataset
    if use_prior:
        order_products = data['order_products_prior']
        dataset_name = 'prior (32M rows)'
    else:
        order_products = data['order_products_train']
        dataset_name = 'train (1.4M rows)'
    
    print(f"\n📊 Using {dataset_name}")
    
    # Join products to get product names
    print("\n🔗 Step 1: Joining order_products with products...")
    order_items = order_products.merge(
        data['products'][['product_id', 'product_name']], 
        on='product_id', 
        how='left'
    )
    print(f"   ✅ {len(order_items):,} rows after join")
    
    # Sample orders if needed
    if sample_size:
        print(f"\n🎯 Step 2: Sampling {sample_size:,} random orders...")
        unique_orders = order_items['order_id'].unique()
        sampled_orders = np.random.choice(unique_orders, min(sample_size, len(unique_orders)), replace=False)
        order_items = order_items[order_items['order_id'].isin(sampled_orders)]
        print(f"   ✅ {len(order_items):,} rows after sampling")
    
    # Create Transaction, Item format
    print("\n📝 Step 3: Creating Transaction, Item format...")
    transactions = order_items[['order_id', 'product_name']].copy()
    transactions.columns = ['Transaction', 'Item']
    
    # Remove any null items
    transactions = transactions[transactions['Item'].notna()]
    
    print(f"\n✅ Conversion Complete!")
    print(f"   📊 Total Transactions: {transactions['Transaction'].nunique():,}")
    print(f"   📦 Total Items (rows): {len(transactions):,}")
    print(f"   🛍️ Unique Products: {transactions['Item'].nunique():,}")
    print(f"   📈 Avg Items/Transaction: {len(transactions)/transactions['Transaction'].nunique():.2f}")
    
    return transactions


def save_and_show_stats(transactions, output_file='data/instacart_transactions.csv'):
    """
    Save transactions and show statistics
    """
    print("\n" + "="*80)
    print("💾 Saving and Analyzing Data")
    print("="*80)
    
    # Save
    transactions.to_csv(output_file, index=False)
    print(f"\n✅ Saved to: {output_file}")
    
    # Show stats
    print("\n📊 Dataset Statistics:")
    print(f"   Total Records: {len(transactions):,}")
    print(f"   Unique Transactions: {transactions['Transaction'].nunique():,}")
    print(f"   Unique Products: {transactions['Item'].nunique():,}")
    
    # Basket size distribution
    basket_sizes = transactions.groupby('Transaction').size()
    print(f"\n🛒 Basket Size Distribution:")
    print(f"   Min: {basket_sizes.min()}")
    print(f"   Max: {basket_sizes.max()}")
    print(f"   Mean: {basket_sizes.mean():.2f}")
    print(f"   Median: {basket_sizes.median():.0f}")
    
    # Top products
    print(f"\n🏆 Top 10 Products:")
    top_products = transactions['Item'].value_counts().head(10)
    for i, (product, count) in enumerate(top_products.items(), 1):
        print(f"   {i}. {product}: {count:,} times")
    
    # Sample transactions
    print(f"\n📋 Sample Transactions:")
    sample_trans = transactions.groupby('Transaction').head(5).groupby('Transaction')['Item'].apply(list).head(3)
    for trans_id, items in sample_trans.items():
        print(f"   Transaction {trans_id}: {items}")


def main():
    """
    Main conversion function
    """
    print("\n" + "="*80)
    print("🛒 INSTACART TO TRANSACTION CONVERTER")
    print("="*80)
    print("\nThis script converts Instacart's relational database format")
    print("into a simple Transaction, Item format for Market Basket Analysis")
    
    # Load data
    data = load_instacart_data()
    
    # Convert with sampling
    print("\n" + "="*80)
    print("⚙️ Conversion Options:")
    print("="*80)
    print("\n1. Sample 50,000 transactions (recommended for testing)")
    print("2. Sample 100,000 transactions (good balance)")
    print("3. Use ALL 3.4M transactions (requires ~8GB RAM!)")
    
    # For now, let's do 50K sample
    transactions = convert_to_transaction_format(
        data, 
        sample_size=50000,  # Sample 50K orders
        use_prior=True      # Use prior dataset (larger)
    )
    
    # Save and analyze
    save_and_show_stats(transactions, 'data/instacart_transactions_50k.csv')
    
    print("\n" + "="*80)
    print("✅ SUCCESS! Real transaction data ready for training!")
    print("="*80)
    print("\n🎯 Next Steps:")
    print("   1. Check the CSV: data/instacart_transactions_50k.csv")
    print("   2. Upload to Streamlit app at: http://localhost:8502")
    print("   3. Or train models: python train_models_indian.py")
    print("   4. Compare results with synthetic BigBasket data!")


if __name__ == '__main__':
    main()
