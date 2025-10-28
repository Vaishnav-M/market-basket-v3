"""
Download and prepare real-world datasets for Market Basket Analysis
"""

import pandas as pd
import numpy as np
import os
import requests
from io import BytesIO


def download_online_retail():
    """
    Download UCI Online Retail Dataset (540K transactions)
    Best quality, ready-to-use dataset for learning
    """
    print("\n" + "="*80)
    print("📦 DOWNLOADING ONLINE RETAIL DATASET (UCI Repository)")
    print("="*80)
    
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
    
    print(f"\n📥 Downloading from: {url}")
    print("   This may take a few minutes (45 MB file)...")
    
    try:
        # Download
        response = requests.get(url)
        df = pd.read_excel(BytesIO(response.content))
        
        print(f"✅ Downloaded {len(df):,} records")
        
        # Show sample
        print(f"\n📊 Sample Data:")
        print(df.head())
        
        # Clean data
        print(f"\n🧹 Cleaning data...")
        original_size = len(df)
        
        # Remove nulls
        df = df.dropna(subset=['CustomerID', 'Description'])
        print(f"   Removed {original_size - len(df):,} rows with missing values")
        
        # Remove returns (negative quantities)
        df = df[df['Quantity'] > 0]
        df = df[df['UnitPrice'] > 0]
        
        # Remove non-product items
        df = df[~df['Description'].str.contains('POSTAGE|BANK CHARGES|DISCOUNT', case=False, na=False)]
        
        print(f"✅ Cleaned dataset: {len(df):,} records")
        
        # Transform to Market Basket format
        print(f"\n🔄 Transforming to transaction format...")
        
        df = df.rename(columns={
            'InvoiceNo': 'Transaction',
            'Description': 'Item'
        })
        
        # Select columns
        df = df[['Transaction', 'Item', 'Quantity', 'UnitPrice', 'CustomerID', 'InvoiceDate', 'Country']]
        
        # Save
        os.makedirs('data', exist_ok=True)
        output_path = 'data/online_retail.csv'
        df.to_csv(output_path, index=False)
        
        print(f"✅ Saved to: {output_path}")
        
        # Statistics
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total Transactions: {df['Transaction'].nunique():,}")
        print(f"   Unique Products: {df['Item'].nunique():,}")
        print(f"   Unique Customers: {df['CustomerID'].nunique():,}")
        print(f"   Total Items Sold: {df['Quantity'].sum():,.0f}")
        print(f"   Date Range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
        print(f"   Countries: {df['Country'].nunique()}")
        
        # Top products
        print(f"\n🏆 Top 10 Products:")
        top_products = df.groupby('Item')['Quantity'].sum().sort_values(ascending=False).head(10)
        for idx, (product, qty) in enumerate(top_products.items(), 1):
            print(f"   {idx}. {product}: {qty:,.0f} sold")
        
        return df
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {str(e)}")
        print("   Possible solutions:")
        print("   1. Check your internet connection")
        print("   2. Try again later (server might be down)")
        print("   3. Download manually from: https://archive.ics.uci.edu/ml/datasets/online+retail")
        return None


def download_groceries_dataset():
    """
    Download Groceries Dataset from Kaggle (requires Kaggle API)
    """
    print("\n" + "="*80)
    print("📦 DOWNLOADING GROCERIES DATASET (Kaggle)")
    print("="*80)
    
    print("\n⚠️  This requires Kaggle API credentials")
    print("   Setup instructions:")
    print("   1. Go to https://www.kaggle.com/account")
    print("   2. Click 'Create New API Token'")
    print("   3. Save kaggle.json to ~/.kaggle/ (Linux/Mac) or C:\\Users\\<User>\\.kaggle\\ (Windows)")
    print("   4. Install: pip install kaggle")
    
    try:
        import kaggle
        
        print("\n📥 Downloading from Kaggle...")
        
        # Download
        kaggle.api.dataset_download_files(
            'heeraldedhia/groceries-dataset',
            path='data/groceries',
            unzip=True
        )
        
        # Load
        df = pd.read_csv('data/groceries/groceries - groceries.csv')
        
        print(f"✅ Downloaded {len(df):,} records")
        
        # Transform to standard format
        df = df.rename(columns={
            'Member_number': 'Transaction',
            'itemDescription': 'Item'
        })
        
        # Save
        df.to_csv('data/groceries_clean.csv', index=False)
        
        print(f"✅ Saved to: data/groceries_clean.csv")
        print(f"   Transactions: {df['Transaction'].nunique():,}")
        print(f"   Unique Products: {df['Item'].nunique():,}")
        
        return df
        
    except ImportError:
        print("\n❌ Kaggle API not installed")
        print("   Install with: pip install kaggle")
        return None
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return None


def create_sample_from_large_dataset(input_path, output_path, n_transactions=1000):
    """
    Create a smaller sample from large dataset for testing
    
    Args:
        input_path: Path to large CSV file
        output_path: Path to save sample
        n_transactions: Number of transactions to sample
    """
    print(f"\n📊 Creating sample from {input_path}...")
    
    df = pd.read_csv(input_path)
    
    # Sample random transactions
    transactions = df['Transaction'].unique()
    sampled = np.random.choice(transactions, min(n_transactions, len(transactions)), replace=False)
    
    sample_df = df[df['Transaction'].isin(sampled)]
    
    sample_df.to_csv(output_path, index=False)
    
    print(f"✅ Saved sample to {output_path}")
    print(f"   Transactions: {sample_df['Transaction'].nunique():,}")
    print(f"   Products: {sample_df['Item'].nunique():,}")
    
    return sample_df


def compare_datasets():
    """
    Compare available datasets
    """
    print("\n" + "="*80)
    print("📊 DATASET COMPARISON")
    print("="*80)
    
    datasets = []
    
    # Check sample data
    if os.path.exists('data/sample_transactions.csv'):
        df = pd.read_csv('data/sample_transactions.csv')
        datasets.append({
            'Name': 'Sample Data (Built-in)',
            'Transactions': df['Transaction'].nunique(),
            'Products': df['Item'].nunique(),
            'Size': 'Small',
            'Quality': '⭐⭐⭐',
            'Use Case': 'Learning/Testing'
        })
    
    # Check online retail
    if os.path.exists('data/online_retail.csv'):
        df = pd.read_csv('data/online_retail.csv')
        datasets.append({
            'Name': 'Online Retail (UCI)',
            'Transactions': df['Transaction'].nunique(),
            'Products': df['Item'].nunique(),
            'Size': 'Large',
            'Quality': '⭐⭐⭐⭐⭐',
            'Use Case': 'Production'
        })
    
    # Check groceries
    if os.path.exists('data/groceries_clean.csv'):
        df = pd.read_csv('data/groceries_clean.csv')
        datasets.append({
            'Name': 'Groceries (Kaggle)',
            'Transactions': df['Transaction'].nunique(),
            'Products': df['Item'].nunique(),
            'Size': 'Medium',
            'Quality': '⭐⭐⭐⭐',
            'Use Case': 'Production'
        })
    
    if datasets:
        df_comparison = pd.DataFrame(datasets)
        print("\n" + df_comparison.to_string(index=False))
    else:
        print("\n⚠️  No datasets found. Run download functions first!")
    
    print("\n")


def main():
    """
    Main function to download all datasets
    """
    print("\n" + "="*80)
    print("🚀 DATASET DOWNLOADER FOR MARKET BASKET ANALYSIS")
    print("="*80)
    
    print("\n📋 Available datasets:")
    print("   1. Online Retail (540K transactions) - Recommended ⭐")
    print("   2. Groceries (9K transactions)")
    print("   3. Both")
    
    choice = input("\n👉 Choose dataset to download (1/2/3): ").strip()
    
    if choice == '1' or choice == '3':
        df = download_online_retail()
        
        if df is not None:
            # Create smaller samples
            print("\n🔪 Creating smaller samples for testing...")
            create_sample_from_large_dataset(
                'data/online_retail.csv',
                'data/online_retail_1k.csv',
                n_transactions=1000
            )
            create_sample_from_large_dataset(
                'data/online_retail.csv',
                'data/online_retail_5k.csv',
                n_transactions=5000
            )
    
    if choice == '2' or choice == '3':
        download_groceries_dataset()
    
    # Show comparison
    compare_datasets()
    
    print("\n✅ DOWNLOAD COMPLETE!")
    print("\n🎯 Next Steps:")
    print("   1. Run: python train_models.py --data data/online_retail_1k.csv")
    print("   2. Compare results with sample data")
    print("   3. Scale up to larger datasets")
    print("\n")


if __name__ == "__main__":
    main()
