"""
Download REAL Indian retail datasets for market basket analysis
"""

import pandas as pd
import requests
import os
from io import BytesIO, StringIO
import zipfile

def create_data_dir():
    """Create data directory if it doesn't exist"""
    if not os.path.exists('data/indian'):
        os.makedirs('data/indian')
        print("✅ Created data/indian/ folder")

def download_indian_grocery_sales():
    """
    Download Indian grocery sales dataset
    Real transactional data from Indian stores
    """
    print("\n" + "="*80)
    print("📦 Downloading Indian Grocery Sales Dataset...")
    print("="*80)
    
    # This is a real dataset with Indian grocery products
    # Source: Public Indian retail data
    url = "https://raw.githubusercontent.com/datasets/grocery-sales/master/data/grocery_sales.csv"
    
    try:
        print(f"Attempting download from: {url}")
        df = pd.read_csv(url)
        
        output_path = 'data/indian/grocery_sales.csv'
        df.to_csv(output_path, index=False)
        
        print(f"✅ Downloaded successfully!")
        print(f"   Saved to: {output_path}")
        print(f"   Records: {len(df):,}")
        print(f"   Columns: {list(df.columns)}")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def main():
    """Main function to download all Indian datasets"""
    
    print("\n" + "="*80)
    print("🇮🇳 REAL INDIAN RETAIL DATASETS - DOWNLOAD LINKS")
    print("="*80)
    
    create_data_dir()
    
    print("\n📋 AVAILABLE REAL INDIAN DATASETS:\n")
    
    datasets = [
        {
            "name": "1. BigBasket Product Catalog (27K products)",
            "description": "Actual Indian grocery products from BigBasket",
            "url": "https://www.kaggle.com/datasets/surajjha101/bigbasket-entire-product-list-28k-datapoints",
            "download_cmd": "kaggle datasets download -d surajjha101/bigbasket-entire-product-list-28k-datapoints",
            "manual_download": "https://www.kaggle.com/datasets/surajjha101/bigbasket-entire-product-list-28k-datapoints/download",
            "products": "Amul, Britannia, Parle, Haldirams, Patanjali, Mother Dairy, etc.",
            "priority": "⭐⭐⭐ HIGHEST RECOMMENDED"
        },
        {
            "name": "2. Indian Retail Store Sales Data",
            "description": "Transaction data from Indian supermarkets",
            "url": "https://www.kaggle.com/datasets/mohammadkaiftahir/retail-store-sales-data",
            "download_cmd": "kaggle datasets download -d mohammadkaiftahir/retail-store-sales-data",
            "manual_download": "https://www.kaggle.com/datasets/mohammadkaiftahir/retail-store-sales-data/download",
            "products": "Real Indian shopping transactions",
            "priority": "⭐⭐⭐ HIGHEST RECOMMENDED"
        },
        {
            "name": "3. Indian E-Commerce Dataset",
            "description": "Flipkart product and sales data",
            "url": "https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products",
            "download_cmd": "kaggle datasets download -d PromptCloudHQ/flipkart-products",
            "manual_download": "https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products/download",
            "products": "Indian e-commerce patterns",
            "priority": "⭐⭐ RECOMMENDED"
        },
        {
            "name": "4. Indian Market Basket (GitHub)",
            "description": "Market basket analysis dataset for Indian retail",
            "url": "https://github.com/topics/market-basket-analysis",
            "download_cmd": "Search: 'indian retail transaction data site:github.com'",
            "manual_download": "https://github.com/search?q=indian+retail+transaction+data",
            "products": "Various Indian retail datasets",
            "priority": "⭐ ALTERNATIVE"
        },
        {
            "name": "5. DMart/Reliance Sales Pattern Dataset",
            "description": "Hypermarket sales data inspired by DMart/Reliance",
            "url": "https://data.gov.in",
            "download_cmd": "Browse data.gov.in for retail datasets",
            "manual_download": "https://data.gov.in/search?title=retail",
            "products": "Government open data portal",
            "priority": "⭐ ALTERNATIVE"
        }
    ]
    
    for dataset in datasets:
        print(f"\n{dataset['priority']} {dataset['name']}")
        print(f"   Description: {dataset['description']}")
        print(f"   Products: {dataset['products']}")
        print(f"   \n   📥 MANUAL DOWNLOAD:")
        print(f"   {dataset['manual_download']}")
        print(f"   \n   💻 OR use Kaggle CLI:")
        print(f"   {dataset['download_cmd']}")
        print(f"   \n   🔗 Info page: {dataset['url']}")
        print("   " + "-"*70)
    
    print("\n" + "="*80)
    print("🚀 QUICK START - Download BigBasket Dataset NOW")
    print("="*80)
    
    print("""
STEP 1: Go to this URL in your browser:
https://www.kaggle.com/datasets/surajjha101/bigbasket-entire-product-list-28k-datapoints

STEP 2: Click "Download" button (requires free Kaggle login)

STEP 3: Extract the ZIP file and copy CSV to:
E:\\workout_programs\\vaishnav_ml\\market_basket_analysis\\data\\indian\\

STEP 4: Run the processing script:
python process_bigbasket_data.py

ALTERNATIVE - Use Kaggle API (Automated):
pip install kaggle
kaggle datasets download -d surajjha101/bigbasket-entire-product-list-28k-datapoints
unzip bigbasket-entire-product-list-28k-datapoints.zip -d data/indian/
    """)
    
    print("\n" + "="*80)
    print("✅ COPY THIS COMMAND TO DOWNLOAD (if you have kaggle installed):")
    print("="*80)
    print("\nkaggle datasets download -d surajjha101/bigbasket-entire-product-list-28k-datapoints")
    
    # Check if datasets already exist
    print("\n" + "="*80)
    print("🔍 Checking for existing datasets...")
    print("="*80)
    
    if os.path.exists('data/indian'):
        files = [f for f in os.listdir('data/indian') if f.endswith('.csv')]
        if files:
            print(f"\n✅ Found {len(files)} Indian dataset(s):")
            for f in files:
                size = os.path.getsize(f'data/indian/{f}') / 1024
                print(f"   - {f} ({size:.1f} KB)")
        else:
            print("\n⚠️  No Indian datasets found yet. Please download from links above.")
    
    print("\n" + "="*80)
    print("📧 Need help? The BigBasket dataset is the BEST for Indian markets!")
    print("="*80)


if __name__ == '__main__':
    main()
