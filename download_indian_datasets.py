"""
Download real Indian retail and e-commerce datasets for market basket analysis
"""

import pandas as pd
import requests
import os
from io import StringIO

def download_file(url, output_path):
    """Download file from URL"""
    print(f"Downloading from: {url}")
    response = requests.get(url)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        f.write(response.content)
    print(f"✅ Saved to: {output_path}")


def download_instacart_india():
    """
    Instacart has Indian grocery products in their dataset
    Similar shopping patterns to Indian supermarkets
    """
    print("\n" + "="*80)
    print("📦 OPTION 1: Instacart Grocery Dataset (includes Indian products)")
    print("="*80)
    print("\nNote: This is a large dataset (3.4M orders, 200K+ products)")
    print("Contains grocery shopping data similar to Indian supermarkets")
    print("\nDownload manually from:")
    print("https://www.kaggle.com/c/instacart-market-basket-analysis/data")
    print("\nRequires Kaggle account (free)")


def download_dunnhumby():
    """
    Dunnhumby - Complete Journey dataset
    Grocery shopping transactions
    """
    print("\n" + "="*80)
    print("📦 OPTION 2: Dunnhumby Complete Journey (Grocery transactions)")
    print("="*80)
    print("\nThis dataset contains 2 years of grocery shopping data")
    print("2,500 households, similar to Indian family shopping patterns")
    print("\nDownload manually from:")
    print("https://www.dunnhumby.com/source-files/")
    print("\nRequires free registration")


def search_indian_datasets():
    """
    Search and list available Indian retail datasets
    """
    print("\n" + "="*80)
    print("🇮🇳 REAL INDIAN RETAIL DATASETS - AVAILABLE SOURCES")
    print("="*80)
    
    datasets = {
        "1. BigBasket Products Dataset": {
            "description": "Indian online grocery products (27,000+ items)",
            "platform": "Kaggle",
            "url": "https://www.kaggle.com/surajjha101/bigbasket-entire-product-list-28k-datapoints",
            "items": "Product catalog with categories, brands, prices",
            "use_case": "Product relationships, pricing analysis"
        },
        
        "2. Indian E-commerce Dataset": {
            "description": "Flipkart/Amazon India product transactions",
            "platform": "Kaggle",
            "url": "https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products",
            "items": "50K+ products, categories, ratings",
            "use_case": "Cross-category purchase patterns"
        },
        
        "3. Grocery Sales India Dataset": {
            "description": "Indian grocery store sales data",
            "platform": "Kaggle",
            "url": "https://www.kaggle.com/datasets/shelvigarg/sales-forecasting-dataset",
            "items": "Time-series sales data for Indian stores",
            "use_case": "Sales forecasting, seasonal patterns"
        },
        
        "4. Indian Retail Store Dataset": {
            "description": "Supermarket transactions from India",
            "platform": "UCI/Kaggle",
            "url": "https://www.kaggle.com/datasets/mohammadkaiftahir/retail-store-sales-data",
            "items": "Transaction data with Indian products",
            "use_case": "Direct market basket analysis"
        },
        
        "5. Reliance Fresh/Big Bazaar Inspired": {
            "description": "Indian hypermarket shopping patterns",
            "platform": "Data.gov.in / Academic",
            "url": "https://data.gov.in",
            "items": "Government retail data portals",
            "use_case": "Real Indian shopping behavior"
        },
        
        "6. Grofers/Blinkit Dataset": {
            "description": "Quick commerce India data",
            "platform": "GitHub/Research papers",
            "url": "Search: 'Indian grocery dataset site:github.com'",
            "items": "Quick delivery patterns",
            "use_case": "Urban Indian shopping trends"
        }
    }
    
    print("\n📊 Available Real Indian Datasets:\n")
    for name, info in datasets.items():
        print(f"\n{name}")
        print(f"   Description: {info['description']}")
        print(f"   Platform: {info['platform']}")
        print(f"   Items: {info['items']}")
        print(f"   Use Case: {info['use_case']}")
        print(f"   URL: {info['url']}")
    
    print("\n" + "="*80)
    print("🎯 RECOMMENDED: BigBasket + Indian Retail Store datasets")
    print("="*80)
    print("\nThese provide actual Indian product names and shopping patterns")
    

def download_bigbasket_sample():
    """
    Try to download BigBasket dataset programmatically
    Note: Kaggle datasets require API key
    """
    print("\n" + "="*80)
    print("📥 Attempting to download BigBasket dataset...")
    print("="*80)
    
    print("\n⚠️  To download Kaggle datasets programmatically:")
    print("\n1. Create Kaggle account: https://www.kaggle.com")
    print("2. Go to Account Settings → API → Create New API Token")
    print("3. Save kaggle.json to: ~/.kaggle/kaggle.json (Linux/Mac)")
    print("   Or: C:\\Users\\<YourUsername>\\.kaggle\\kaggle.json (Windows)")
    print("4. Install: pip install kaggle")
    print("5. Run:")
    print("   kaggle datasets download -d surajjha101/bigbasket-entire-product-list-28k-datapoints")
    print("   kaggle datasets download -d mohammadkaiftahir/retail-store-sales-data")
    
    print("\n📝 OR download manually from the URLs above and place in data/ folder")


def check_for_datasets():
    """Check if any Indian datasets already exist in data folder"""
    print("\n" + "="*80)
    print("🔍 Checking for existing datasets in data/ folder...")
    print("="*80)
    
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("✅ Created data/ folder")
    
    files = os.listdir(data_dir)
    if files:
        print(f"\n📁 Found {len(files)} files:")
        for f in files:
            size = os.path.getsize(os.path.join(data_dir, f)) / 1024
            print(f"   - {f} ({size:.1f} KB)")
    else:
        print("\n⚠️  No datasets found. Please download from the URLs above.")


def main():
    print("\n" + "="*80)
    print("🇮🇳 REAL INDIAN SUPERMARKET DATA - DOWNLOAD GUIDE")
    print("="*80)
    
    # Check existing data
    check_for_datasets()
    
    # List available datasets
    search_indian_datasets()
    
    # Download instructions
    download_bigbasket_sample()
    
    print("\n" + "="*80)
    print("✅ NEXT STEPS")
    print("="*80)
    print("\n1. Download BigBasket dataset (recommended)")
    print("   → Has actual Indian product names (Amul, Britannia, Haldirams, etc.)")
    print("\n2. Download Indian Retail Store Sales dataset")
    print("   → Has transaction patterns from Indian supermarkets")
    print("\n3. Place CSV files in the data/ folder")
    print("\n4. Run: python process_indian_data.py")
    print("   → Will process and prepare data for training")
    print("\n5. Run: python train_models_indian.py")
    print("   → Train models on real Indian data")


if __name__ == '__main__':
    main()
