"""
Generate realistic Indian supermarket transaction data
Includes culturally relevant products, shopping patterns, and seasonal trends
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Indian Supermarket Product Categories
INDIAN_PRODUCTS = {
    'Staples & Grains': [
        'Tata Rice 5kg', 'India Gate Basmati 1kg', 'Aashirvaad Atta 10kg',
        'Fortune Chakki Atta 5kg', 'Sona Masoori Rice 5kg', 'Toor Dal 1kg',
        'Moong Dal 500g', 'Chana Dal 1kg', 'Urad Dal 500g', 'Rajma 500g'
    ],
    'Spices & Masala': [
        'MDH Garam Masala', 'Everest Turmeric Powder', 'Red Chilli Powder 100g',
        'Coriander Powder 50g', 'Cumin Seeds 100g', 'Mustard Seeds 50g',
        'Curry Leaves Fresh', 'Kashmiri Red Chilli', 'Biryani Masala',
        'Chaat Masala 50g', 'Kitchen King Masala', 'Pav Bhaji Masala'
    ],
    'Cooking Essentials': [
        'Fortune Sunflower Oil 1L', 'Postman Mustard Oil 500ml', 
        'Rajdhani Besan 1kg', 'Catch Salt 1kg', 'Amul Ghee 500ml',
        'Parachute Coconut Oil 500ml', 'Gingelly Oil 500ml',
        'Aachi Sambar Powder', 'Rasam Powder', 'MTR Ready Mix Dosa'
    ],
    'Dairy & Beverages': [
        'Amul Milk 1L', 'Mother Dairy Curd 400g', 'Amul Butter 100g',
        'Britannia Paneer 200g', 'Nestle Milkmaid', 'Tata Tea Gold 250g',
        'Red Label Tea 500g', 'Bru Coffee 50g', 'Horlicks 500g',
        'Boost 500g', 'Amul Buttermilk 200ml', 'Lassi Cup'
    ],
    'Snacks & Biscuits': [
        'Parle-G Biscuits', 'Good Day Cookies', 'Britannia Marie Gold',
        'Haldirams Bhujia 200g', 'Kurkure Masala Munch', 'Lays Chips',
        'Haldirams Namkeen Mix', 'Bikaji Soan Papdi', 'Britannia Nutrichoice',
        'Hide & Seek Biscuits', 'Krackjack Biscuits', 'Monaco Biscuits'
    ],
    'Instant Foods': [
        'Maggi 2-Minute Noodles', 'Top Ramen', 'Yippee Noodles',
        'MTR Ready to Eat Meals', 'Gits Dosa Mix', 'Haldirams Ready Mix',
        'Kissan Ketchup', 'Maggi Sauce', 'Ching\'s Schezwan Chutney',
        'Ready Idli Mix', 'MTR Upma Mix', 'Knorr Soup'
    ],
    'Household & Personal': [
        'Vim Dishwash Bar', 'Surf Excel 1kg', 'Rin Detergent',
        'Lizol Floor Cleaner', 'Harpic Toilet Cleaner', 'Dettol Soap',
        'Lifebuoy Soap', 'Clinic Plus Shampoo', 'Colgate Toothpaste',
        'Patanjali Toothpaste', 'Santoor Soap', 'Nirma Washing Powder'
    ],
    'Fresh Produce': [
        'Tomatoes 1kg', 'Onions 1kg', 'Potatoes 2kg', 'Green Chillies 100g',
        'Coriander Leaves', 'Curry Leaves Bunch', 'Ginger 200g',
        'Garlic 200g', 'Capsicum 500g', 'Cabbage 1pc',
        'Cauliflower 1pc', 'Beans 500g', 'Carrots 500g'
    ],
    'Sweets & Desserts': [
        'Haldirams Rasgulla', 'MTR Gulab Jamun Mix', 'Amul Ice Cream',
        'Vadilal Ice Cream', 'Britannia Cake', 'Cadbury Dairy Milk',
        'Nestle KitKat', 'Amul Chocolate', 'Parle Monaco', 
        'Bikaji Ladoo', 'Haldirams Soan Papdi', 'Mysore Pak'
    ],
    'Condiments & Pickles': [
        'Mother\'s Recipe Pickle', 'Priya Mango Pickle', 'Nilons Chutney',
        'Patanjali Chyawanprash', 'Kissan Jam', 'MTR Pickle',
        'Andhra Pickle', 'Gongura Pickle', 'Lemon Pickle',
        'Mixed Veg Pickle', 'Garlic Pickle', 'Red Chilli Pickle'
    ]
}

# Shopping patterns by customer type
CUSTOMER_PATTERNS = {
    'Family_Monthly': {
        'avg_items': 35,
        'categories': ['Staples & Grains', 'Spices & Masala', 'Cooking Essentials', 
                      'Dairy & Beverages', 'Household & Personal', 'Fresh Produce'],
        'frequency': 'monthly'
    },
    'Bachelor_Weekly': {
        'avg_items': 12,
        'categories': ['Instant Foods', 'Snacks & Biscuits', 'Dairy & Beverages'],
        'frequency': 'weekly'
    },
    'Festival_Shopper': {
        'avg_items': 50,
        'categories': ['Sweets & Desserts', 'Spices & Masala', 'Cooking Essentials',
                      'Staples & Grains', 'Snacks & Biscuits'],
        'frequency': 'seasonal'
    },
    'Daily_Shopper': {
        'avg_items': 8,
        'categories': ['Fresh Produce', 'Dairy & Beverages', 'Snacks & Biscuits'],
        'frequency': 'daily'
    },
    'Bulk_Buyer': {
        'avg_items': 45,
        'categories': ['Staples & Grains', 'Cooking Essentials', 'Household & Personal',
                      'Spices & Masala', 'Condiments & Pickles'],
        'frequency': 'quarterly'
    }
}

# Regional preferences (different states have different preferences)
REGIONAL_PREFERENCES = {
    'North': ['Aashirvaad Atta 10kg', 'Amul Butter 100g', 'Britannia Paneer 200g', 
              'Patanjali Chyawanprash', 'Haldirams Bhujia 200g'],
    'South': ['Sona Masoori Rice 5kg', 'Aachi Sambar Powder', 'Curry Leaves Fresh',
              'Gingelly Oil 500ml', 'MTR Ready to Eat Meals'],
    'West': ['Fortune Chakki Atta 5kg', 'Pav Bhaji Masala', 'Amul Milk 1L',
             'Britannia Marie Gold', 'Vadilal Ice Cream'],
    'East': ['India Gate Basmati 1kg', 'Postman Mustard Oil 500ml', 'Rasgulla',
             'Priya Mango Pickle', 'Haldirams Soan Papdi']
}

# Festival seasons (higher sales during these periods)
FESTIVALS = {
    'Diwali': {'month': 11, 'boost_categories': ['Sweets & Desserts', 'Snacks & Biscuits', 'Cooking Essentials']},
    'Holi': {'month': 3, 'boost_categories': ['Sweets & Desserts', 'Snacks & Biscuits']},
    'Raksha_Bandhan': {'month': 8, 'boost_categories': ['Sweets & Desserts']},
    'Onam': {'month': 9, 'boost_categories': ['Fresh Produce', 'Cooking Essentials', 'Spices & Masala']},
    'Pongal': {'month': 1, 'boost_categories': ['Staples & Grains', 'Sweets & Desserts']}
}


def generate_transaction(customer_type, region, month, transaction_id):
    """Generate a single transaction"""
    pattern = CUSTOMER_PATTERNS[customer_type]
    
    # Adjust basket size based on season
    is_festival = any(f['month'] == month for f in FESTIVALS.values())
    basket_size = int(pattern['avg_items'] * (1.5 if is_festival else 1) * random.uniform(0.7, 1.3))
    
    # Select products based on customer pattern
    products = []
    for category in pattern['categories']:
        if category in INDIAN_PRODUCTS:
            num_items = max(1, int(basket_size * random.uniform(0.1, 0.3)))
            category_products = random.sample(INDIAN_PRODUCTS[category], 
                                            min(num_items, len(INDIAN_PRODUCTS[category])))
            products.extend(category_products)
    
    # Add regional favorites (20% chance)
    if random.random() < 0.2 and region in REGIONAL_PREFERENCES:
        regional_item = random.choice(REGIONAL_PREFERENCES[region])
        if regional_item not in products:
            products.append(regional_item)
    
    # Create transaction records
    records = []
    for product in products:
        records.append({
            'Transaction': transaction_id,
            'Item': product,
            'CustomerType': customer_type,
            'Region': region,
            'Month': month
        })
    
    return records


def generate_indian_dataset(num_transactions=5000, output_file='data/indian_supermarket.csv'):
    """Generate complete Indian supermarket dataset"""
    
    print(f"🇮🇳 Generating Indian Supermarket Dataset...")
    print(f"   Target Transactions: {num_transactions}")
    
    all_transactions = []
    transaction_id = 1
    
    # Customer type distribution
    customer_types = list(CUSTOMER_PATTERNS.keys())
    customer_weights = [0.35, 0.20, 0.10, 0.25, 0.10]  # Family, Bachelor, Festival, Daily, Bulk
    
    # Region distribution
    regions = list(REGIONAL_PREFERENCES.keys())
    
    # Generate transactions across 12 months
    for _ in range(num_transactions):
        customer_type = random.choices(customer_types, weights=customer_weights)[0]
        region = random.choice(regions)
        month = random.randint(1, 12)
        
        transaction = generate_transaction(customer_type, region, month, transaction_id)
        all_transactions.extend(transaction)
        
        transaction_id += 1
    
    # Create DataFrame
    df = pd.DataFrame(all_transactions)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    # Print statistics
    print(f"\n✅ Dataset Generated Successfully!")
    print(f"   File: {output_file}")
    print(f"   Total Records: {len(df):,}")
    print(f"   Unique Transactions: {df['Transaction'].nunique():,}")
    print(f"   Unique Products: {df['Item'].nunique():,}")
    print(f"\n📊 Customer Type Distribution:")
    for ctype, count in df.groupby('CustomerType')['Transaction'].nunique().items():
        print(f"      {ctype}: {count:,} transactions")
    
    print(f"\n🌍 Regional Distribution:")
    for region, count in df.groupby('Region')['Transaction'].nunique().items():
        print(f"      {region}: {count:,} transactions")
    
    print(f"\n🛒 Top 10 Products:")
    top_products = df['Item'].value_counts().head(10)
    for product, count in top_products.items():
        print(f"      {product}: {count:,} purchases")
    
    return df


if __name__ == '__main__':
    # Generate different dataset sizes
    
    # Small dataset for quick testing (1K transactions)
    print("\n" + "="*80)
    generate_indian_dataset(1000, 'data/indian_supermarket_1k.csv')
    
    # Medium dataset for training (5K transactions)
    print("\n" + "="*80)
    generate_indian_dataset(5000, 'data/indian_supermarket_5k.csv')
    
    # Large dataset for production (20K transactions)
    print("\n" + "="*80)
    generate_indian_dataset(20000, 'data/indian_supermarket.csv')
    
    print("\n" + "="*80)
    print("✅ All Indian supermarket datasets generated successfully!")
    print("\nNext steps:")
    print("1. Run: python train_models_indian.py")
    print("2. Launch: streamlit run app_indian.py")
