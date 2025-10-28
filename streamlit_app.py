"""
🛒 Market Basket Analysis - Complete ML Dashboard
Features: Association Rules, Customer Segmentation, Purchase Prediction
Works with Instacart 50K sample OR custom CSV upload
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import plotly.express as px

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from analysis import MarketBasketAnalyzer
from ml_models import CustomerSegmentation, PurchasePrediction
from utils import (
    get_transaction_stats,
    get_top_products,
    plot_top_products,
    plot_rules_scatter,
    generate_placement_recommendations,
    export_results
)

# Page config
st.set_page_config(
    page_title="Market Basket Analysis Dashboard",
    page_icon="🛒",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'rules' not in st.session_state:
    st.session_state.rules = None
if 'transactions' not in st.session_state:
    st.session_state.transactions = None
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'X_train_columns' not in st.session_state:
    st.session_state.X_train_columns = None
if 'target_product' not in st.session_state:
    st.session_state.target_product = None

# Header
st.markdown('<h1 class="main-header">🛒 Market Basket Analysis - ML Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Association Rules · Customer Segmentation · Purchase Prediction")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Data source
st.sidebar.subheader("📁 Data Source")
data_option = st.sidebar.radio(
    "Choose dataset:",
    ["📦 Instacart 50K (Real Data)", "📤 Upload Custom CSV"],
    help="Use pre-loaded Instacart sample or upload your own"
)

uploaded_file = None
if data_option == "📤 Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Transaction Data (CSV)",
        type=['csv'],
        help="CSV file with 'Transaction' and 'Item' columns"
    )
else:
    # Auto-load Instacart data
    uploaded_file = "data/instacart_transactions_50k.csv"

st.sidebar.markdown("---")

# Performance settings
st.sidebar.subheader("⚡ Performance")
max_products = st.sidebar.number_input(
    "Max Products (for speed)",
    min_value=50,
    max_value=1000,
    value=200,
    step=50,
    help="Limit products to avoid memory issues"
)

st.sidebar.markdown("---")

# Algorithm parameters
st.sidebar.subheader("🎛️ Algorithm Parameters")

min_support = st.sidebar.slider(
    "Minimum Support (%)",
    min_value=0.01,
    max_value=5.0,
    value=0.1,
    step=0.01,
    help="Minimum frequency of itemset"
) / 100

min_confidence = st.sidebar.slider(
    "Minimum Confidence (%)",
    min_value=1,
    max_value=100,
    value=5,
    step=1,
    help="Minimum probability threshold"
) / 100

min_lift = st.sidebar.slider(
    "Minimum Lift",
    min_value=1.0,
    max_value=5.0,
    value=1.2,
    step=0.1,
    help="Minimum correlation strength"
)

st.sidebar.markdown("---")
st.sidebar.info("""
**✨ Features:**
- 🔍 Association Rules
- 👥 Customer Segments  
- 🎯 Purchase Prediction

**📊 Instacart Sample:**
- 50K real orders
- 29K unique products
- Avg 10 items/basket
""")

# Main content
if uploaded_file is not None:
    try:
        # Read data
        if isinstance(uploaded_file, str):
            # File path (Instacart sample)
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded Instacart 50K: {len(df):,} rows")
        else:
            # Uploaded file
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded: {len(df):,} rows")
        
        # Show preview
        with st.expander("📋 Data Preview"):
            st.dataframe(df.head(20))
            st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
        
        # Column mapping
        if isinstance(uploaded_file, str):
            # Instacart data has standard columns
            transaction_col = 'Transaction'
            item_col = 'Item'
        else:
            # Let user map columns
            st.sidebar.subheader("Column Mapping")
            transaction_col = st.sidebar.selectbox(
                "Transaction ID Column",
                options=df.columns.tolist(),
                index=0 if 'Transaction' not in df.columns else df.columns.tolist().index('Transaction')
            )
            item_col = st.sidebar.selectbox(
                "Item Column",
                options=df.columns.tolist(),
                index=1 if 'Item' not in df.columns else df.columns.tolist().index('Item')
            )
        
        # Filter data for performance
        st.info(f"🎯 Filtering to top {max_products} most popular products...")
        product_counts = df[item_col].value_counts()
        popular_products = product_counts.head(max_products).index
        
        df_filtered = df[df[item_col].isin(popular_products)].copy()
        st.success(f"✅ Filtered: {df[item_col].nunique():,} → {df_filtered[item_col].nunique():,} products ({len(df_filtered):,} records)")
        
        # Get transactions
        transactions = df_filtered.groupby(transaction_col)[item_col].apply(list).values.tolist()
        all_products = sorted(list(set([item for transaction in transactions for item in transaction])))
        
        st.markdown("---")
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["🔍 Association Rules", "👥 Customer Segmentation", "🎯 Purchase Prediction"])
        
        # ============================================================================
        # TAB 1: ASSOCIATION RULES
        # ============================================================================
        with tab1:
            st.subheader("🔍 Market Basket Analysis (Apriori Algorithm)")
            st.write("Find products that are frequently bought together")
            
            if st.button("🚀 Run Association Rules Analysis", type="primary"):
                with st.spinner("Analyzing transactions..."):
                    # Initialize analyzer
                    analyzer = MarketBasketAnalyzer(
                        min_support=min_support,
                        min_confidence=min_confidence,
                        min_lift=min_lift
                    )
                    
                    analyzer.transactions = transactions
                    st.session_state.analyzer = analyzer
                    st.session_state.transactions = transactions
                    
                    # Get statistics
                    stats = get_transaction_stats(transactions)
                
                # Display statistics
                st.subheader("📊 Transaction Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Transactions", f"{stats['total_transactions']:,}")
                with col2:
                    st.metric("Unique Products", f"{stats['unique_products']:,}")
                with col3:
                    st.metric("Total Items Sold", f"{stats['total_items_sold']:,}")
                with col4:
                    st.metric("Avg Items/Transaction", f"{stats['avg_items_per_transaction']:.2f}")
                
                # Top products
                st.subheader("🏆 Top Products")
                top_products = get_top_products(transactions, n=10)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.dataframe(top_products, use_container_width=True)
                
                with col2:
                    fig = plot_top_products(transactions, n=10)
                    st.pyplot(fig)
                
                # Run analysis
                st.subheader("🔍 Association Rules Analysis")
                
                encoded_df = analyzer.encode_transactions()
                frequent_itemsets = analyzer.find_frequent_itemsets(encoded_df)
                
                if len(frequent_itemsets) == 0:
                    st.warning("⚠️ No frequent itemsets found. Try lowering the minimum support threshold.")
                else:
                    st.success(f"✅ Found {len(frequent_itemsets)} frequent itemsets")
                    
                    rules = analyzer.generate_rules(metric='lift')
                    
                    if len(rules) == 0:
                        st.warning("⚠️ No association rules found. Try lowering the confidence or lift thresholds.")
                    else:
                        st.success(f"✅ Generated {len(rules)} association rules")
                        
                        # Display top rules
                        st.subheader("📈 Top Association Rules")
                        
                        top_n = st.slider("Number of rules to display", min_value=5, max_value=50, value=10)
                        top_rules = analyzer.get_top_rules(n=top_n, metric='lift')
                        
                        formatted_rules = analyzer.format_rules_for_display(top_rules)
                        st.dataframe(formatted_rules, use_container_width=True)
                        
                        # Visualizations
                        st.subheader("📊 Visualizations")
                        
                        fig_scatter = plot_rules_scatter(top_rules)
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        # Placement recommendations
                        st.subheader("💡 Shelf Placement Recommendations")
                        
                        recommendations = generate_placement_recommendations(top_rules, top_n=10)
                        st.dataframe(recommendations, use_container_width=True)
                        
                        st.session_state.rules = rules
                        
                        # Export
                        if st.button("Download Results as CSV"):
                            export_results(top_rules, filename='market_basket_results.csv')
                            st.success("✅ Results exported!")
            
            # Product search - persists after button
            if st.session_state.analyzer is not None and st.session_state.rules is not None:
                st.subheader("🔎 Product Recommendation Search")
                
                all_products_search = sorted(list(set([item for transaction in st.session_state.transactions for item in transaction])))
                selected_product = st.selectbox("Select a product:", all_products_search, key="product_search")
                
                if selected_product:
                    product_recs = st.session_state.analyzer.get_product_recommendations(selected_product)
                    
                    if len(product_recs) > 0:
                        st.write(f"**Products frequently bought with '{selected_product}':**")
                        formatted_recs = st.session_state.analyzer.format_rules_for_display(product_recs)
                        st.dataframe(formatted_recs, use_container_width=True)
                    else:
                        st.info(f"No strong associations found for '{selected_product}'")
        
        # ============================================================================
        # TAB 2: CUSTOMER SEGMENTATION
        # ============================================================================
        with tab2:
            st.subheader("👥 Customer Segmentation (K-Means Clustering)")
            st.write("Group customers based on shopping behavior")
            
            n_clusters = st.slider("Number of Customer Segments", min_value=2, max_value=8, value=4)
            
            if st.button("🚀 Run Customer Segmentation", type="primary", key="seg_btn"):
                with st.spinner("Segmenting customers..."):
                    # Initialize segmentation
                    segmenter = CustomerSegmentation(n_clusters=n_clusters)
                    
                    # Create features
                    customer_features = segmenter.create_customer_features(df_filtered, transaction_col, item_col)
                    
                    # Fit model
                    segmenter.fit(customer_features)
                    
                    # Get profiles
                    profiles = segmenter.get_segment_profiles()
                    
                    st.success(f"✅ Identified {n_clusters} customer segments!")
                    
                    # Display profiles
                    st.subheader("📊 Segment Profiles")
                    st.dataframe(profiles, use_container_width=True)
                    
                    # Distribution
                    st.subheader("👥 Customer Distribution by Segment")
                    segment_counts = customer_features['cluster'].value_counts().sort_index()
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig = px.bar(
                            x=segment_counts.index,
                            y=segment_counts.values,
                            labels={'x': 'Segment ID', 'y': 'Number of Customers'},
                            title='Customer Distribution Across Segments'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        for idx, row in profiles.iterrows():
                            st.metric(
                                row['Segment_Name'],
                                f"{segment_counts[idx]} customers",
                                f"{segment_counts[idx]/len(customer_features)*100:.1f}%"
                            )
                    
                    # Insights
                    st.subheader("💡 Segment Insights")
                    for idx, row in profiles.iterrows():
                        with st.expander(f"{row['Segment_Name']} (Segment {idx})"):
                            st.write(f"**Average Items per Transaction:** {row['num_items']:.1f}")
                            st.write(f"**Product Variety:** {row['unique_categories']:.1f}")
                            st.write(f"**Transaction Value:** ₹{row['transaction_value']:.2f}")
                            st.write(f"**Number of Customers:** {segment_counts[idx]}")
        
        # ============================================================================
        # TAB 3: PURCHASE PREDICTION
        # ============================================================================
        with tab3:
            st.subheader("🎯 Next Purchase Prediction (Random Forest)")
            st.write("Predict what customers will buy based on their current basket")
            
            # Select target product
            target_product = st.selectbox("Select Product to Predict:", all_products)
            
            if st.button("🚀 Train Prediction Model", type="primary", key="pred_btn"):
                with st.spinner(f"Training model to predict '{target_product}' purchases..."):
                    # Initialize predictor
                    predictor = PurchasePrediction()
                    
                    # Prepare data
                    X_train, X_test, y_train, y_test = predictor.prepare_training_data(
                        df_filtered, target_product, transaction_col, item_col
                    )
                    
                    # Train
                    predictor.train(X_train, y_train)
                    
                    # Evaluate
                    report = predictor.evaluate(X_test, y_test)
                    
                    # Store in session
                    st.session_state.predictor = predictor
                    st.session_state.X_train_columns = X_train.columns.tolist()
                    st.session_state.target_product = target_product
                    
                    st.success(f"✅ Model trained to predict '{target_product}' purchases!")
                    
                    # Metrics
                    st.subheader("📊 Model Performance")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Accuracy", f"{report['accuracy']:.2%}")
                    with col2:
                        st.metric("Precision", f"{report['1']['precision']:.2%}")
                    with col3:
                        st.metric("Recall", f"{report['1']['recall']:.2%}")
                    with col4:
                        st.metric("F1-Score", f"{report['1']['f1-score']:.2%}")
                    
                    # Feature importance
                    st.subheader("🔍 What Products Predict This Purchase?")
                    feature_importance = predictor.get_feature_importance(X_train.columns.tolist())
                    
                    top_features = feature_importance.head(10)
                    
                    fig = px.bar(
                        top_features,
                        x='Importance',
                        y='Product',
                        orientation='h',
                        title=f'Top 10 Products that Predict {target_product} Purchase',
                        labels={'Importance': 'Feature Importance', 'Product': 'Product'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Live prediction - persists after button
            if st.session_state.predictor is not None:
                st.subheader("🛒 Test Live Prediction")
                st.write(f"Select products in a customer's basket and see if they'll buy **{st.session_state.target_product}**:")
                
                selected_products = st.multiselect(
                    "Products in basket:",
                    [p for p in all_products if p != st.session_state.target_product],
                    key="prediction_basket"
                )
                
                if selected_products and st.button("Predict", key="predict_btn"):
                    # Create basket
                    basket = {product: 1 if product in selected_products else 0 
                             for product in st.session_state.X_train_columns}
                    
                    # Predict
                    prediction, probability = st.session_state.predictor.predict_next_purchase(basket)
                    
                    if prediction == 1:
                        st.success(f"✅ Customer will likely buy '{st.session_state.target_product}' ({probability:.1%} confidence)")
                        st.write(f"💡 **Recommendation:** Suggest '{st.session_state.target_product}' to this customer!")
                    else:
                        st.info(f"❌ Customer unlikely to buy '{st.session_state.target_product}' ({probability:.1%} confidence)")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.error("Please make sure your CSV has the correct format with transaction and item columns.")

else:
    # Welcome message
    st.info("👈 Select data source from sidebar to begin")
    
    st.markdown("""
    ### 📖 How to Use
    
    1. **Choose Data**: Instacart 50K sample OR upload your own CSV
    2. **Configure**: Adjust algorithm parameters in sidebar
    3. **Analyze**: Click buttons in each tab to run analysis
    4. **Explore**: View results, charts, and recommendations
    
    ### 📊 CSV Format Required
    
    | Transaction | Item |
    |-------------|------|
    | 1 | Banana |
    | 1 | Milk |
    | 2 | Bread |
    
    ### ✨ Features
    
    - **🔍 Association Rules**: Find products bought together
    - **👥 Customer Segments**: Group customers by behavior
    - **🎯 Purchase Prediction**: Predict next purchases
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🛒 Market Basket Analysis Dashboard | Built with Streamlit & Python</p>
</div>
""", unsafe_allow_html=True)
