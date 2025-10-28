"""
Streamlit App for Market Basket Analysis
Interactive dashboard for product placement optimization
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

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
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide"
)

# Initialize session state for storing analysis results
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

# Title
st.title("🛒 Market Basket Analysis - Product Placement Optimizer")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Transaction Data (CSV)",
    type=['csv'],
    help="CSV file with 'Transaction' and 'Item' columns"
)

# Parameters
st.sidebar.subheader("Algorithm Parameters")

# Add product limit for memory efficiency
max_products = st.sidebar.number_input(
    "Max Products (for performance)",
    min_value=50,
    max_value=500,
    value=200,
    step=50,
    help="Limit products to avoid memory errors"
)

min_support = st.sidebar.slider(
    "Minimum Support (%)",
    min_value=0.01,
    max_value=5.0,
    value=0.1,
    step=0.01,
    help="Minimum frequency of itemset in transactions"
) / 100

min_confidence = st.sidebar.slider(
    "Minimum Confidence (%)",
    min_value=1,
    max_value=100,
    value=5,
    step=1,
    help="Minimum probability of buying item B when item A is purchased"
) / 100

min_lift = st.sidebar.slider(
    "Minimum Lift",
    min_value=1.0,
    max_value=5.0,
    value=1.2,
    step=0.1,
    help="Minimum lift value (>1 means positive correlation)"
)

# Main content
if uploaded_file is not None:
    try:
        # Read uploaded file
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ File uploaded successfully! ({len(df)} rows)")
        
        # Show data preview
        with st.expander("📋 Data Preview"):
            st.dataframe(df.head(20))
            st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
        
        # Column selection
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
        
        # **FILTER DATA UPFRONT** (before any analysis)
        st.info(f"🎯 Filtering to top {max_products} most popular products for performance...")
        product_counts = df[item_col].value_counts()
        popular_products = product_counts.head(max_products).index
        
        df_filtered = df[df[item_col].isin(popular_products)].copy()
        st.success(f"✅ Filtered: {df[item_col].nunique():,} → {df_filtered[item_col].nunique():,} products")
        st.info(f"� Using {len(df_filtered):,} / {len(df):,} records ({len(df_filtered)/len(df)*100:.1f}%)")
        
        # Get transactions for all tabs
        transactions = df_filtered.groupby(transaction_col)[item_col].apply(list).values.tolist()
        all_products = sorted(list(set([item for transaction in transactions for item in transaction])))
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["🔍 Association Rules", "👥 Customer Segmentation", "🎯 Purchase Prediction"])
        
        # ============================================================================
        # TAB 1: ASSOCIATION RULES (APRIORI)
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
                    
                    # Use pre-filtered transactions
                    analyzer.transactions = transactions
                    
                    # Store in session state
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
                
                # Run market basket analysis
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
                        
                        # Scatter plot
                        fig_scatter = plot_rules_scatter(top_rules)
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        # Placement recommendations
                        st.subheader("💡 Shelf Placement Recommendations")
                        
                        recommendations = generate_placement_recommendations(top_rules, top_n=10)
                        st.dataframe(recommendations, use_container_width=True)
                        
                        # Store rules in session state
                        st.session_state.rules = rules
                        
                        # Export results
                        st.subheader("💾 Export Results")
                        
                        if st.button("Download Results as CSV"):
                            export_results(top_rules, filename='market_basket_results.csv')
                            st.success("✅ Results exported to market_basket_results.csv")
            
            # Product search - OUTSIDE the button so it persists
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
            
            if st.button("🚀 Run Customer Segmentation", type="primary"):
                with st.spinner("Segmenting customers..."):
                    # Initialize segmentation model
                    segmenter = CustomerSegmentation(n_clusters=n_clusters)
                    
                    # Create customer features
                    customer_features = segmenter.create_customer_features(df_filtered, transaction_col, item_col)
                    
                    # Fit model
                    segmenter.fit(customer_features)
                    
                    # Get segment profiles
                    profiles = segmenter.get_segment_profiles()
                    
                    st.success(f"✅ Identified {n_clusters} customer segments!")
                    
                    # Display segment profiles
                    st.subheader("📊 Segment Profiles")
                    st.dataframe(profiles, use_container_width=True)
                    
                    # Show customer distribution
                    st.subheader("👥 Customer Distribution by Segment")
                    segment_counts = customer_features['cluster'].value_counts().sort_index()
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        import plotly.express as px
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
                    
                    # Segment insights
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
            
            # Select target product to predict
            all_products = sorted(list(set([item for transaction in transactions for item in transaction])))
            target_product = st.selectbox("Select Product to Predict:", all_products)
            
            if st.button("🚀 Train Prediction Model", type="primary"):
                with st.spinner(f"Training model to predict '{target_product}' purchases..."):
                    # Initialize prediction model
                    predictor = PurchasePrediction()
                    
                    # Prepare data
                    X_train, X_test, y_train, y_test = predictor.prepare_training_data(
                        df_filtered, target_product, transaction_col, item_col
                    )
                    
                    # Train model
                    predictor.train(X_train, y_train)
                    
                    # Evaluate model
                    report = predictor.evaluate(X_test, y_test)
                    
                    # Store in session state
                    st.session_state.predictor = predictor
                    st.session_state.X_train_columns = X_train.columns.tolist()
                    st.session_state.target_product = target_product
                    
                    st.success(f"✅ Model trained to predict '{target_product}' purchases!")
                    
                    # Display metrics
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
                    
                    import plotly.express as px
                    fig = px.bar(
                        top_features,
                        x='Importance',
                        y='Product',
                        orientation='h',
                        title=f'Top 10 Products that Predict {target_product} Purchase',
                        labels={'Importance': 'Feature Importance', 'Product': 'Product'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Interactive prediction - OUTSIDE the button so it persists
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
                        st.info(f"Customer unlikely to buy '{st.session_state.target_product}' ({probability:.1%} confidence)")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.error("Please make sure your CSV has the correct format with transaction and item columns.")

else:
    # Welcome message
    st.info("👈 Upload a CSV file to begin analysis")
    
    st.markdown("""
    ### 📖 How to Use
    
    1. **Upload Data**: Upload a CSV file with transaction data
       - Required columns: Transaction ID and Item name
    
    2. **Configure Parameters**: Adjust algorithm thresholds in the sidebar
       - **Support**: Minimum frequency of itemset (lower = more results)
       - **Confidence**: Minimum probability threshold (higher = stronger rules)
       - **Lift**: Minimum correlation strength (>1 means positive association)
    
    3. **Run Analysis**: Click "Run Analysis" to generate insights
    
    4. **Explore Results**: View top products, association rules, and placement recommendations
    
    ### 📊 Sample Data Format
    
    Your CSV should look like this:
    
    | Transaction | Item        |
    |-------------|-------------|
    | 1           | Bread       |
    | 1           | Milk        |
    | 1           | Butter      |
    | 2           | Bread       |
    | 2           | Eggs        |
    | 3           | Milk        |
    | 3           | Bread       |
    | 3           | Butter      |
    
    ### 🎯 What You'll Get
    
    - **Top Products**: Most frequently purchased items
    - **Association Rules**: Product combinations with high lift
    - **Placement Recommendations**: Optimal shelf placements to boost sales
    - **Product Search**: Find what customers buy with specific products
    
    ### 📥 Sample Datasets
    
    You can download sample datasets from:
    - [Kaggle - Groceries Dataset](https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset)
    - [UCI - Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail)
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Market Basket Analysis for Retail Optimization")
