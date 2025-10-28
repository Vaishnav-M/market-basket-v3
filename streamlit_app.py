"""
🛒 Instacart Market Basket Analysis - Streamlit App
Real transaction data from 3.4M+ Instacart orders
Discover shopping patterns, customer segments, and product associations
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Page config
st.set_page_config(
    page_title="Instacart Market Basket Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'transactions_df' not in st.session_state:
    st.session_state.transactions_df = None
if 'rules_df' not in st.session_state:
    st.session_state.rules_df = None

# Header
st.markdown('<h1 class="main-header">🛒 Instacart Market Basket Analysis</h1>', unsafe_allow_html=True)
st.markdown("### Discover Shopping Patterns from Real Customer Data")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/000000/shopping-cart.png", width=100)
    st.title("📊 Control Panel")
    
    st.markdown("### 📁 Data Source")
    data_option = st.radio(
        "Choose dataset:",
        ["Instacart 50K (Real Data)", "Upload Custom CSV"],
        help="Use pre-converted Instacart data or upload your own Transaction,Item format CSV"
    )
    
    if data_option == "Upload Custom CSV":
        uploaded_file = st.file_uploader(
            "Upload Transaction CSV",
            type=['csv'],
            help="CSV with columns: Transaction, Item"
        )
    else:
        uploaded_file = None
    
    st.markdown("---")
    st.markdown("### ⚙️ Analysis Settings")
    
    min_support = st.slider(
        "Min Support",
        min_value=0.001,
        max_value=0.1,
        value=0.01,
        step=0.001,
        help="Minimum frequency for itemsets (1% = 0.01)"
    )
    
    min_confidence = st.slider(
        "Min Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Minimum confidence for association rules"
    )
    
    min_lift = st.slider(
        "Min Lift",
        min_value=1.0,
        max_value=5.0,
        value=1.5,
        step=0.1,
        help="Minimum lift (strength) for rules"
    )
    
    st.markdown("---")
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        st.session_state.run_analysis = True
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.info("""
    **Real Instacart Data:**
    - 50,000 customer orders
    - 29,084 unique products
    - Avg 10 items/basket
    - From 3.4M+ orders dataset
    """)

# Load data
@st.cache_data
def load_data(file_path=None):
    """Load transaction data from file or uploaded CSV"""
    if file_path:
        df = pd.read_csv(file_path)
    else:
        # Default Instacart data
        df = pd.read_csv('data/instacart_transactions_50k.csv')
    
    return df

@st.cache_data
def prepare_basket_data(df):
    """Convert transaction format to basket matrix"""
    # Group items by transaction
    baskets = df.groupby('Transaction')['Item'].apply(list).values
    
    # Transform to one-hot encoded matrix
    te = TransactionEncoder()
    te_ary = te.fit(baskets).transform(baskets)
    basket_df = pd.DataFrame(te_ary, columns=te.columns_)
    
    return basket_df, baskets

@st.cache_data
def run_apriori_analysis(basket_df, min_support, min_confidence, min_lift):
    """Run Apriori algorithm and generate association rules"""
    # Find frequent itemsets
    frequent_itemsets = apriori(basket_df, min_support=min_support, use_colnames=True)
    
    if len(frequent_itemsets) == 0:
        return None, None
    
    # Generate rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    if len(rules) == 0:
        return frequent_itemsets, None
    
    # Filter by lift
    rules = rules[rules['lift'] >= min_lift]
    
    # Convert frozensets to strings for display
    rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    
    # Sort by lift
    rules = rules.sort_values('lift', ascending=False)
    
    return frequent_itemsets, rules

# Main content
if not st.session_state.data_loaded or (uploaded_file is not None):
    # Load data
    try:
        if uploaded_file:
            st.session_state.transactions_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Uploaded: {len(st.session_state.transactions_df)} rows")
        else:
            st.session_state.transactions_df = load_data()
            st.success("✅ Loaded Instacart 50K dataset")
        
        st.session_state.data_loaded = True
        
    except FileNotFoundError:
        st.error("""
        ⚠️ **Instacart dataset not found!**
        
        Please run the converter first:
        ```bash
        python convert_instacart_data.py
        ```
        
        Or upload your own CSV file using the sidebar.
        """)
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# Display data overview
if st.session_state.data_loaded:
    df = st.session_state.transactions_df
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📦 Total Purchases", f"{len(df):,}")
    
    with col2:
        n_transactions = df['Transaction'].nunique()
        st.metric("🛒 Unique Baskets", f"{n_transactions:,}")
    
    with col3:
        n_products = df['Item'].nunique()
        st.metric("🏷️ Unique Products", f"{n_products:,}")
    
    with col4:
        avg_basket = len(df) / n_transactions
        st.metric("📊 Avg Items/Basket", f"{avg_basket:.1f}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Association Rules", 
        "📈 Product Analytics", 
        "🛒 Basket Patterns",
        "💡 Recommendations"
    ])
    
    with tab1:
        st.header("🔍 Product Association Rules")
        st.markdown("Discover which products are frequently bought together")
        
        if st.session_state.get('run_analysis', False):
            with st.spinner("🔄 Mining association rules..."):
                # Prepare basket data
                basket_df, baskets = prepare_basket_data(df)
                
                # Run Apriori
                frequent_itemsets, rules = run_apriori_analysis(
                    basket_df, min_support, min_confidence, min_lift
                )
                
                st.session_state.rules_df = rules
                st.session_state.run_analysis = False
        
        if st.session_state.rules_df is not None and len(st.session_state.rules_df) > 0:
            rules = st.session_state.rules_df
            
            st.success(f"✅ Found {len(rules)} strong association rules!")
            
            # Top rules
            st.subheader("🏆 Top 10 Association Rules")
            
            display_cols = ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']
            display_df = rules[display_cols].head(10).copy()
            display_df.columns = ['IF Customer Buys', 'THEN Also Buys', 'Support', 'Confidence', 'Lift']
            
            # Format percentages
            display_df['Support'] = display_df['Support'].apply(lambda x: f"{x*100:.2f}%")
            display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x*100:.1f}%")
            display_df['Lift'] = display_df['Lift'].apply(lambda x: f"{x:.2f}x")
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Visualization
            st.subheader("📊 Rules Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Confidence vs Support scatter
                fig = px.scatter(
                    rules,
                    x='support',
                    y='confidence',
                    size='lift',
                    color='lift',
                    hover_data=['antecedents_str', 'consequents_str'],
                    labels={
                        'support': 'Support',
                        'confidence': 'Confidence',
                        'lift': 'Lift'
                    },
                    title='Rules: Support vs Confidence (sized by Lift)',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Lift distribution
                fig = px.histogram(
                    rules,
                    x='lift',
                    nbins=30,
                    title='Distribution of Lift Values',
                    labels={'lift': 'Lift', 'count': 'Number of Rules'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Download rules
            st.download_button(
                "📥 Download Association Rules (CSV)",
                data=rules.to_csv(index=False),
                file_name="instacart_association_rules.csv",
                mime="text/csv"
            )
            
        elif st.session_state.get('run_analysis', False):
            st.warning("⚠️ No rules found! Try lowering the minimum thresholds.")
        else:
            st.info("👆 Click '🚀 Run Analysis' in the sidebar to discover association rules!")
    
    with tab2:
        st.header("📈 Product Analytics")
        
        # Top products
        st.subheader("🏆 Top 20 Most Popular Products")
        
        product_counts = df['Item'].value_counts().head(20)
        
        fig = px.bar(
            x=product_counts.values,
            y=product_counts.index,
            orientation='h',
            labels={'x': 'Number of Purchases', 'y': 'Product'},
            title='Most Frequently Purchased Products',
            color=product_counts.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False, height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Product search
        st.subheader("🔎 Product Search")
        search_term = st.text_input("Search for a product:", "Banana")
        
        if search_term:
            matching = df[df['Item'].str.contains(search_term, case=False, na=False)]
            
            if len(matching) > 0:
                product_stats = matching.groupby('Item').agg({
                    'Transaction': 'count'
                }).reset_index()
                product_stats.columns = ['Product', 'Purchase Count']
                product_stats = product_stats.sort_values('Purchase Count', ascending=False)
                
                st.dataframe(product_stats, use_container_width=True, hide_index=True)
            else:
                st.warning(f"No products found matching '{search_term}'")
    
    with tab3:
        st.header("🛒 Basket Analysis")
        
        # Basket size distribution
        basket_sizes = df.groupby('Transaction').size()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Basket Size Distribution")
            
            fig = px.histogram(
                basket_sizes,
                nbins=50,
                labels={'value': 'Items in Basket', 'count': 'Number of Baskets'},
                title='How many items do customers buy?'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Basket Size Statistics")
            
            stats_df = pd.DataFrame({
                'Metric': ['Min', 'Max', 'Mean', 'Median', '25th Percentile', '75th Percentile'],
                'Value': [
                    basket_sizes.min(),
                    basket_sizes.max(),
                    f"{basket_sizes.mean():.1f}",
                    basket_sizes.median(),
                    basket_sizes.quantile(0.25),
                    basket_sizes.quantile(0.75)
                ]
            })
            
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Sample baskets
        st.subheader("🛍️ Sample Customer Baskets")
        
        n_samples = st.slider("Number of baskets to show:", 1, 10, 5)
        
        sample_transactions = df['Transaction'].unique()[:n_samples]
        
        for trans_id in sample_transactions:
            basket = df[df['Transaction'] == trans_id]['Item'].tolist()
            
            with st.expander(f"🛒 Basket #{trans_id} ({len(basket)} items)"):
                st.write(", ".join(basket))
    
    with tab4:
        st.header("💡 Smart Recommendations")
        
        st.markdown("""
        ### 🎯 Product Placement Strategies
        
        Based on the association rules discovered from real Instacart data:
        """)
        
        if st.session_state.rules_df is not None and len(st.session_state.rules_df) > 0:
            rules = st.session_state.rules_df
            
            # Strategy 1: Cross-selling opportunities
            st.subheader("🛍️ Cross-Selling Opportunities")
            
            top_rules = rules.nlargest(5, 'lift')
            
            for idx, rule in top_rules.iterrows():
                antecedent = rule['antecedents_str']
                consequent = rule['consequents_str']
                confidence = rule['confidence'] * 100
                lift = rule['lift']
                
                st.info(f"""
                **If customer buys:** {antecedent}  
                **Recommend:** {consequent}  
                📊 {confidence:.0f}% of customers who buy {antecedent} also buy {consequent}  
                💪 {lift:.1f}x stronger than random chance
                """)
            
            # Strategy 2: Product bundling
            st.subheader("📦 Suggested Product Bundles")
            
            bundle_rules = rules[rules['lift'] > 2.0].head(3)
            
            for idx, rule in bundle_rules.iterrows():
                items = list(rule['antecedents']) + list(rule['consequents'])
                st.success(f"**Bundle #{idx+1}:** {', '.join(items)}")
            
        else:
            st.info("Run the analysis to get recommendations!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🛒 Built with Streamlit | Data: Instacart Market Basket Dataset | 50K Real Customer Orders</p>
</div>
""", unsafe_allow_html=True)
