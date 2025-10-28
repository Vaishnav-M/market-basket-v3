"""
Utility functions for market basket analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter


def get_transaction_stats(transactions):
    """
    Get basic statistics about transactions
    
    Args:
        transactions: List of transaction lists
    
    Returns:
        Dictionary with statistics
    """
    all_items = [item for transaction in transactions for item in transaction]
    
    stats = {
        'total_transactions': len(transactions),
        'total_items_sold': len(all_items),
        'unique_products': len(set(all_items)),
        'avg_items_per_transaction': len(all_items) / len(transactions),
        'min_items': min(len(t) for t in transactions),
        'max_items': max(len(t) for t in transactions)
    }
    
    return stats


def get_top_products(transactions, n=10):
    """
    Get top N most frequently purchased products
    
    Args:
        transactions: List of transaction lists
        n: Number of top products to return
    
    Returns:
        DataFrame with top products and their counts
    """
    all_items = [item for transaction in transactions for item in transaction]
    item_counts = Counter(all_items)
    top_items = item_counts.most_common(n)
    
    df = pd.DataFrame(top_items, columns=['Product', 'Count'])
    df['Percentage'] = (df['Count'] / len(transactions) * 100).round(2)
    
    return df


def plot_top_products(transactions, n=10):
    """
    Plot top N most frequently purchased products
    
    Args:
        transactions: List of transaction lists
        n: Number of top products to plot
    
    Returns:
        Matplotlib figure
    """
    top_products = get_top_products(transactions, n)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top_products, x='Count', y='Product', palette='viridis', ax=ax)
    ax.set_title(f'Top {n} Most Purchased Products', fontsize=16, fontweight='bold')
    ax.set_xlabel('Number of Transactions', fontsize=12)
    ax.set_ylabel('Product', fontsize=12)
    
    return fig


def plot_rules_scatter(rules_df):
    """
    Create scatter plot of support vs confidence colored by lift
    
    Args:
        rules_df: DataFrame with association rules
    
    Returns:
        Plotly figure
    """
    # Convert frozensets to strings for plotly compatibility
    rules_plot = rules_df.copy()
    rules_plot['antecedents_str'] = rules_plot['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules_plot['consequents_str'] = rules_plot['consequents'].apply(lambda x: ', '.join(list(x)))
    
    fig = px.scatter(
        rules_plot,
        x='support',
        y='confidence',
        color='lift',
        size='lift',
        hover_data=['antecedents_str', 'consequents_str'],
        title='Association Rules: Support vs Confidence (colored by Lift)',
        labels={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift',
                'antecedents_str': 'If Customer Buys', 'consequents_str': 'Then Also Buys'},
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title='Support',
        yaxis_title='Confidence',
        hovermode='closest'
    )
    
    return fig


def plot_network_graph(rules_df, min_lift=2.0, max_nodes=20):
    """
    Create network graph of product associations
    
    Args:
        rules_df: DataFrame with association rules
        min_lift: Minimum lift threshold for edges
        max_nodes: Maximum number of nodes to display
    
    Returns:
        Plotly figure
    """
    # Filter rules by lift
    filtered_rules = rules_df[rules_df['lift'] >= min_lift].head(max_nodes)
    
    # Extract unique items
    items = set()
    for _, row in filtered_rules.iterrows():
        items.update(row['antecedents'])
        items.update(row['consequents'])
    
    items = list(items)[:max_nodes]
    
    # Create edges
    edges = []
    for _, row in filtered_rules.iterrows():
        for ant in row['antecedents']:
            for cons in row['consequents']:
                if ant in items and cons in items:
                    edges.append({
                        'source': ant,
                        'target': cons,
                        'lift': row['lift']
                    })
    
    # Simple visualization (you can enhance with networkx if needed)
    edge_text = [f"{e['source']} → {e['target']}<br>Lift: {e['lift']:.2f}" for e in edges]
    
    return {
        'items': items,
        'edges': edges,
        'description': f"Network graph with {len(items)} products and {len(edges)} associations"
    }


def generate_placement_recommendations(rules_df, top_n=10):
    """
    Generate shelf placement recommendations based on association rules
    
    Args:
        rules_df: DataFrame with association rules
        top_n: Number of top recommendations
    
    Returns:
        DataFrame with placement recommendations
    """
    recommendations = []
    
    for _, row in rules_df.head(top_n).iterrows():
        antecedents = ', '.join(list(row['antecedents']))
        consequents = ', '.join(list(row['consequents']))
        
        recommendation = {
            'Product A': antecedents,
            'Product B': consequents,
            'Lift': f"{row['lift']:.2f}",
            'Confidence': f"{row['confidence']:.2%}",
            'Recommendation': f"Place {consequents} near {antecedents} (customers {row['confidence']:.0%} likely to buy both)"
        }
        
        recommendations.append(recommendation)
    
    return pd.DataFrame(recommendations)


def export_results(rules_df, filename='market_basket_results.csv'):
    """
    Export association rules to CSV
    
    Args:
        rules_df: DataFrame with association rules
        filename: Output filename
    """
    export_df = rules_df.copy()
    export_df['antecedents'] = export_df['antecedents'].apply(lambda x: ', '.join(list(x)))
    export_df['consequents'] = export_df['consequents'].apply(lambda x: ', '.join(list(x)))
    
    export_df.to_csv(filename, index=False)
    print(f"Results exported to {filename}")
