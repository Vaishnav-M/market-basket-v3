"""
Market Basket Analysis - Core Analysis Module
Uses Apriori algorithm to find product associations
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


class MarketBasketAnalyzer:
    """
    Performs market basket analysis on transaction data
    """
    
    def __init__(self, min_support=0.01, min_confidence=0.3, min_lift=1.0):
        """
        Initialize analyzer with threshold parameters
        
        Args:
            min_support: Minimum support threshold (default 1%)
            min_confidence: Minimum confidence threshold (default 30%)
            min_lift: Minimum lift threshold (default 1.0)
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.transactions = None
        self.frequent_itemsets = None
        self.rules = None
    
    def load_transactions(self, filepath, transaction_col='Transaction', item_col='Item'):
        """
        Load transaction data from CSV file
        
        Args:
            filepath: Path to CSV file
            transaction_col: Name of transaction ID column
            item_col: Name of item column
        
        Returns:
            DataFrame with transactions
        """
        df = pd.read_csv(filepath)
        
        # Group items by transaction
        transactions = df.groupby(transaction_col)[item_col].apply(list).values.tolist()
        self.transactions = transactions
        
        print(f"Loaded {len(transactions)} transactions")
        return transactions
    
    def encode_transactions(self):
        """
        Convert transaction list to one-hot encoded DataFrame
        
        Returns:
            One-hot encoded DataFrame
        """
        te = TransactionEncoder()
        te_ary = te.fit(self.transactions).transform(self.transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        return df
    
    def find_frequent_itemsets(self, encoded_df=None):
        """
        Find frequent itemsets using Apriori algorithm
        
        Args:
            encoded_df: One-hot encoded transaction DataFrame (optional)
        
        Returns:
            DataFrame with frequent itemsets
        """
        if encoded_df is None:
            encoded_df = self.encode_transactions()
        
        self.frequent_itemsets = apriori(
            encoded_df, 
            min_support=self.min_support, 
            use_colnames=True
        )
        
        print(f"Found {len(self.frequent_itemsets)} frequent itemsets")
        return self.frequent_itemsets
    
    def generate_rules(self, metric='lift'):
        """
        Generate association rules from frequent itemsets
        
        Args:
            metric: Metric to use for sorting rules ('lift', 'confidence', 'support')
        
        Returns:
            DataFrame with association rules
        """
        if self.frequent_itemsets is None:
            raise ValueError("Must find frequent itemsets first!")
        
        self.rules = association_rules(
            self.frequent_itemsets, 
            metric="confidence",
            min_threshold=self.min_confidence
        )
        
        # Filter by lift
        self.rules = self.rules[self.rules['lift'] >= self.min_lift]
        
        # Sort by metric
        self.rules = self.rules.sort_values(by=metric, ascending=False)
        
        print(f"Generated {len(self.rules)} association rules")
        return self.rules
    
    def get_top_rules(self, n=10, metric='lift'):
        """
        Get top N association rules
        
        Args:
            n: Number of top rules to return
            metric: Metric to sort by
        
        Returns:
            DataFrame with top N rules
        """
        if self.rules is None:
            raise ValueError("Must generate rules first!")
        
        return self.rules.nlargest(n, metric)
    
    def get_product_recommendations(self, product):
        """
        Get product recommendations for a given product
        
        Args:
            product: Product name
        
        Returns:
            DataFrame with recommended products
        """
        if self.rules is None:
            raise ValueError("Must generate rules first!")
        
        # Find rules where the product is in antecedents
        recommendations = self.rules[
            self.rules['antecedents'].apply(lambda x: product in x)
        ].sort_values('lift', ascending=False)
        
        return recommendations[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    
    def format_rules_for_display(self, rules_df):
        """
        Format rules for human-readable display
        
        Args:
            rules_df: DataFrame with association rules
        
        Returns:
            DataFrame with formatted rules
        """
        formatted = rules_df.copy()
        formatted['antecedents'] = formatted['antecedents'].apply(lambda x: ', '.join(list(x)))
        formatted['consequents'] = formatted['consequents'].apply(lambda x: ', '.join(list(x)))
        formatted['support'] = formatted['support'].apply(lambda x: f"{x:.2%}")
        formatted['confidence'] = formatted['confidence'].apply(lambda x: f"{x:.2%}")
        formatted['lift'] = formatted['lift'].apply(lambda x: f"{x:.2f}")
        
        return formatted[['antecedents', 'consequents', 'support', 'confidence', 'lift']]


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = MarketBasketAnalyzer(
        min_support=0.01,
        min_confidence=0.3,
        min_lift=1.0
    )
    
    # Load and analyze (assuming data file exists)
    # transactions = analyzer.load_transactions('data/transactions.csv')
    # encoded_df = analyzer.encode_transactions()
    # frequent_itemsets = analyzer.find_frequent_itemsets(encoded_df)
    # rules = analyzer.generate_rules(metric='lift')
    # top_rules = analyzer.get_top_rules(n=10)
    # print(analyzer.format_rules_for_display(top_rules))
    
    print("MarketBasketAnalyzer initialized. Load transaction data to begin analysis.")
