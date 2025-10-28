"""
Advanced ML Models for Market Basket Analysis
Includes Customer Segmentation (K-Means) and Purchase Prediction
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, silhouette_score
import pickle


class CustomerSegmentation:
    """
    Segment customers based on purchase behavior using K-Means clustering
    """
    
    def __init__(self, n_clusters=4):
        """
        Initialize customer segmentation model
        
        Args:
            n_clusters: Number of customer segments (default 4)
        """
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = StandardScaler()
        self.customer_features = None
        self.cluster_profiles = None
    
    def create_customer_features(self, transactions_df, transaction_col='Transaction', item_col='Item'):
        """
        Create RFM-like features for each customer
        
        Args:
            transactions_df: DataFrame with transaction data
            transaction_col: Name of transaction ID column
            item_col: Name of item column
        
        Returns:
            DataFrame with customer features
        """
        # Group by transaction to get customer-level data
        customer_data = []
        
        for trans_id in transactions_df[transaction_col].unique():
            trans = transactions_df[transactions_df[transaction_col] == trans_id]
            
            features = {
                'customer_id': trans_id,
                'num_items': len(trans),  # Number of items purchased
                'unique_categories': len(trans[item_col].unique()),  # Variety
                'transaction_value': len(trans) * 100  # Proxy for total value in INR (₹100 per item average)
            }
            
            customer_data.append(features)
        
        self.customer_features = pd.DataFrame(customer_data)
        return self.customer_features
    
    def fit(self, customer_features_df=None):
        """
        Fit K-Means clustering model
        
        Args:
            customer_features_df: DataFrame with customer features (optional)
        
        Returns:
            Fitted model
        """
        if customer_features_df is None:
            customer_features_df = self.customer_features
        
        # Select numeric features for clustering
        feature_cols = ['num_items', 'unique_categories', 'transaction_value']
        X = customer_features_df[feature_cols]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit K-Means
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        customer_features_df['cluster'] = self.kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score
        score = silhouette_score(X_scaled, customer_features_df['cluster'])
        
        # Create cluster profiles
        self.cluster_profiles = customer_features_df.groupby('cluster')[feature_cols].mean()
        
        print(f"✅ Customer segmentation complete!")
        print(f"📊 Silhouette Score: {score:.3f}")
        
        return self.kmeans
    
    def predict_segment(self, customer_features):
        """
        Predict customer segment for new data
        
        Args:
            customer_features: Array or DataFrame with customer features
        
        Returns:
            Predicted cluster labels
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted yet!")
        
        X_scaled = self.scaler.transform(customer_features)
        return self.kmeans.predict(X_scaled)
    
    def get_segment_profiles(self):
        """
        Get detailed profiles for each customer segment
        
        Returns:
            DataFrame with segment characteristics
        """
        if self.cluster_profiles is None:
            raise ValueError("Model not fitted yet!")
        
        profiles = self.cluster_profiles.copy()
        
        # Sort by transaction value to assign meaningful names
        profiles_sorted = profiles.sort_values('transaction_value', ascending=False)
        
        # Add descriptive labels based on ranking
        segment_names = []
        n_segments = len(profiles_sorted)
        
        for i, (idx, row) in enumerate(profiles_sorted.iterrows()):
            # Determine tier based on position
            if i < n_segments * 0.25:  # Top 25%
                if row['unique_categories'] > profiles['unique_categories'].median():
                    name = "💎 VIP Shoppers"
                else:
                    name = "🛒 Bulk Buyers"
            elif i < n_segments * 0.5:  # Next 25%
                if row['unique_categories'] > profiles['unique_categories'].median():
                    name = "🌟 Premium Shoppers"
                else:
                    name = "� Regular Buyers"
            elif i < n_segments * 0.75:  # Next 25%
                if row['unique_categories'] > profiles['unique_categories'].median():
                    name = "🔍 Variety Seekers"
                else:
                    name = "🛍️ Casual Shoppers"
            else:  # Bottom 25%
                if row['num_items'] < profiles['num_items'].median():
                    name = "⚡ Quick Shoppers"
                else:
                    name = "💰 Budget Shoppers"
            
            segment_names.append((idx, name))
        
        # Map names back to original index order
        name_dict = dict(segment_names)
        profiles['Segment_Name'] = profiles.index.map(name_dict)
        profiles['Segment_ID'] = profiles.index
        
        return profiles[['Segment_ID', 'Segment_Name', 'num_items', 'unique_categories', 'transaction_value']]
    
    def save_model(self, filepath='models/customer_segmentation.pkl'):
        """Save the trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump({'kmeans': self.kmeans, 'scaler': self.scaler}, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/customer_segmentation.pkl'):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.kmeans = data['kmeans']
            self.scaler = data['scaler']
        print(f"Model loaded from {filepath}")


class PurchasePrediction:
    """
    Predict next purchase using Random Forest
    """
    
    def __init__(self):
        """Initialize purchase prediction model"""
        # Use class_weight='balanced' to handle imbalanced classes
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            max_depth=10,
            class_weight='balanced',  # Automatically adjust weights for imbalanced classes
            min_samples_split=5,       # Prevent overfitting
            min_samples_leaf=2         # Prevent overfitting
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_products = None
    
    def create_features(self, transactions_df, transaction_col='Transaction', item_col='Item'):
        """
        Create features for purchase prediction
        
        Args:
            transactions_df: DataFrame with transaction data
            transaction_col: Name of transaction ID column
            item_col: Name of item column
        
        Returns:
            X (features), y (target products)
        """
        # Get all unique products
        all_products = transactions_df[item_col].unique()
        
        # Create binary matrix: transaction x product
        transaction_product_matrix = []
        transaction_ids = []
        
        for trans_id in transactions_df[transaction_col].unique():
            trans_items = transactions_df[transactions_df[transaction_col] == trans_id][item_col].tolist()
            
            # Create binary features
            features = {product: 1 if product in trans_items else 0 for product in all_products}
            transaction_product_matrix.append(features)
            transaction_ids.append(trans_id)
        
        df = pd.DataFrame(transaction_product_matrix)
        self.feature_names = df.columns.tolist()
        
        return df, transaction_ids
    
    def prepare_training_data(self, transactions_df, target_product, transaction_col='Transaction', item_col='Item'):
        """
        Prepare training data for predicting a specific product purchase
        
        Args:
            transactions_df: DataFrame with transaction data
            target_product: Product to predict
            transaction_col: Name of transaction ID column
            item_col: Name of item column
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        X, _ = self.create_features(transactions_df, transaction_col, item_col)
        
        # Target: whether customer bought the target product
        y = X[target_product].copy()
        
        # Features: all products except the target
        X_features = X.drop(columns=[target_product])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train):
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training labels
        
        Returns:
            Trained model
        """
        # Save feature names
        self.feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        print("✅ Purchase prediction model trained!")
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Classification report
        """
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print("\n📊 Model Performance:")
        print(f"Accuracy: {report['accuracy']:.2%}")
        print(f"Precision: {report['1']['precision']:.2%}")
        print(f"Recall: {report['1']['recall']:.2%}")
        print(f"F1-Score: {report['1']['f1-score']:.2%}")
        
        return report
    
    def predict_next_purchase(self, customer_basket):
        """
        Predict if customer will buy target product
        
        Args:
            customer_basket: Dictionary of products in basket {product: 1/0}
        
        Returns:
            Prediction (0 or 1) and probability
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Create feature vector with correct column order
        if self.feature_names is not None:
            X = pd.DataFrame([{k: customer_basket.get(k, 0) for k in self.feature_names}])
        else:
            X = pd.DataFrame([customer_basket])
        
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0][1]
        
        return prediction, probability
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance for prediction
        
        Args:
            feature_names: List of feature names
        
        Returns:
            DataFrame with feature importance
        """
        importances = self.model.feature_importances_
        
        df = pd.DataFrame({
            'Product': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return df
    
    def save_model(self, filepath='models/purchase_prediction.pkl'):
        """Save the trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model, 
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/purchase_prediction.pkl'):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data.get('feature_names', None)
        print(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    print("Advanced ML Models initialized!")
    print("\n1. Customer Segmentation (K-Means)")
    print("2. Purchase Prediction (Random Forest)")
