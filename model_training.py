import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_and_preprocess_data():
    """Load and preprocess the house price data"""
    print("Loading datasets...")
    
    # Use relative paths for deployment
    train_path = "data/train.csv"
    test_path = "data/test.csv"

    
    df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Training data: {df.shape}, Test data: {test_df.shape}")
    
    return df, test_df

def feature_engineering(df, test_df):
    """Advanced feature engineering for better performance"""
    print("Performing feature engineering...")
    
    # Combine for consistent preprocessing
    combined = pd.concat([df, test_df], axis=0)
    
    # Drop columns with too many missing values
    cols_to_drop = ['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'Id']
    combined = combined.drop([col for col in cols_to_drop if col in combined.columns], axis=1)
    
    # Create new features
    combined['TotalSF'] = combined.get('TotalBsmtSF', 0) + combined.get('1stFlrSF', 0) + combined.get('2ndFlrSF', 0)
    combined['TotalBath'] = combined.get('FullBath', 0) + 0.5 * combined.get('HalfBath', 0) + \
                           combined.get('BsmtFullBath', 0) + 0.5 * combined.get('BsmtHalfBath', 0)
    combined['Age'] = combined.get('YrSold', 2020) - combined.get('YearBuilt', 2000)
    combined['RemodAge'] = combined.get('YrSold', 2020) - combined.get('YearRemodAdd', 2000)
    
    # Fill numerical missing values
    numerical_cols = combined.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if combined[col].isnull().sum() > 0:
            combined[col].fillna(combined[col].median(), inplace=True)
    
    # Fill categorical missing values
    categorical_cols = combined.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if combined[col].isnull().sum() > 0:
            combined[col].fillna('Missing', inplace=True)
    
    # One-hot encoding for categorical variables (limit to top categories)
    categorical_cols = combined.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        top_categories = combined[col].value_counts().head(10).index
        combined[col] = combined[col].apply(lambda x: x if x in top_categories else 'Other')
    
    combined_encoded = pd.get_dummies(combined, columns=categorical_cols, drop_first=True)
    
    # Remove duplicate columns
    combined_encoded = combined_encoded.loc[:, ~combined_encoded.columns.duplicated()]
    
    # Split back
    train_processed = combined_encoded.iloc[:len(df)].copy()
    test_processed = combined_encoded.iloc[len(df):].copy()
    
    return train_processed, test_processed

def train_model(train_processed):
    """Train the XGBoost model with improved parameters"""
    print("Training improved model...")
    
    # Prepare features and target
    X = train_processed.drop(['SalePrice'], axis=1)
    y = train_processed['SalePrice']
    
    # Train-test split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Use better parameters based on house price prediction best practices
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        early_stopping_rounds=50
    )
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )
    
    # Predictions and evaluation
    y_pred = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    r2 = r2_score(y_valid, y_pred)
    
    print(f"Validation RMSE: ${rmse:,.2f}")
    print(f"Validation R² Score: {r2:.4f}")
    print(f"Average Price: ${y_valid.mean():,.2f}")
    print(f"RMSE as % of Average: {(rmse/y_valid.mean())*100:.2f}%")
    
    return model, X.columns.tolist(), rmse, r2

def save_model_and_features(model, feature_names, rmse, r2):
    """Save the trained model and feature information"""
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model, 'models/house_price_model.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    # Save model info
    model_info = {
        'rmse': rmse,
        'r2': r2,
        'feature_count': len(feature_names),
        'model_type': 'XGBoost'
    }
    joblib.dump(model_info, 'models/model_info.pkl')
    
    print("Model and features saved successfully!")
    print(f"Model trained with {len(feature_names)} features")

if __name__ == "__main__":
    # Load data
    df, test_df = load_and_preprocess_data()
    
    # Feature engineering
    train_processed, test_processed = feature_engineering(df, test_df)
    
    # Train model
    model, feature_names, rmse, r2 = train_model(train_processed)
    
    # Save everything
    save_model_and_features(model, feature_names, rmse, r2)