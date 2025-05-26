import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, 
    OneHotEncoder, 
    FunctionTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

# Create necessary directories
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Constants
TARGET = 'Default'
RANDOM_STATE = 42

class FeatureEngineer(BaseEstimator, TransformerMixin):    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Create debt-to-income ratio groups
        X['DTI_Group'] = pd.cut(
            X['DTIRatio'],
            bins=[-np.inf, 0.3, 0.6, np.inf],
            labels=['low', 'medium', 'high']
        )
        
        # Create loan amount to income ratio
        X['LoanToIncomeRatio'] = X['LoanAmount'] / (X['Income'] + 1e-6)
        
        # Create credit score groups
        X['CreditScore_Group'] = pd.cut(
            X['CreditScore'],
            bins=[0, 600, 700, 750, 850],
            labels=['poor', 'fair', 'good', 'excellent']
        )
        
        # Employment stability (longer employment = more stable)
        X['EmploymentStability'] = X['MonthsEmployed'] / (X['Age'] * 12)
        
        # Loan amount to term ratio
        X['MonthlyPayment'] = X['LoanAmount'] / X['LoanTerm']
        
        return X

class DataPreprocessor:
    def __init__(self):
        self.categorical_features = [
            'Education', 'EmploymentType', 'MaritalStatus', 
            'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'
        ]
        
        self.numerical_features = [
            'Age', 'Income', 'LoanAmount', 'CreditScore',
            'MonthsEmployed', 'NumCreditLines', 'InterestRate',
            'LoanTerm', 'DTIRatio'
        ]
        
        self.engineered_features = [
            'DTI_Group', 'LoanToIncomeRatio', 'CreditScore_Group',
            'EmploymentStability', 'MonthlyPayment'
        ]
        
        self.preprocessor = self._create_preprocessor()
    
    def _create_preprocessor(self):
        # Numerical pipeline
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, self.numerical_features),
                ('cat', categorical_pipeline, self.categorical_features),
                ('feat_eng', 'passthrough', self.engineered_features)
            ])
        
        return preprocessor
    
    def _get_feature_names(self, preprocessor, X_engineered):
        """Get feature names after transformation."""
        # Get numerical feature names
        num_features = self.numerical_features
        
        # Get categorical feature names after one-hot encoding
        cat_transformer = preprocessor.named_transformers_['cat']
        cat_features = list(cat_transformer.named_steps['onehot'].get_feature_names_out(self.categorical_features))
        
        # Combine all feature names
        return num_features + cat_features + self.engineered_features
    
    def process(self, X, y=None, fit=False):
        # Feature engineering
        feature_engineer = FeatureEngineer()
        X_engineered = feature_engineer.transform(X)
        
        # Apply preprocessing
        if fit:
            X_processed = self.preprocessor.fit_transform(X_engineered)
            # Get feature names after fitting
            self.feature_names = self._get_feature_names(self.preprocessor, X_engineered)
        else:
            X_processed = self.preprocessor.transform(X_engineered)
            
        return X_processed

def load_data(train_path='data/raw/train.csv', test_path='data/raw/test.csv'):
    """Load and split data into features and target."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=[TARGET, 'LoanID'])
    y_train = train_df[TARGET]
    
    X_test = test_df.drop(columns=[TARGET, 'LoanID'])
    y_test = test_df[TARGET]
    
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test, feature_names):
    """Save processed data to parquet files."""
    # Convert to DataFrames with proper column names
    train_df = pd.DataFrame(X_train, columns=feature_names)
    test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Add target variable
    train_df[TARGET] = y_train.values
    test_df[TARGET] = y_test.values
    
    # Save to parquet
    train_df.to_parquet('data/processed/train.parquet', index=False)
    test_df.to_parquet('data/processed/test.parquet', index=False)
    
    print(f"Processed data saved to data/processed/")
    print(f"Training set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")

def main():
    try:
        print("Starting data processing...")
        
        # Load data
        print("Loading data...")
        X_train, X_test, y_train, y_test = load_data()
        print(f"Loaded {len(X_train)} training samples and {len(X_test)} test samples")
        
        # Initialize preprocessor
        print("Initializing preprocessor...")
        preprocessor = DataPreprocessor()
        
        # Process training data
        print("Processing training data...")
        X_train_processed = preprocessor.process(X_train, y_train, fit=True)
        print(f"Training data processed. Shape: {X_train_processed.shape}")
        
        # Process test data
        print("Processing test data...")
        X_test_processed = preprocessor.process(X_test, y_test, fit=False)
        print(f"Test data processed. Shape: {X_test_processed.shape}")
        
        # Save processed data
        print("Saving processed data...")
        save_processed_data(
            X_train_processed, X_test_processed, 
            y_train, y_test,
            preprocessor.feature_names
        )
        
        # Save preprocessor for inference
        os.makedirs('models', exist_ok=True)
        joblib.dump(preprocessor, 'models/preprocessor.joblib')
        print("\nProcessing completed successfully!")
        print("Preprocessor saved to: models/preprocessor.joblib")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()