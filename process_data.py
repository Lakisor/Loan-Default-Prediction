import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

TARGET = 'Default'
RANDOM_STATE = 42

class FeatureEngineer(BaseEstimator, TransformerMixin):    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        X['DTI_Group'] = pd.cut(
            X['DTIRatio'],
            bins=[-np.inf, 0.3, 0.6, np.inf],
            labels=['low', 'medium', 'high']
        )
        
        X['LoanToIncomeRatio'] = X['LoanAmount'] / (X['Income'] + 1e-6)
        
        X['CreditScore_Group'] = pd.cut(
            X['CreditScore'],
            bins=[0, 600, 700, 750, 850],
            labels=['poor', 'fair', 'good', 'excellent']
        )
        
        X['EmploymentStability'] = X['MonthsEmployed'] / (X['Age'] * 12)
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
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, self.numerical_features),
                ('cat', categorical_pipeline, self.categorical_features),
                ('feat_eng', 'passthrough', self.engineered_features)
            ])
        
        return preprocessor
    
    def _get_feature_names(self, preprocessor, X_engineered):
        num_features = self.numerical_features
        cat_transformer = preprocessor.named_transformers_['cat']
        cat_features = list(cat_transformer.named_steps['onehot'].get_feature_names_out(self.categorical_features))
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
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=[TARGET, 'LoanID'])
    y_train = train_df[TARGET]
    
    X_test = test_df.drop(columns=[TARGET, 'LoanID'])
    y_test = test_df[TARGET]
    
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test, feature_names):
    train_df = pd.DataFrame(X_train, columns=feature_names)
    test_df = pd.DataFrame(X_test, columns=feature_names)
    
    train_df[TARGET] = y_train.values
    test_df[TARGET] = y_test.values
    
    train_df.to_parquet('data/processed/train.parquet', index=False)
    test_df.to_parquet('data/processed/test.parquet', index=False)

def main():
    X_train, X_test, y_train, y_test = load_data()
    preprocessor = DataPreprocessor()
    
    X_train_processed = preprocessor.process(X_train, y_train, fit=True)
    X_test_processed = preprocessor.process(X_test, y_test, fit=False)
    
    save_processed_data(
        X_train_processed, X_test_processed, 
        y_train, y_test,
        preprocessor.feature_names
    )
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.joblib')

if __name__ == "__main__":
    main()