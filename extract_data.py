import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_data():
    if not os.path.exists('data/raw'):
        os.makedirs('data/raw')
    
    df = pd.read_csv('Loan_default.csv')
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_df.to_csv('data/raw/train.csv', index=False)
    test_df.to_csv('data/raw/test.csv', index=False)
    
    print(f"Размер обучающей выборки: {len(train_df)}")
    print(f"Размер тестовой выборки: {len(test_df)}")
    print(f"Процентное соотношение: {len(train_df)/len(df)*100:.1f}% / {len(test_df)/len(df)*100:.1f}%")

if __name__ == "__main__":
    split_data()