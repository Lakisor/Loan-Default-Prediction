import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def load_data():
    return pd.read_csv('data/raw/train.csv')

def analyze_data():
    df = load_data()
    
    target = 'Default'
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    correlation = numeric_df.corr()
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    df[target].value_counts().plot(kind='bar')
    plt.title('Распределение целевой переменной')
    
    plt.subplot(2, 2, 2)
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Корреляционная матрица')
    
    numeric_cols = numeric_df.columns.drop(target)
    fig, axes = plt.subplots(nrows=2, ncols=len(numeric_cols)//2 + 1, figsize=(15, 10))
    for ax, col in zip(axes.flatten(), numeric_cols):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f'Распределение {col}')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_data()