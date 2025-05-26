import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Загрузка данных
def load_data():
    return pd.read_csv('data/raw/train.csv')

def analyze_data():
    df = load_data()
    
    # 1. Базовая информация
    print("\n=== Базовая информация о датасете ===")
    print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
    print("\nТипы данных:")
    print(df.dtypes)
    
    # 2. Пропущенные значения
    print("\n=== Пропущенные значения ===")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    # 3. Статистика по числовым признакам
    print("\n=== Статистика по числовым признакам ===")
    print(df.describe())
    
    # 4. Анализ категориальных признаков
    print("\n=== Анализ категориальных признаков ===")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())
    
    # 5. Анализ целевой переменной
    print("\n=== Анализ целевой переменной ===")
    target = 'Default'
    print(df[target].value_counts(normalize=True))
    
    # 6. Корреляционный анализ
    print("\n=== Корреляционный анализ ===")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    correlation = numeric_df.corr()
    print(correlation[target].sort_values(ascending=False))
    
    # 7. Визуализация распределений
    plt.figure(figsize=(15, 10))
    
    # Распределение целевой переменной
    plt.subplot(2, 2, 1)
    df[target].value_counts().plot(kind='bar')
    plt.title('Распределение целевой переменной')
    
    # Корреляционная матрица
    plt.subplot(2, 2, 2)
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Корреляционная матрица')
    
    # Распределение числовых признаков
    numeric_cols = numeric_df.columns.drop(target)
    fig, axes = plt.subplots(nrows=2, ncols=len(numeric_cols)//2 + 1, figsize=(15, 10))
    for ax, col in zip(axes.flatten(), numeric_cols):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f'Распределение {col}')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_data()