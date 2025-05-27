import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List
import os

try:
    plt.style.use('ggplot')
    sns.set_palette("husl")
except:
    plt.style.use('default')
    sns.set_palette("husl")

def load_results(results_dir: str = 'results') -> Dict:
    """Загружает результаты моделей из JSON файла."""
    results_files = list(Path(results_dir).glob('model_results_*.json'))
    if not results_files:
        raise FileNotFoundError("Файлы с результатами не найдены в директории 'results'")
    
    latest_file = max(results_files, key=os.path.getmtime)
    with open(latest_file, 'r') as f:
        return json.load(f)

def plot_metrics_comparison(results: Dict) -> None:
    """Строит сравнение метрик между моделями."""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision']
    model_names = [name for name in results.keys() if 'test_metrics' in results[name]]
    
    data = []
    for model_name in model_names:
        model_metrics = results[model_name]['test_metrics']
        for metric in metrics:
            if metric in model_metrics:
                data.append({
                    'Model': model_name,
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': model_metrics[metric]
                })
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Metric', y='Value', hue='Model', data=df)
    plt.title('Сравнение метрик моделей на тестовом наборе')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', 
                   xytext=(0, 10), 
                   textcoords='offset points')
    
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig('results/plots/metrics_comparison.png')
    plt.close()

def plot_roc_curves(results: Dict) -> None:
    """Строит ROC-кривые для всех моделей."""
    plt.figure(figsize=(10, 8))
    
    for model_name, model_data in results.items():
        if 'roc_curve' in model_data['test_metrics']:
            fpr, tpr, _ = model_data['test_metrics']['roc_curve']
            roc_auc = model_data['test_metrics']['roc_auc']
            plt.plot(fpr, tpr, lw=2, 
                    label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривые моделей')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    plt.savefig('results/plots/roc_curves.png')
    plt.close()

def plot_precision_recall_curves(results: Dict) -> None:
    """Строит Precision-Recall кривые для всех моделей."""
    plt.figure(figsize=(10, 8))
    
    for model_name, model_data in results.items():
        if 'precision_recall_curve' in model_data['test_metrics']:
            precision, recall, _ = model_data['test_metrics']['precision_recall_curve']
            avg_precision = model_data['test_metrics']['average_precision']
            plt.plot(recall, precision, lw=2,
                    label=f'{model_name} (AP = {avg_precision:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall кривые моделей')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    plt.savefig('results/plots/precision_recall_curves.png')
    plt.close()

def plot_feature_importances(results: Dict) -> None:
    """Строит график важности признаков для моделей, которые её поддерживают."""
    for model_name, model_data in results.items():
        if 'feature_importances' in model_data['test_metrics']:
            feature_importance = model_data['test_metrics']['feature_importances']
            feature_names = model_data['test_metrics'].get('feature_names', 
                                                         [f'Feature {i}' for i in range(len(feature_importance))])
            
            indices = np.argsort(feature_importance)[::-1]
            top_n = min(15, len(feature_importance))
            plt.figure(figsize=(12, 8))
            plt.title(f'Топ-{top_n} важных признаков ({model_name})')
            plt.bar(range(top_n), 
                   feature_importance[indices][:top_n], 
                   align='center')
            plt.xticks(range(top_n), 
                      [feature_names[i] for i in indices][:top_n], 
                      rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'results/plots/feature_importance_{model_name.lower().replace(" ", "_")}.png')
            plt.close()

def generate_report(results: Dict) -> None:
    """Генерирует текстовый отчет с основными метриками."""
    report = "# Отчет по результатам обучения моделей\n\n"
    
    for model_name, model_data in results.items():
        report += f"## {model_name}\n\n"
        report += "### Метрики на тестовом наборе\n"
        
        metrics = model_data['test_metrics']
        report += "| Метрика | Значение |\n"
        report += "|---------|----------|\n"
        for metric, value in metrics.items():
            if metric not in ['roc_curve', 'precision_recall_curve', 'confusion_matrix', 'feature_importances', 'feature_names']:
                report += f"| {metric.replace('_', ' ').title()} | {value:.4f} |\n"
        
        report += "\n"
    
    os.makedirs('results', exist_ok=True)
    report_path = os.path.join('results', 'model_report.md')
    with open(report_path, 'w', encoding='utf-8-sig') as f:
        f.write(report)
        
    print(f"Отчет сохранен в {os.path.abspath(report_path)}")

def main():
    """Основная функция для анализа результатов."""
    try:
        results = load_results()
        
        plot_metrics_comparison(results)
        plot_roc_curves(results)
        plot_precision_recall_curves(results)
        plot_feature_importances(results)
        
        generate_report(results)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    main()
