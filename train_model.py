import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
)
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Константы
RANDOM_STATE = 42
N_JOBS = -1  # Использовать все ядра процессора
CV_SPLITS = 5
METRICS = {
    'accuracy': accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'f1': f1_score,
    'roc_auc': roc_auc_score
}

class ModelTrainer:
    def __init__(self, models_dir='models', results_dir='results'):
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.preprocessor = None
        self.create_directories()
        
        # Define categorical columns that need encoding
        self.categorical_columns = ['DTI_Group', 'CreditScore_Group']
        
    def create_directories(self):
        """Создает необходимые директории."""
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'plots'), exist_ok=True)
    
    def _preprocess_data(self, X, y=None, fit=False):
        """Предобработка данных: кодирование категориальных признаков."""
        logger.info(f"Начало предобработки данных. fit={fit}")
        logger.info(f"Исходные колонки: {X.columns.tolist()}")
        logger.info(f"Категориальные колонки: {self.categorical_columns}")
        
        # Проверяем наличие категориальных колонок
        missing_columns = [col for col in self.categorical_columns if col not in X.columns]
        if missing_columns:
            logger.warning(f"Следующие категориальные колонки отсутствуют в данных: {missing_columns}")
            # Удаляем отсутствующие колонки из списка для обработки
            self.categorical_columns = [col for col in self.categorical_columns if col not in missing_columns]
        
        X_processed = X.copy()
        
        # Применяем one-hot кодирование к категориальным признакам
        if fit:
            from sklearn.preprocessing import OneHotEncoder
            self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            
            # Кодируем категориальные признаки
            logger.info(f"Применение OneHotEncoder к колонкам: {self.categorical_columns}")
            encoded_cols = self.encoder.fit_transform(X_processed[self.categorical_columns])
            
            # Создаем имена для закодированных столбцов
            feature_names = []
            for i, col in enumerate(self.categorical_columns):
                categories = self.encoder.categories_[i]
                logger.info(f"Колонка {col} имеет категории: {categories}")
                for category in categories:
                    feature_name = f"{col}_{category}"
                    feature_names.append(feature_name)
            
            logger.info(f"Создано {len(feature_names)} новых бинарных признаков")
            
            # Создаем DataFrame с закодированными признаками
            encoded_df = pd.DataFrame(
                encoded_cols,
                columns=feature_names,
                index=X_processed.index
            )
            
            # Удаляем исходные категориальные столбцы и добавляем закодированные
            X_processed = X_processed.drop(columns=self.categorical_columns)
            X_processed = pd.concat([X_processed, encoded_df], axis=1)
            
            # Сохраняем имена признаков для последующего использования
            self.feature_names = X_processed.columns.tolist()
            
        else:
            # Для тестовых данных используем уже обученный кодировщик
            encoded_cols = self.encoder.transform(X_processed[self.categorical_columns])
            
            # Создаем DataFrame с закодированными признаками
            feature_names = []
            for i, col in enumerate(self.categorical_columns):
                for category in self.encoder.categories_[i]:
                    feature_names.append(f"{col}_{category}")
            
            encoded_df = pd.DataFrame(
                encoded_cols,
                columns=feature_names,
                index=X_processed.index
            )
            
            # Удаляем исходные категориальные столбцы и добавляем закодированные
            X_processed = X_processed.drop(columns=self.categorical_columns)
            X_processed = pd.concat([X_processed, encoded_df], axis=1)
        
        # Преобразуем все в числовой формат
        logger.info("Преобразование всех признаков в числовой формат...")
        try:
            X_processed = X_processed.astype(float)
            logger.info("Преобразование в числовой формат успешно завершено")
        except Exception as e:
            logger.error(f"Ошибка при преобразовании в числовой формат: {str(e)}")
            # Выводим информацию о проблемных колонках
            for col in X_processed.columns:
                try:
                    pd.to_numeric(X_processed[col])
                except Exception:
                    logger.error(f"Проблемная колонка: {col}, типы значений: {X_processed[col].apply(type).unique()}")
            raise
            
        logger.info(f"Предобработка завершена. Итоговые размеры: {X_processed.shape}")
        logger.info(f"Итоговые колонки: {X_processed.columns.tolist()}")
        
        return X_processed, y
    
    def load_data(self):
        """Загружает обработанные данные и разделяет на признаки и целевую переменную."""
        logger.info("Загрузка обработанных данных...")
        try:
            # Загружаем данные
            train_df = pd.read_parquet('data/processed/train.parquet')
            test_df = pd.read_parquet('data/processed/test.parquet')
            
            logger.info(f"Загружены данные. Размеры: train={train_df.shape}, test={test_df.shape}")
            
            # Проверяем наличие целевой переменной
            if 'Default' not in train_df.columns or 'Default' not in test_df.columns:
                raise ValueError("Целевая переменная 'Default' не найдена в данных")
            
            # Отделяем целевую переменную
            X_train = train_df.drop(columns=['Default'])
            y_train = train_df['Default'].astype(int)
            
            X_test = test_df.drop(columns=['Default'])
            y_test = test_df['Default'].astype(int)
            
            # Проверяем баланс классов
            logger.info(f"Распределение классов в обучающей выборке: \n{y_train.value_counts(normalize=True).to_dict()}")
            
            # Предобработка данных
            logger.info("Предобработка данных...")
            X_train, y_train = self._preprocess_data(X_train, y_train, fit=True)
            X_test, y_test = self._preprocess_data(X_test, y_test, fit=False)
            
            logger.info(f"Разделение данных выполнено. Признаки: {X_train.shape[1]}, train samples: {len(X_train)}, test samples: {len(X_test)}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {str(e)}", exc_info=True)
            raise
    
    def get_models(self):
        """Возвращает словарь моделей для обучения."""
        return {
            'LogisticRegression': LogisticRegression(
                max_iter=1000, 
                random_state=RANDOM_STATE,
                class_weight='balanced',
                n_jobs=N_JOBS
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS
            ),
            'XGBoost': XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS
            )
        }
    
    def evaluate_model(self, model, X, y_true, model_name):
        """Оценивает модель и возвращает метрики."""
        # Предсказания
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        # Рассчитываем метрики
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'average_precision': average_precision_score(y_true, y_proba)
        }
        
        # Матрица ошибок
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC и PR кривые
        self.plot_roc_curve(y_true, y_proba, model_name)
        self.plot_precision_recall_curve(y_true, y_proba, model_name)
        self.plot_confusion_matrix(cm, model_name)
        
        return metrics, cm
    
    def plot_roc_curve(self, y_true, y_proba, model_name):
        """Строит ROC-кривую."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        
        # Сохраняем график
        plot_path = os.path.join(self.results_dir, 'plots', f'roc_curve_{model_name}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curve(self, y_true, y_proba, model_name):
        """Строит Precision-Recall кривую."""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                 label=f'Precision-Recall (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        
        # Сохраняем график
        plot_path = os.path.join(self.results_dir, 'plots', f'pr_curve_{model_name}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, cm, model_name):
        """Визуализирует матрицу ошибок."""
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Not Default', 'Default'],
                    yticklabels=['Not Default', 'Default'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Confusion Matrix - {model_name}')
        
        # Сохраняем график
        plot_path = os.path.join(self.results_dir, 'plots', f'confusion_matrix_{model_name}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
    
    def cross_validate_model(self, model, X, y, model_name):
        """Выполняет кросс-валидацию модели."""
        print(f"\nКросс-валидация модели {model_name}...")
        
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc',
            'average_precision': 'average_precision'
        }
        
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        cv_results = cross_validate(
            model, X, y, 
            cv=cv, 
            scoring=scoring,
            n_jobs=N_JOBS,
            return_train_score=True,
            return_estimator=False
        )
        
        # Вычисляем средние значения метрик
        cv_metrics = {}
        for metric in scoring.keys():
            cv_metrics[f'train_{metric}'] = np.mean(cv_results[f'train_{metric}'])
            cv_metrics[f'val_{metric}'] = np.mean(cv_results[f'test_{metric}'])
        
        return cv_metrics
    
    def save_results(self, results):
        """Сохраняет результаты обучения."""
        # Сохраняем метрики в JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(self.results_dir, f'model_results_{timestamp}.json')
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nРезультаты сохранены в {results_file}")
    
    def train_and_evaluate(self):
        """Обучает и оценивает модели."""
        try:
            # Загружаем данные
            logger.info("Загрузка данных...")
            X_train, X_test, y_train, y_test = self.load_data()
            
            # Получаем модели
            logger.info("Инициализация моделей...")
            models = self.get_models()
            
            results = {}
            
            for model_name, model in models.items():
                logger.info(f"\n{'='*50}")
                logger.info(f"Обучение модели: {model_name}")
                logger.info(f"{'='*50}")
                
                try:
                    # Обучаем модель
                    logger.info("Обучение модели...")
                    model.fit(X_train, y_train)
                    logger.info("Обучение завершено успешно")
                    
                    # Кросс-валидация
                    logger.info("Проведение кросс-валидации...")
                    cv_metrics = self.cross_validate_model(model, X_train, y_train, model_name)
                    
                    # Оценка на тестовом наборе
                    logger.info("Оценка на тестовом наборе...")
                    test_metrics, _ = self.evaluate_model(model, X_test, y_test, model_name)
                    
                    # Сохраняем модель
                    os.makedirs(self.models_dir, exist_ok=True)
                    model_path = os.path.join(self.models_dir, f'{model_name}.joblib')
                    joblib.dump(model, model_path)
                    logger.info(f"Модель сохранена в {model_path}")
                    
                    # Сохраняем результаты
                    results[model_name] = {
                        'cv_metrics': cv_metrics,
                        'test_metrics': test_metrics,
                        'model_path': model_path,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Выводим результаты
                    logger.info("\nРезультаты кросс-валидации:")
                    for metric, value in cv_metrics.items():
                        logger.info(f"{metric}: {value:.4f}")
                    
                    logger.info("\nРезультаты на тестовом наборе:")
                    for metric, value in test_metrics.items():
                        logger.info(f"{metric}: {value:.4f}")
                        
                except Exception as e:
                    logger.error(f"Ошибка при обучении модели {model_name}: {str(e)}", exc_info=True)
                    continue
            
            # Сохраняем все результаты
            self.save_results(results)
            logger.info("Обучение и оценка моделей завершены")
            
            return results
            
        except Exception as e:
            logger.error(f"Критическая ошибка в train_and_evaluate: {str(e)}", exc_info=True)
            raise
        
        # Сохраняем все результаты
        self.save_results(results)
        
        return results

def main():
    try:
        logger.info("Начало работы скрипта обучения моделей")
        
        # Инициализируем тренер
        logger.info("Инициализация ModelTrainer...")
        trainer = ModelTrainer()
        
        # Обучаем и оцениваем модели
        logger.info("Запуск обучения и оценки моделей...")
        results = trainer.train_and_evaluate()
        
        logger.info("Обучение и оценка моделей завершены успешно!")
        print("\nОбучение и оценка моделей завершены успешно!")
        
    except Exception as e:
        error_msg = f"Произошла ошибка: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"\n{error_msg}")
        import traceback
        traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    main()
