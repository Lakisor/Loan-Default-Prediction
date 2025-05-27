import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
import mlflow
import mlflow.sklearn
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
)
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

mlruns_dir = Path.cwd() / "mlruns"
mlruns_dir.mkdir(exist_ok=True)
mlflow.set_tracking_uri(mlruns_dir.as_uri())

RANDOM_STATE = 42
N_JOBS = -1
CV_SPLITS = 5
METRICS = {
    'accuracy': accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'f1': f1_score,
    'roc_auc': roc_auc_score
}

class ModelTrainer:
    def __init__(self, models_dir='models', results_dir='results', experiment_name='LoanDefaultPrediction'):
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.experiment_name = experiment_name
        self.preprocessor = None
        self.create_directories()
        self.categorical_columns = ['DTI_Group', 'CreditScore_Group']
        
        mlflow.set_experiment(experiment_name)
        self.client = mlflow.tracking.MlflowClient()
        
    def create_directories(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'plots').mkdir(exist_ok=True)
    
    def _preprocess_data(self, X, y=None, fit=False):
        missing_columns = [col for col in self.categorical_columns if col not in X.columns]
        if missing_columns:
            self.categorical_columns = [col for col in self.categorical_columns if col not in missing_columns]
        
        X_processed = X.copy()
        
        if fit:
            from sklearn.preprocessing import OneHotEncoder
            self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            
            encoded_cols = self.encoder.fit_transform(X_processed[self.categorical_columns])
            
            feature_names = []
            for i, col in enumerate(self.categorical_columns):
                categories = self.encoder.categories_[i]
                for category in categories:
                    feature_names.append(f"{col}_{category}")
            
            encoded_df = pd.DataFrame(
                encoded_cols,
                columns=feature_names,
                index=X_processed.index
            )
            
            X_processed = X_processed.drop(columns=self.categorical_columns)
            X_processed = pd.concat([X_processed, encoded_df], axis=1)
            self.feature_names = X_processed.columns.tolist()
            
        else:
            encoded_cols = self.encoder.transform(X_processed[self.categorical_columns])
            
            feature_names = []
            for i, col in enumerate(self.categorical_columns):
                for category in self.encoder.categories_[i]:
                    feature_names.append(f"{col}_{category}")
            
            encoded_df = pd.DataFrame(
                encoded_cols,
                columns=feature_names,
                index=X_processed.index
            )
            
            X_processed = X_processed.drop(columns=self.categorical_columns)
            X_processed = pd.concat([X_processed, encoded_df], axis=1)
        
        X_processed = X_processed.astype(float)
        return X_processed, y
    
    def load_data(self):
        train_df = pd.read_parquet('data/processed/train.parquet')
        test_df = pd.read_parquet('data/processed/test.parquet')
        
        if 'Default' not in train_df.columns or 'Default' not in test_df.columns:
            raise ValueError("Целевая переменная 'Default' не найдена в данных")
        
        X_train = train_df.drop(columns=['Default'])
        y_train = train_df['Default'].astype(int)
        
        X_test = test_df.drop(columns=['Default'])
        y_test = test_df['Default'].astype(int)
        
        X_train, y_train = self._preprocess_data(X_train, y_train, fit=True)
        X_test, y_test = self._preprocess_data(X_test, y_test, fit=False)
        
        return X_train, X_test, y_train, y_test
    
    def get_models(self):
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
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'average_precision': average_precision_score(y_true, y_proba)
        }
        
        self.plot_roc_curve(y_true, y_proba, model_name)
        self.plot_precision_recall_curve(y_true, y_proba, model_name)
        
        cm = confusion_matrix(y_true, y_pred)
        self.plot_confusion_matrix(cm, model_name)
        
        return metrics, y_proba
    
    def plot_roc_curve(self, y_true, y_proba, model_name):
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
        
        plot_path = os.path.join(self.results_dir, 'plots', f'roc_curve_{model_name}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curve(self, y_true, y_proba, model_name):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        average_precision = average_precision_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.step(recall, precision, where='post', label=f'AP={average_precision:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc='best')
        
        plot_path = os.path.join(self.results_dir, 'plots', f'precision_recall_curve_{model_name}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, cm, model_name):
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Not Default', 'Default'],
                    yticklabels=['Not Default', 'Default'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Confusion Matrix - {model_name}')
        
        plot_path = os.path.join(self.results_dir, 'plots', f'confusion_matrix_{model_name}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
    
    def cross_validate_model(self, model, X, y, model_name):
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
        
        cv_metrics = {}
        for metric in scoring.keys():
            cv_metrics[f'train_{metric}'] = np.mean(cv_results[f'train_{metric}'])
            cv_metrics[f'val_{metric}'] = np.mean(cv_results[f'test_{metric}'])
        
        return cv_metrics
    
    def save_results(self, results):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(self.results_dir, f'model_results_{timestamp}.json')
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
    
    def train_and_evaluate(self):
        try:
            X_train, X_test, y_train, y_test = self.load_data()
            models = self.get_models()
            results = {}
            
            with mlflow.start_run(run_name='data_characteristics'):
                mlflow.log_params({
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'features': X_train.shape[1],
                    'positive_class_ratio': y_train.mean()
                })
            
            for model_name, model in models.items():
                with mlflow.start_run(run_name=model_name, nested=True) as run:
                    try:
                        params = model.get_params()
                        mlflow.log_params(params)
                        
                        model.fit(X_train, y_train)
                        cv_metrics = self.cross_validate_model(model, X_train, y_train, model_name)
                        test_metrics, y_proba = self.evaluate_model(model, X_test, y_test, model_name)
                        
                        mlflow.log_metrics({
                            'test_accuracy': test_metrics['accuracy'],
                            'test_precision': test_metrics['precision'],
                            'test_recall': test_metrics['recall'],
                            'test_f1': test_metrics['f1'],
                            'test_roc_auc': test_metrics['roc_auc']
                        })
                        
                        for metric_name, metric_value in cv_metrics.items():
                            mlflow.log_metric(f'cv_{metric_name}', metric_value)
                        
                        model_path = self.models_dir / f'{model_name}'
                        mlflow.sklearn.log_model(model, "model")
                        
                        for plot_file in (self.results_dir / 'plots').glob(f'*{model_name}*'):
                            mlflow.log_artifact(str(plot_file), 'plots')
                        
                        results[model_name] = {
                            'cv_metrics': cv_metrics,
                            'test_metrics': test_metrics,
                            'model_uri': f'runs:/{run.info.run_id}/model',
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        mlflow.set_tag('model_type', model_name)
                        mlflow.set_tag('framework', 'scikit-learn')
                        
                    except Exception as e:
                        mlflow.log_param('error', str(e))
                        continue
                    else:
                        mlflow.set_tag('status', 'completed')
            
            self.save_results(results)
            return results
            
        except Exception as e:
            mlflow.set_tag('status', 'failed')
            mlflow.log_param('error', str(e))
            raise

def main():
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f'LoanDefaultPrediction_{timestamp}'
        
        mlruns_dir = Path.cwd() / "mlruns"
        mlruns_dir.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(mlruns_dir.as_uri())
        
        trainer = ModelTrainer(experiment_name=experiment_name)
        trainer.train_and_evaluate()
        
        print(f"\nTraining completed. To view results, run:")
        print(f"mlflow ui --backend-store-uri {mlflow.get_tracking_uri()}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    main()
