import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,  
    f1_score,
    roc_curve,
    roc_auc_score,  
    precision_recall_curve,
    average_precision_score,  
    confusion_matrix,
    ConfusionMatrixDisplay  
)
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import GridSearchCV
class RandomForestModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.scaler = StandardScaler()
        self.is_trained = False
    def preprocess_data(self, X):
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled
    def predict(self, X):
        if not self.is_trained:
            raise Exception("Model is not trained yet.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    def predict_proba(self, X):
        if not self.is_trained:
            raise Exception("Model is not trained yet.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1_score": f1_score(y, y_pred),
            "roc_auc": roc_auc_score(y, y_proba),
            "average_precision": average_precision_score(y, y_proba)
        }
        return metrics
    def save_model(self, file_path):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, file_path)
    def load_model(self, file_path):
        data = joblib.load(file_path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = True
    def hyperparameter_tuning(self, X, y, param_grid, cv=10):
        X_scaled = self.preprocess_data(X)
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=-1)
        grid_search.fit(X_scaled, y)
        self.model = grid_search.best_estimator_
        self.is_trained = True
        return grid_search.best_params_
    def train(self, X, y):
        X_scaled = self.preprocess_data(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
    def plots(self, X, y, output_dir):
        if not self.is_trained:
            raise Exception("Model is not trained yet.")
        y_proba = self.predict_proba(X)
        fpr, tpr, _ = roc_curve(y, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(y, y_proba))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()
        precision, recall, _ = precision_recall_curve(y, y_proba)
        plt.figure()
        plt.plot(recall, precision, label='Precision-Recall curve (area = %0.2f)' % average_precision_score(y, y_proba))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
        plt.close()
    def confusion_matrix_plot(self, X, y, output_dir):
        if not self.is_trained:
            raise Exception("Model is not trained yet.")
        y_pred = self.predict(X)
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()