import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
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
import os


class SVMModel:
    def __init__(self, random_state=42):
        # Use log_loss for probability support
        self.model = SGDClassifier(loss='log_loss', random_state=random_state)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def fit(self, X, y):
        """Train the model on training data."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        return self
    
    def predict(self, X):
        if not self.is_trained:
            raise Exception("Model is not trained yet.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise Exception("Model is not trained yet.")
        X_scaled = self.scaler.transform(X)
        # With log_loss, predict_proba is available
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_scaled)
            return proba[:, 1] if proba.ndim > 1 else proba
        else:
            # Fallback: sigmoid of decision function
            decision_values = self.model.decision_function(X_scaled)
            return 1 / (1 + np.exp(-decision_values))
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1_score": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_proba),
            "average_precision": average_precision_score(y, y_proba)
        }
        return metrics
    
    def save_model(self, file_path):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }, file_path)
    
    def load_model(self, file_path):
        data = joblib.load(file_path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = data.get('is_trained', True)
    
    def hyperparameter_tuning(self, X, y, param_grid, cv=5):
        X_scaled = self.scaler.fit_transform(X)
        grid_search = GridSearchCV(self.model, param_grid, cv=cv, scoring='accuracy')
        grid_search.fit(X_scaled, y)
        self.model = grid_search.best_estimator_
        self.is_trained = True
        return grid_search.best_params_
    
    def plots(self, X, y, output_dir):
        if not self.is_trained:
            raise Exception("Model is not trained yet.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        y_proba = self.predict_proba(X)
        roc_auc = roc_auc_score(y, y_proba)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y, y_proba)
        avg_precision = average_precision_score(y, y_proba)
        plt.figure()
        plt.plot(recall, precision, label=f'PR curve (area = {avg_precision:.2f})')
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
        
        os.makedirs(output_dir, exist_ok=True)
        
        y_pred = self.predict(X)
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()