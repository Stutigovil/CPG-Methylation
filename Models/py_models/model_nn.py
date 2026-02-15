import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
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
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import optuna
from tqdm import tqdm


# Custom Dataset
class PDDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Neural Network Architecture
class PDNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super(PDNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Main Model Class
class NNModel:
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3,
                 learning_rate=1e-3, batch_size=32, epochs=100, random_state=42):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PDNetwork(input_dim, hidden_dims, dropout_rate).to(self.device)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    def preprocess_data(self, X, fit=True):
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        return X_scaled

    def train(self, X, y, X_val=None, y_val=None, early_stopping_patience=10, verbose=True):
        X_scaled = self.preprocess_data(X, fit=True)
        
        # Create validation set if not provided
        if X_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.2, stratify=y, random_state=self.random_state
            )
        else:
            X_train, y_train = X_scaled, y
            X_val = self.scaler.transform(X_val)
        
        train_dataset = PDDataset(X_train, y_train)
        val_dataset = PDDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Handle class imbalance
        class_counts = np.bincount(y_train.astype(int))
        pos_weight = torch.tensor([class_counts[0] / class_counts[1]], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        
        best_val_auc = 0
        patience_counter = 0
        train_losses = []
        val_aucs = []
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            val_auc = self._evaluate_auc(val_loader)
            val_aucs.append(val_auc)
            scheduler.step(val_auc)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_train_loss:.4f} - Val AUC: {val_auc:.4f}")
            
            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        self.is_trained = True
        
        return {'train_losses': train_losses, 'val_aucs': val_aucs, 'best_val_auc': best_val_auc}

    def _evaluate_auc(self, data_loader):
        self.model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                outputs = torch.sigmoid(self.model(X_batch)).squeeze()
                all_probs.extend(outputs.cpu().numpy())
                all_labels.extend(y_batch.numpy())
        
        return roc_auc_score(all_labels, all_probs)

    def predict(self, X):
        if not self.is_trained:
            raise Exception("Model is not trained yet.")
        
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = torch.sigmoid(self.model(X_tensor)).squeeze()
            predictions = (outputs > 0.5).int().cpu().numpy()
        
        return predictions

    def predict_proba(self, X):
        if not self.is_trained:
            raise Exception("Model is not trained yet.")
        
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = torch.sigmoid(self.model(X_tensor)).squeeze().cpu().numpy()
        
        return outputs

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        specificity = tn / (tn + fp)
        
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),  # sensitivity
            "specificity": specificity,
            "f1_score": f1_score(y, y_pred),
            "roc_auc": roc_auc_score(y, y_proba),
            "average_precision": average_precision_score(y, y_proba)
        }
        return metrics

    def hyperparameter_tuning(self, X, y, n_trials=50, cv=5):
        """Optuna-based hyperparameter tuning"""
        X_scaled = self.scaler.fit_transform(X)
        
        def objective(trial):
            # Hyperparameter search space
            n_layers = trial.suggest_int('n_layers', 1, 4)
            hidden_dims = [trial.suggest_int(f'hidden_dim_{i}', 32, 512) for i in range(n_layers)]
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            
            # Cross-validation
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            cv_scores = []
            
            for train_idx, val_idx in skf.split(X_scaled, y):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = PDNetwork(self.input_dim, hidden_dims, dropout_rate).to(self.device)
                
                train_dataset = PDDataset(X_train, y_train)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                
                class_counts = np.bincount(y_train.astype(int))
                pos_weight = torch.tensor([class_counts[0] / class_counts[1]], device=self.device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                
                # Train for fewer epochs during tuning
                model.train()
                for epoch in range(30):
                    for X_batch, y_batch in train_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        optimizer.zero_grad()
                        outputs = model(X_batch).squeeze()
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        optimizer.step()
                
                # Evaluate
                model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                    val_probs = torch.sigmoid(model(X_val_tensor)).squeeze().cpu().numpy()
                
                cv_scores.append(roc_auc_score(y_val, val_probs))
            
            return np.mean(cv_scores)
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best parameters and retrain
        best_params = study.best_params
        n_layers = best_params['n_layers']
        best_hidden_dims = [best_params[f'hidden_dim_{i}'] for i in range(n_layers)]
        
        self.hidden_dims = best_hidden_dims
        self.dropout_rate = best_params['dropout_rate']
        self.learning_rate = best_params['learning_rate']
        self.batch_size = best_params['batch_size']
        
        # Rebuild and train with best params
        self.model = PDNetwork(self.input_dim, self.hidden_dims, self.dropout_rate).to(self.device)
        self.train(X, y, verbose=False)
        
        return best_params, study.best_value

    def save_model(self, file_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'is_trained': self.is_trained
        }, file_path)

    def load_model(self, file_path):
        checkpoint = torch.load(file_path, map_location=self.device)
        self.scaler = checkpoint['scaler']
        self.input_dim = checkpoint['input_dim']
        self.hidden_dims = checkpoint['hidden_dims']
        self.dropout_rate = checkpoint['dropout_rate']
        self.model = PDNetwork(self.input_dim, self.hidden_dims, self.dropout_rate).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = checkpoint['is_trained']

    def plots(self, X, y, output_dir):
        if not self.is_trained:
            raise Exception("Model is not trained yet.")
        
        os.makedirs(output_dir, exist_ok=True)
        y_proba = self.predict_proba(X)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y, y_proba):.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR curve (AP = {average_precision_score(y, y_proba):.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def confusion_matrix_plot(self, X, y, output_dir):
        if not self.is_trained:
            raise Exception("Model is not trained yet.")
        
        os.makedirs(output_dir, exist_ok=True)
        y_pred = self.predict(X)
        cm = confusion_matrix(y, y_pred)
        
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Control', 'PD'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_training_history(self, history, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(history['train_losses'])
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        
        axes[1].plot(history['val_aucs'])
        axes[1].set_title('Validation AUC')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()