#!/usr/bin/env python3
"""
Machine Learning Pipeline for Parkinson's Disease Classification
Using DNA Methylation Data

This script:
1. Loads methylation data from chunked parquet files
2. Performs feature selection to reduce from 473K to manageable features
3. Trains multiple ML models (Logistic Regression, Random Forest, SVM, XGBoost)
4. Evaluates model performance with cross-validation
5. Generates visualizations and statistical reports

Author: Auto-generated
Date: 2025-11-19
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import gc
import warnings
warnings.filterwarnings('ignore')

# Machine learning imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.decomposition import PCA
import joblib

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
MERGED_DATA_DIR = Path('merged_methylation_data')
METADATA_FILE = 'sample_disease_mapping.csv'
CPG_INFO_FILE = 'cpg_info.csv'
OUTPUT_DIR = Path('ml_analysis')
RANDOM_STATE = 42

# Feature selection parameters
VARIANCE_THRESHOLD = 0.01  # Remove CpGs with variance < 0.01
TOP_K_FEATURES = 5000  # Select top 5000 features by ANOVA F-test
FINAL_FEATURES = 1000  # Final number of features after all filtering

# Model parameters
TEST_SIZE = 0.2  # 20% for testing
CV_FOLDS = 5  # 5-fold cross-validation


def print_progress(message):
    """Print timestamped progress message"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def setup_directories():
    """Create output directory structure"""
    print_progress("Setting up output directories...")

    dirs = [
        OUTPUT_DIR,
        OUTPUT_DIR / 'data',
        OUTPUT_DIR / 'feature_selection',
        OUTPUT_DIR / 'models',
        OUTPUT_DIR / 'results',
        OUTPUT_DIR / 'visualizations'
    ]

    for dir_path in dirs:
        dir_path.mkdir(exist_ok=True, parents=True)

    print_progress(f"  Created {len(dirs)} output directories")


def load_metadata():
    """Load patient metadata and prepare labels"""
    print_progress("Loading patient metadata...")

    metadata = pd.read_csv(METADATA_FILE)

    # Create binary labels: 1 = Parkinson's disease, 0 = Control
    metadata['label'] = (metadata['Disease_State'] == "Parkinson's disease").astype(int)

    print_progress(f"  Loaded {len(metadata)} samples")
    print_progress(f"  Parkinson's disease: {metadata['label'].sum()}")
    print_progress(f"  Control: {(metadata['label'] == 0).sum()}")

    return metadata


def preselect_features_from_chunks(y, target_features=TOP_K_FEATURES):
    """
    First pass: scan all chunks and pre-select promising features
    based on variance and correlation with labels.
    This avoids loading all 473K features into memory at once.

    Returns:
        selected_cpgs: List of CpG IDs to load
    """
    print_progress("=" * 60)
    print_progress("INCREMENTAL FEATURE PRE-SELECTION (Memory-efficient)")
    print_progress("=" * 60)

    chunk_files = sorted(MERGED_DATA_DIR.glob('*.parquet'))
    annotation_cols = ['IlmnID', 'CHR', 'MAPINFO', 'Strand', 'Infinium_Design_Type', 'Genome_Build']

    all_candidates = []

    for i, chunk_file in enumerate(chunk_files, 1):
        print_progress(f"Scanning chunk {i}/{len(chunk_files)}...")

        df_chunk = pd.read_parquet(chunk_file)
        cpg_ids = df_chunk['IlmnID'].values
        beta_values = df_chunk.drop(columns=annotation_cols).T  # Transpose
        beta_values.columns = cpg_ids  # Set column names to CpG IDs

        # Calculate variance
        variances = beta_values.var(axis=0)

        # Keep only CpGs with variance > threshold
        high_var_mask = variances > VARIANCE_THRESHOLD
        if high_var_mask.sum() == 0:
            print_progress(f"  No features passed variance threshold in this chunk")
            continue

        beta_filtered = beta_values.loc[:, high_var_mask]
        cpg_filtered = beta_filtered.columns.tolist()

        # Calculate F-scores for filtered features
        from scipy import stats
        f_scores = []
        for cpg in cpg_filtered:
            cpg_values = beta_filtered[cpg].values
            control_vals = cpg_values[y == 0]
            pd_vals = cpg_values[y == 1]
            f_stat, p_val = stats.f_oneway(control_vals, pd_vals)
            f_scores.append((cpg, f_stat, p_val))

        # Keep top features from this chunk
        chunk_top_k = min(1000, len(f_scores))  # Top 1000 per chunk
        f_scores_sorted = sorted(f_scores, key=lambda x: x[1], reverse=True)[:chunk_top_k]

        all_candidates.extend(f_scores_sorted)

        print_progress(f"  Chunk {i}: {len(cpg_ids)} CpGs → {len(cpg_filtered)} passed variance → kept top {len(f_scores_sorted)}")

        del df_chunk, beta_values, beta_filtered
        gc.collect()

    # Select overall top K features
    print_progress(f"\nCombining candidates from all chunks...")
    print_progress(f"  Total candidates: {len(all_candidates):,}")

    all_candidates_sorted = sorted(all_candidates, key=lambda x: x[1], reverse=True)
    selected_cpgs = [cpg for cpg, _, _ in all_candidates_sorted[:target_features]]

    print_progress(f"  Selected top {len(selected_cpgs):,} features for full loading")

    # Save preselected CpGs
    pd.DataFrame({
        'CpG_ID': selected_cpgs,
        'F_score': [f for _, f, _ in all_candidates_sorted[:target_features]],
        'p_value': [p for _, _, p in all_candidates_sorted[:target_features]]
    }).to_csv(OUTPUT_DIR / 'feature_selection' / 'preselected_cpgs.csv', index=False)

    return selected_cpgs


def load_selected_methylation_data(selected_cpgs):
    """
    Second pass: load only the pre-selected CpG sites from all chunks.
    Much more memory-efficient than loading everything.
    """
    print_progress("\nLoading pre-selected CpG sites from chunks...")

    chunk_files = sorted(MERGED_DATA_DIR.glob('*.parquet'))
    annotation_cols = ['IlmnID', 'CHR', 'MAPINFO', 'Strand', 'Infinium_Design_Type', 'Genome_Build']
    selected_cpgs_set = set(selected_cpgs)

    all_selected_data = []

    for i, chunk_file in enumerate(chunk_files, 1):
        df_chunk = pd.read_parquet(chunk_file)

        # Find which selected CpGs are in this chunk
        chunk_cpgs = df_chunk['IlmnID'].values
        selected_in_chunk = [cpg for cpg in chunk_cpgs if cpg in selected_cpgs_set]

        if not selected_in_chunk:
            continue

        # Extract only selected CpGs
        chunk_selected = df_chunk[df_chunk['IlmnID'].isin(selected_in_chunk)]
        beta_values = chunk_selected.drop(columns=annotation_cols).T
        beta_values.columns = chunk_selected['IlmnID'].values

        all_selected_data.append(beta_values)

        print_progress(f"  Chunk {i}/{len(chunk_files)}: Found {len(selected_in_chunk)} selected CpGs")

        del df_chunk, chunk_selected
        gc.collect()

    # Concatenate selected features
    print_progress("  Concatenating selected features...")
    methylation_df = pd.concat(all_selected_data, axis=1)

    # Ensure column order matches selected_cpgs
    methylation_df = methylation_df[[cpg for cpg in selected_cpgs if cpg in methylation_df.columns]]

    print_progress(f"  Loaded matrix shape: {methylation_df.shape}")

    return methylation_df


def perform_feature_selection(X, y, sample_ids):
    """
    Perform multi-stage feature selection to reduce dimensionality.

    Args:
        X: Feature matrix (samples × CpGs)
        y: Labels
        sample_ids: Sample IDs for tracking

    Returns:
        X_selected: Reduced feature matrix
        selected_cpgs: List of selected CpG IDs
        selection_stats: DataFrame with selection statistics
    """
    print_progress("=" * 60)
    print_progress("FEATURE SELECTION PIPELINE")
    print_progress("=" * 60)

    initial_features = X.shape[1]
    cpg_names = X.columns.tolist()

    # Stage 1: Remove low variance features
    print_progress(f"Stage 1: Variance thresholding (threshold={VARIANCE_THRESHOLD})")
    print_progress(f"  Initial features: {initial_features:,}")

    variance_selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    X_var = variance_selector.fit_transform(X)
    selected_var_mask = variance_selector.get_support()
    cpg_names_var = [cpg for cpg, selected in zip(cpg_names, selected_var_mask) if selected]

    print_progress(f"  Features after variance filtering: {len(cpg_names_var):,}")
    print_progress(f"  Removed: {initial_features - len(cpg_names_var):,}")

    # Stage 2: Statistical feature selection (ANOVA F-test)
    print_progress(f"\nStage 2: ANOVA F-test (selecting top {TOP_K_FEATURES:,} features)")

    k_best = min(TOP_K_FEATURES, X_var.shape[1])
    anova_selector = SelectKBest(f_classif, k=k_best)
    X_anova = anova_selector.fit_transform(X_var, y)

    # Get selected feature indices
    selected_anova_indices = anova_selector.get_support(indices=True)
    cpg_names_anova = [cpg_names_var[i] for i in selected_anova_indices]

    # Get F-scores and p-values
    f_scores = anova_selector.scores_[selected_anova_indices]
    p_values = anova_selector.pvalues_[selected_anova_indices]

    print_progress(f"  Features after ANOVA filtering: {len(cpg_names_anova):,}")

    # Stage 3: Remove highly correlated features
    print_progress(f"\nStage 3: Correlation filtering (target={FINAL_FEATURES:,} features)")

    X_anova_df = pd.DataFrame(X_anova, columns=cpg_names_anova, index=sample_ids)

    # Calculate correlation matrix
    print_progress("  Calculating correlation matrix...")
    corr_matrix = X_anova_df.corr().abs()

    # Find highly correlated pairs
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Remove features with correlation > 0.95
    to_drop = [column for column in upper_triangle.columns
               if any(upper_triangle[column] > 0.95)]

    X_final = X_anova_df.drop(columns=to_drop)

    # If still too many features, keep top N by F-score
    if X_final.shape[1] > FINAL_FEATURES:
        # Sort by F-score and keep top FINAL_FEATURES
        feature_scores = pd.DataFrame({
            'CpG': cpg_names_anova,
            'F_score': f_scores,
            'p_value': p_values
        })
        feature_scores = feature_scores[~feature_scores['CpG'].isin(to_drop)]
        feature_scores = feature_scores.nlargest(FINAL_FEATURES, 'F_score')

        X_final = X_final[feature_scores['CpG'].tolist()]

    print_progress(f"  Removed {len(to_drop):,} highly correlated features")
    print_progress(f"  Final feature count: {X_final.shape[1]:,}")

    # Create selection statistics DataFrame
    selected_cpgs = X_final.columns.tolist()
    selection_stats = pd.DataFrame({
        'CpG_ID': cpg_names_anova,
        'F_score': f_scores,
        'p_value': p_values,
        'selected': [cpg in selected_cpgs for cpg in cpg_names_anova]
    })

    # Save selection results
    print_progress("\nSaving feature selection results...")

    # Save selected CpG IDs
    pd.DataFrame({'CpG_ID': selected_cpgs}).to_csv(
        OUTPUT_DIR / 'feature_selection' / 'selected_cpgs.csv',
        index=False
    )

    # Save selection statistics
    selection_stats.to_csv(
        OUTPUT_DIR / 'feature_selection' / 'selection_statistics.csv',
        index=False
    )

    print_progress(f"  Saved selected CpG list and statistics")

    return X_final, selected_cpgs, selection_stats


def split_data(X, y, metadata):
    """Split data into training and test sets"""
    print_progress("Splitting data into train/test sets...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print_progress(f"  Training set: {X_train.shape[0]} samples")
    print_progress(f"  Test set: {X_test.shape[0]} samples")
    print_progress(f"  Train PD/Control: {y_train.sum()}/{(y_train==0).sum()}")
    print_progress(f"  Test PD/Control: {y_test.sum()}/{(y_test==0).sum()}")

    # Save train/test data
    print_progress("  Saving train/test splits...")
    X_train.to_parquet(OUTPUT_DIR / 'data' / 'X_train.parquet')
    X_test.to_parquet(OUTPUT_DIR / 'data' / 'X_test.parquet')
    pd.DataFrame(y_train, columns=['label']).to_csv(OUTPUT_DIR / 'data' / 'y_train.csv', index=False)
    pd.DataFrame(y_test, columns=['label']).to_csv(OUTPUT_DIR / 'data' / 'y_test.csv', index=False)

    return X_train, X_test, y_train, y_test


def train_models(X_train, X_test, y_train, y_test):
    """Train multiple ML models and evaluate performance"""
    print_progress("=" * 60)
    print_progress("MODEL TRAINING & EVALUATION")
    print_progress("=" * 60)

    # Standardize features
    print_progress("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, OUTPUT_DIR / 'models' / 'scaler.pkl')

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
            C=1.0,
            penalty='l2'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            max_depth=10,
            min_samples_split=10,
            n_jobs=-1
        ),
        'SVM (Linear)': SVC(
            kernel='linear',
            random_state=RANDOM_STATE,
            probability=True,
            C=1.0
        )
    }

    results = []
    trained_models = {}
    predictions = {}

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for model_name, model in models.items():
        print_progress(f"\nTraining: {model_name}")

        # Cross-validation on training set
        print_progress(f"  Performing {CV_FOLDS}-fold cross-validation...")
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        print_progress(f"  CV AUC-ROC: {cv_mean:.4f} (+/- {cv_std:.4f})")

        # Train on full training set
        print_progress(f"  Training on full training set...")
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_proba)

        print_progress(f"  Train Accuracy: {train_acc:.4f}")
        print_progress(f"  Test Accuracy: {test_acc:.4f}")
        print_progress(f"  Test AUC-ROC: {test_auc:.4f}")
        print_progress(f"  Test F1-Score: {test_f1:.4f}")

        # Store results
        results.append({
            'Model': model_name,
            'CV_AUC_Mean': cv_mean,
            'CV_AUC_Std': cv_std,
            'Train_Accuracy': train_acc,
            'Test_Accuracy': test_acc,
            'Test_Precision': test_precision,
            'Test_Recall': test_recall,
            'Test_F1': test_f1,
            'Test_AUC': test_auc
        })

        trained_models[model_name] = model
        predictions[model_name] = {
            'y_pred': y_test_pred,
            'y_proba': y_test_proba
        }

        # Save model
        joblib.dump(model, OUTPUT_DIR / 'models' / f"{model_name.replace(' ', '_').lower()}.pkl")

    # Save performance metrics
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'results' / 'model_performance.csv', index=False)

    print_progress("\nModel Performance Summary:")
    print(results_df.to_string(index=False))

    return trained_models, predictions, results_df, scaler, X_test_scaled


def extract_feature_importance(models, feature_names):
    """Extract and save feature importance from models"""
    print_progress("\nExtracting feature importance...")

    importance_data = []

    for model_name, model in models.items():
        if hasattr(model, 'coef_'):  # Linear models
            importance = np.abs(model.coef_[0])
        elif hasattr(model, 'feature_importances_'):  # Tree-based models
            importance = model.feature_importances_
        else:
            continue

        for cpg, imp in zip(feature_names, importance):
            importance_data.append({
                'Model': model_name,
                'CpG_ID': cpg,
                'Importance': imp
            })

    importance_df = pd.DataFrame(importance_data)
    importance_df.to_csv(OUTPUT_DIR / 'feature_selection' / 'feature_importance.csv', index=False)

    print_progress(f"  Saved feature importance for {len(models)} models")

    return importance_df


def generate_visualizations(models, predictions, y_test, X_test, feature_names, results_df):
    """Generate all visualization plots"""
    print_progress("=" * 60)
    print_progress("GENERATING VISUALIZATIONS")
    print_progress("=" * 60)

    viz_dir = OUTPUT_DIR / 'visualizations'

    # 1. ROC Curves
    print_progress("Generating ROC curves...")
    plt.figure(figsize=(10, 8))

    for model_name, pred_dict in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, pred_dict['y_proba'])
        auc = roc_auc_score(y_test, pred_dict['y_proba'])
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Parkinson\'s Disease Classification', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print_progress("  Saved: roc_curves.png")

    # 2. Feature Importance (for Random Forest)
    print_progress("Generating feature importance plot...")
    if 'Random Forest' in models:
        rf_model = models['Random Forest']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:30]  # Top 30 features

        plt.figure(figsize=(12, 8))
        plt.barh(range(30), importances[indices])
        plt.yticks(range(30), [feature_names[i] for i in indices])
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('CpG Sites', fontsize=12)
        plt.title('Top 30 Most Important CpG Sites (Random Forest)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(viz_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print_progress("  Saved: feature_importance.png")

    # 3. PCA Plot
    print_progress("Generating PCA plot...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_test)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, cmap='RdYlBu',
                         s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Disease Status', ticks=[0, 1])
    plt.clim(-0.5, 1.5)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    plt.title('PCA - Test Set Samples', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / 'pca_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print_progress("  Saved: pca_plot.png")

    # 4. Confusion Matrices
    print_progress("Generating confusion matrices...")
    fig, axes = plt.subplots(1, len(predictions), figsize=(5 * len(predictions), 4))
    if len(predictions) == 1:
        axes = [axes]

    for idx, (model_name, pred_dict) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, pred_dict['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Control', 'PD'], yticklabels=['Control', 'PD'])
        axes[idx].set_title(f'{model_name}', fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(viz_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print_progress("  Saved: confusion_matrices.png")

    # 5. Model Performance Comparison
    print_progress("Generating model comparison plot...")
    metrics = ['Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1', 'Test_AUC']

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results_df))
    width = 0.15

    for i, metric in enumerate(metrics):
        offset = width * (i - 2)
        ax.bar(x + offset, results_df[metric], width, label=metric.replace('Test_', ''))

    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print_progress("  Saved: model_comparison.png")

    print_progress(f"  Generated 5 visualization files")


def main():
    """Main execution pipeline"""
    start_time = time.time()

    print_progress("=" * 60)
    print_progress("ML PIPELINE - PARKINSON'S DISEASE CLASSIFICATION")
    print_progress("=" * 60)

    # Setup
    setup_directories()

    # Load metadata
    metadata = load_metadata()
    y = metadata['label'].values

    # Incremental feature pre-selection (memory-efficient)
    selected_cpgs = preselect_features_from_chunks(y, target_features=TOP_K_FEATURES)

    # Load only selected features
    methylation_df = load_selected_methylation_data(selected_cpgs)

    # Align metadata with methylation data
    sample_ids = methylation_df.index.tolist()
    metadata = metadata[metadata['Sample_Title'].isin(sample_ids)]
    metadata = metadata.set_index('Sample_Title').loc[sample_ids].reset_index()

    X = methylation_df
    y = metadata['label'].values

    # Additional feature selection on pre-selected features
    X_selected, selected_cpgs_final, selection_stats = perform_feature_selection(X, y, sample_ids)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X_selected, y, metadata)

    # Train models
    models, predictions, results_df, scaler, X_test_scaled = train_models(
        X_train, X_test, y_train, y_test
    )

    # Feature importance
    importance_df = extract_feature_importance(models, X_selected.columns.tolist())

    # Visualizations
    generate_visualizations(models, predictions, y_test, X_test_scaled,
                          X_selected.columns.tolist(), results_df)

    # Final summary
    elapsed_time = time.time() - start_time
    print_progress("=" * 60)
    print_progress(f"PIPELINE COMPLETE in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print_progress("=" * 60)

    print("\nOutput files created:")
    print(f"  Data: {OUTPUT_DIR}/data/ (train/test splits)")
    print(f"  Feature Selection: {OUTPUT_DIR}/feature_selection/ (selected CpGs, statistics)")
    print(f"  Models: {OUTPUT_DIR}/models/ (trained models, scaler)")
    print(f"  Results: {OUTPUT_DIR}/results/ (performance metrics)")
    print(f"  Visualizations: {OUTPUT_DIR}/visualizations/ (plots and figures)")


if __name__ == "__main__":
    main()
