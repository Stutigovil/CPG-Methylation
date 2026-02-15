#!/usr/bin/env python3
"""
Example Prediction Script

This script demonstrates how to use the trained models to make predictions
on new methylation data.

Usage:
    python3 predict.py
"""

import pandas as pd
import joblib
import numpy as np

def main():
    print("=" * 70)
    print("PARKINSON'S DISEASE PREDICTION DEMO")
    print("=" * 70)

    # Load the trained model
    print("\n1. Loading trained Random Forest model...")
    model = joblib.load('ml_analysis/models/random_forest.pkl')
    scaler = joblib.load('ml_analysis/models/scaler.pkl')
    print("   ✓ Model loaded successfully")

    # Load test data as example
    print("\n2. Loading test data...")
    X_test = pd.read_parquet('ml_analysis/data/X_test.parquet')
    y_test = pd.read_csv('ml_analysis/data/y_test.csv')
    print(f"   ✓ Loaded {len(X_test)} test samples")
    print(f"   ✓ Features: {X_test.shape[1]} CpG sites")

    # Make predictions on first 10 samples
    print("\n3. Making predictions on 10 random samples...")

    # Select 10 random samples
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), 10, replace=False)
    X_sample = X_test.iloc[sample_indices]
    y_sample = y_test.iloc[sample_indices]

    # Standardize and predict
    X_sample_scaled = scaler.transform(X_sample)
    predictions = model.predict(X_sample_scaled)
    probabilities = model.predict_proba(X_sample_scaled)[:, 1]

    # Display results
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    print(f"{'#':<4} {'Actual':<15} {'Predicted':<15} {'Confidence':<12} {'Match':<5}")
    print("-" * 70)

    correct = 0
    for i, idx in enumerate(sample_indices):
        actual = "Parkinson's" if y_sample.iloc[i]['label'] == 1 else "Control"
        predicted = "Parkinson's" if predictions[i] == 1 else "Control"
        confidence = probabilities[i] if predictions[i] == 1 else (1 - probabilities[i])

        match = "✓" if actual == predicted else "✗"
        if actual == predicted:
            correct += 1

        print(f"{i+1:<4} {actual:<15} {predicted:<15} {confidence:>6.1%}       {match:<5}")

    accuracy = correct / 10 * 100
    print("-" * 70)
    print(f"Accuracy on these 10 samples: {correct}/10 ({accuracy:.0f}%)")
    print("=" * 70)

    # Show model performance summary
    print("\n4. Overall Model Performance (on full test set):")
    print("-" * 70)

    # Calculate full test set metrics
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_prob)

    print(f"  Test Set Size: {len(y_test)} samples")
    print(f"  Accuracy:      {test_accuracy:.1%}")
    print(f"  Precision:     {test_precision:.1%}")
    print(f"  Recall:        {test_recall:.1%}")
    print(f"  F1-Score:      {test_f1:.3f}")
    print(f"  AUC-ROC:       {test_auc:.3f}")
    print("=" * 70)

    # Show confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    print("\n5. Confusion Matrix:")
    print("-" * 70)
    print(f"                    Predicted")
    print(f"                Control    PD")
    print(f"Actual Control     {cm[0,0]:>4}      {cm[0,1]:>4}")
    print(f"       PD          {cm[1,0]:>4}      {cm[1,1]:>4}")
    print("-" * 70)
    print(f"True Positives (PD correctly identified):  {cm[1,1]}")
    print(f"True Negatives (Control correctly identified): {cm[0,0]}")
    print(f"False Positives (Control wrongly called PD): {cm[0,1]}")
    print(f"False Negatives (PD wrongly called Control): {cm[1,0]}")
    print("=" * 70)

    print("\n✓ Prediction demo complete!")
    print("\nTo use this model on your own data:")
    print("  1. Ensure your data has the same 1,000 CpG sites")
    print("  2. Load selected_cpgs.csv to see which CpGs are needed")
    print("  3. Standardize using the scaler before prediction")
    print("  4. Call model.predict() or model.predict_proba()")
    print("\nSee README.md for detailed usage instructions.")


if __name__ == "__main__":
    main()
