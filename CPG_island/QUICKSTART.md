# Quick Start Guide

**Get up and running in 5 minutes**

---

## For Beginners: Step-by-Step

### Step 1: Install Python Packages (One-time setup)

```bash
cd /home/akash/Downloads/CPG_island
pip install -r requirements.txt
```

Wait for installation to complete (2-3 minutes).

---

### Step 2: Run the Data Processing (First time only)

```bash
python3 merge_methylation_data.py
```

**What this does:** Combines the 16GB methylation data with patient information.

**How long:** ~3 minutes

**Output:** Creates `merged_methylation_data/` folder with 48 files

---

### Step 3: Train the Machine Learning Models (First time only)

```bash
python3 ml_pipeline.py
```

**What this does:** Trains AI models to predict Parkinson's disease.

**How long:** ~2 minutes

**Output:** Creates `ml_analysis/` folder with trained models

---

### Step 4: Use the Models to Make Predictions

Create a file called `predict.py`:

```python
import pandas as pd
import joblib

# Load the trained model
print("Loading model...")
model = joblib.load('ml_analysis/models/random_forest.pkl')
scaler = joblib.load('ml_analysis/models/scaler.pkl')

# Load test data to try it out
X_test = pd.read_parquet('ml_analysis/data/X_test.parquet')
y_test = pd.read_csv('ml_analysis/data/y_test.csv')

# Make predictions on first 5 samples
X_test_scaled = scaler.transform(X_test[:5])
predictions = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)[:, 1]

# Show results
print("\nPredictions for first 5 test samples:")
print("-" * 60)
for i in range(5):
    actual = "Parkinson's" if y_test.iloc[i]['label'] == 1 else "Control"
    predicted = "Parkinson's" if predictions[i] == 1 else "Control"
    confidence = probabilities[i] if predictions[i] == 1 else (1 - probabilities[i])

    match = "✓" if actual == predicted else "✗"
    print(f"Sample {i+1}: Actual={actual:12} | Predicted={predicted:12} | "
          f"Confidence={confidence:.1%} {match}")
```

Run it:
```bash
python3 predict.py
```

**Expected output:**
```
Loading model...

Predictions for first 5 test samples:
------------------------------------------------------------
Sample 1: Actual=Control      | Predicted=Control      | Confidence=72.3% ✓
Sample 2: Actual=Parkinson's  | Predicted=Parkinson's  | Confidence=85.1% ✓
Sample 3: Actual=Control      | Predicted=Parkinson's  | Confidence=55.2% ✗
Sample 4: Actual=Parkinson's  | Predicted=Control      | Confidence=61.4% ✗
Sample 5: Actual=Control      | Predicted=Control      | Confidence=78.9% ✓
```

---

## For Experienced Users: One Command

If you've already processed the data:

```bash
# Just make predictions
python3 -c "
import joblib, pandas as pd
model = joblib.load('ml_analysis/models/random_forest.pkl')
scaler = joblib.load('ml_analysis/models/scaler.pkl')
X = pd.read_parquet('ml_analysis/data/X_test.parquet')
print('Predictions:', model.predict(scaler.transform(X[:10])))
"
```

---

## Visual Outputs

After running the ML pipeline, check these folders:

### 1. Visualizations
```bash
ls ml_analysis/visualizations/
# roc_curves.png - Compare model performance
# feature_importance.png - Top CpG sites
# pca_plot.png - Sample clustering
# confusion_matrices.png - Prediction accuracy
# model_comparison.png - Metrics comparison
```

Open any image:
```bash
xdg-open ml_analysis/visualizations/roc_curves.png
```

### 2. Results
```bash
cat ml_analysis/results/model_performance.csv
# Shows accuracy, AUC, F1-score for each model
```

### 3. Selected Features
```bash
head -20 ml_analysis/feature_selection/selected_cpgs.csv
# Shows the 1,000 CpG sites used in the model
```

---

## Common Questions

**Q: Do I need to run Steps 2-3 every time?**
A: No! Only once. After that, the models are saved and ready to use.

**Q: How do I use my own data?**
A: Your new samples must have methylation values for the same 1,000 CpG sites. Check `ml_analysis/feature_selection/selected_cpgs.csv` for the list.

**Q: Which model is best?**
A: Random Forest (`random_forest.pkl`) performed best with 70.8% AUC-ROC.

**Q: What if I get an error?**
A: Check the [Troubleshooting](README.md#troubleshooting) section in the main README.

---

## Next Steps

1. ✅ **Explore visualizations** - Look at the plots to understand the results
2. ✅ **Read the full README** - Comprehensive documentation with all details
3. ✅ **Analyze top CpG sites** - Find which genomic locations matter most
4. ✅ **Try different thresholds** - Adjust probability cutoffs for predictions

---

**Need help?** See the main [README.md](README.md) for detailed documentation.
