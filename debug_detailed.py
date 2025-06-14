import sys
sys.path.append('src')
from automl.model_selector import ModelSelector
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pandas as pd
import numpy as np

print('🔍 Detailed debugging...')

# Create exact same data as AutoML pipeline
X, y = make_classification(n_samples=100, n_features=4, random_state=42)
X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])

# Simulate feature engineering output
from automl.feature_engineer import FeatureEngineer
from automl.data_analyzer import DataAnalyzer
from automl.utils import validate_input_data

analyzer = DataAnalyzer(verbose=False)
analysis = analyzer.analyze(X_df, y)
feature_engineer = FeatureEngineer(analyzer_results=analysis, verbose=False)
X_clean, y_clean = validate_input_data(X_df, y)
X_processed, pipeline = feature_engineer.fit_transform(X_clean, y_clean)

print(f"X_processed type: {type(X_processed)}")
print(f"y_clean type: {type(y_clean)}")
print(f"y_clean values type: {type(y_clean.values)}")

# Test exact same cross-validation that's failing
print('\n🧪 Testing exact cross-validation setup...')

try:
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = 'accuracy'
    
    # Test with original pandas Series
    print('Test 1: With pandas Series')
    scores1 = cross_val_score(model, X_processed, y_clean, cv=cv, scoring=scoring, n_jobs=1)
    print(f'  ✅ Works: {scores1.mean():.3f}')
    
except Exception as e:
    print(f'  ❌ Pandas Series failed: {e}')
    
    try:
        # Test with numpy array
        print('Test 2: With numpy array')
        scores2 = cross_val_score(model, X_processed, y_clean.values, cv=cv, scoring=scoring, n_jobs=1)
        print(f'  ✅ Works: {scores2.mean():.3f}')
    except Exception as e2:
        print(f'  ❌ Numpy array also failed: {e2}')
        import traceback
        traceback.print_exc()
