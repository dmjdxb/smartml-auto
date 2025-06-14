import sys
sys.path.append('src')
from automl.model_selector import ModelSelector
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np

print('🔍 Testing data types...')

# Create data
X_numpy, y_numpy = make_classification(n_samples=100, n_features=4, random_state=42)
X_pandas = pd.DataFrame(X_numpy, columns=[f'feature_{i}' for i in range(4)])
y_pandas = pd.Series(y_numpy)

print(f"Numpy shapes: X={X_numpy.shape}, y={y_numpy.shape}")
print(f"Pandas shapes: X={X_pandas.shape}, y={y_pandas.shape}")

# Test 1: Numpy arrays
print('\n📊 Test 1: Numpy arrays')
selector1 = ModelSelector(verbose=False)
results1 = selector1.fit(X_numpy, y_numpy, task='classification', time_budget='fast')
print(f"Success: {results1.get('success')}")

# Test 2: Pandas DataFrames  
print('\n📊 Test 2: Pandas DataFrames')
selector2 = ModelSelector(verbose=False)
results2 = selector2.fit(X_pandas, y_pandas, task='classification', time_budget='fast')
print(f"Success: {results2.get('success')}")

# Test 3: Mixed (pandas X, numpy y)
print('\n📊 Test 3: Mixed types')
selector3 = ModelSelector(verbose=False)
results3 = selector3.fit(X_pandas, y_numpy, task='classification', time_budget='fast')
print(f"Success: {results3.get('success')}")
