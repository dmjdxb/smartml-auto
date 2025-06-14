import sys
sys.path.append('src')
from automl.model_selector import ModelSelector
from sklearn.datasets import make_classification
import pandas as pd

print('🔍 Debugging ModelSelector results...')

X, y = make_classification(n_samples=100, n_features=4, random_state=42)
X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])

selector = ModelSelector(verbose=True)
results = selector.fit(X_df, y, task='classification', time_budget='fast')

print(f"\n📊 Results type: {type(results)}")
print(f"📊 Results keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
print(f"📊 Success: {results.get('success', 'Not found')}")
print(f"📊 Best model: {results.get('best_model', 'Not found')}")
print(f"📊 Best algorithm: {results.get('best_algorithm', 'Not found')}")
print(f"📊 Best score: {results.get('best_score', 'Not found')}")

if results.get('best_model') is not None:
    print("✅ Model is available!")
else:
    print("❌ Model is None!")
    print(f"📊 Error: {results.get('error', 'No error field')}")
