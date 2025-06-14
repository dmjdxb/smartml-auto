import sys
sys.path.append('src')

try:
    from automl.model_selector import ModelSelector
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    selector = ModelSelector(verbose=False)
    results = selector.fit(X, y, task='classification', time_budget='fast')
    print("Success!")
except Exception as e:
    print(f'Full error: {e}')
    import traceback
    traceback.print_exc()
