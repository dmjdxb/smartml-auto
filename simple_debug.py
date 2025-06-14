import sys
sys.path.append('src')
from automl.model_selector import ModelSelector
from sklearn.datasets import make_classification
import pandas as pd

# Create data like AutoML does
X, y = make_classification(n_samples=100, n_features=4, random_state=42)
X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])

from automl.feature_engineer import FeatureEngineer
from automl.data_analyzer import DataAnalyzer
from automl.utils import validate_input_data

analyzer = DataAnalyzer(verbose=False)
analysis = analyzer.analyze(X_df, y)
feature_engineer = FeatureEngineer(analyzer_results=analysis, verbose=False)
X_clean, y_clean = validate_input_data(X_df, y)
X_processed, pipeline = feature_engineer.fit_transform(X_clean, y_clean)

# Test ModelSelector with exact same call as AutoML core
print('Testing ModelSelector with AutoML parameters...')

selector = ModelSelector(analyzer_results=analysis, verbose=True)
results = selector.fit(
    X_processed, y_clean,
    task='classification',
    time_budget='fast',
    quality_target=None,
    analyzer_results=analysis
)

print(f"\nFinal Results:")
print(f"Success: {results.get('success')}")
print(f"Best model: {results.get('best_model')}")
print(f"Models tried: {results.get('models_tried')}")
print(f"Error: {results.get('error')}")
