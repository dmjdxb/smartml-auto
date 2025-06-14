import sys
sys.path.append('src')
from automl.model_selector import ModelSelector
from sklearn.datasets import make_classification
import pandas as pd

print('🔍 Debugging ModelSelector _train_and_evaluate_model method...')

# Create exact same setup
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

# Test the exact method that's failing
selector = ModelSelector(analyzer_results=analysis, verbose=False)

# Call the internal method directly
try:
    result = selector._train_and_evaluate_model(
        'random_forest', 'classification', X_processed, y_clean
    )
    print(f"Method result: {result}")
    print(f"Success: {result.get('success')}")
    print(f"Error: {result.get('error', 'None')}")
    
except Exception as e:
    print(f"Method exception: {e}")
    import traceback
    traceback.print_exc()
