import sys
sys.path.append('src')
from automl.data_analyzer import DataAnalyzer
from automl.feature_engineer import FeatureEngineer
from automl.model_selector import ModelSelector
from automl.utils import validate_input_data
from sklearn.datasets import make_classification
import pandas as pd

print('🔍 Debugging AutoML pipeline step by step...')

# Create data
X, y = make_classification(n_samples=100, n_features=4, random_state=42)
X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])

print(f"Original data: X={X_df.shape}, y={len(y)}")

# Step 1: Data Analysis
print('\n📊 Step 1: Data Analysis')
analyzer = DataAnalyzer(verbose=False)
data_analysis = analyzer.analyze(X_df, y)
print(f"Analysis success: {data_analysis.get('task_type', 'unknown')}")

# Step 2: Feature Engineering
print('\n📊 Step 2: Feature Engineering')
feature_engineer = FeatureEngineer(analyzer_results=data_analysis, verbose=False)
X_clean, y_clean = validate_input_data(X_df, y)
X_processed, pipeline = feature_engineer.fit_transform(X_clean, y_clean)

print(f"Processed data types: X={type(X_processed)}, y={type(y_clean)}")
print(f"Processed shapes: X={X_processed.shape}, y={y_clean.shape}")

# Step 3: Model Selection with processed data
print('\n📊 Step 3: Model Selection with processed data')
model_selector = ModelSelector(analyzer_results=data_analysis, verbose=True)
model_results = model_selector.fit(
    X_processed, y_clean,
    task=data_analysis['task_type'],
    time_budget='fast',
    analyzer_results=data_analysis
)

print(f"Model selection success: {model_results.get('success')}")
print(f"Best model: {model_results.get('best_model')}")
print(f"Error: {model_results.get('error', 'None')}")
