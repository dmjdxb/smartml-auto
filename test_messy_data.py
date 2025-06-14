import automl
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

print('⚠️  Testing with MESSY data...')

# Create challenging dataset
X, y = make_classification(n_samples=200, n_features=5, random_state=42)
X_df = pd.DataFrame(X, columns=[f'numeric_{i}' for i in range(5)])

# Add categorical features
np.random.seed(42)
X_df['category_A'] = np.random.choice(['High', 'Medium', 'Low'], size=200)
X_df['category_B'] = np.random.choice(['Type1', 'Type2', 'Type3', 'Type4'], size=200)

# Add missing values
missing_indices = np.random.choice(200, 40, replace=False)
X_df.loc[missing_indices[:20], 'numeric_0'] = np.nan
X_df.loc[missing_indices[20:], 'category_A'] = np.nan

print(f'Dataset with problems:')
print(f'   Shape: {X_df.shape}')
print(f'   Missing values: {X_df.isnull().sum().sum()}')

# Test AutoML on messy data
predictor = automl.train(X_df, y, time_budget='fast', verbose=True)

print(f'✅ Handled messy data successfully!')
print(f'✅ Final accuracy: {predictor.model_analysis["best_score"]:.3f}')
