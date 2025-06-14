import sys
sys.path.append('src')

# Let's temporarily modify the ModelSelector to show the real error
# First, let's look at the current _train_and_evaluate_model method
print('🔍 Finding the real error...')

# Read the current model_selector.py and find the issue
with open('src/automl/model_selector.py', 'r') as f:
    content = f.read()

# Look for the exception handling around line 200-250
lines = content.split('\n')
for i, line in enumerate(lines[200:280], 200):
    if 'except Exception as e:' in line or 'Model selection failed' in line:
        print(f"Line {i}: {line}")
        if i < len(lines) - 3:
            for j in range(1, 4):
                print(f"Line {i+j}: {lines[i+j]}")
        print()
