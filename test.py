import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ecg_features.csv")
print(df.head())  # Show first 5 rows
print("Missing values:\n", df.isnull().sum()) # Check column types & missing values
print(df['label'].value_counts())  # See class distribution
print(df.describe())  # Summary statistics
print(df.columns) 

colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']

numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

plt.figure(figsize=(12, 8))  # Set figure size

for i, col in enumerate(numeric_columns):
    plt.subplot(3, 3, i + 1)  # Adjust layout (3 rows, 3 columns)
    plt.hist(df[col], bins=20, edgecolor='black', color=colors[i % len(colors)], alpha=0.7)
    plt.title(col)  # Set title for each feature
    plt.grid(axis='y', linestyle='--', alpha=0.5)  # Add grid lines for better readability

# Add main title
plt.suptitle("Feature Distribution Histograms", fontsize=16)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# Plot histograms for each feature
#df.hist(figsize=(12, 8), bins=20, edgecolor='black',grid=False, layout=(3, 3) )
#plt.suptitle("Feature Distribution Histograms", fontsize=16)
#plt.show()

