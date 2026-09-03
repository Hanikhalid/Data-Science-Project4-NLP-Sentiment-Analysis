import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("Dataset for Data Analytics.csv")

# Display initial information
print("--- Data Info ---")
df.info()

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- First 5 Rows ---")
print(df.head())