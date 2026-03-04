import pandas as pd
import os

# Check data.xlsx
print("="*60)
print("CHECKING data.xlsx")
print("="*60)
try:
    df_data = pd.read_excel('evaluation/data.xlsx')
    print(f"Shape: {df_data.shape}")
    print(f"\nColumns: {df_data.columns.tolist()}")
    print(f"\nFirst 3 rows:")
    print(df_data.head(3))
    print(f"\nData types:\n{df_data.dtypes}")
except Exception as e:
    print(f"Error: {e}")

# Check corpus.csv
print("\n" + "="*60)
print("CHECKING corpus.csv")
print("="*60)
try:
    df_corpus = pd.read_csv('evaluation/corpus.csv')
    print(f"Shape: {df_corpus.shape}")
    print(f"\nColumns: {df_corpus.columns.tolist()}")
    print(f"\nFirst 3 rows:")
    print(df_corpus.head(3))
    print(f"\nData types:\n{df_corpus.dtypes}")
except Exception as e:
    print(f"Error: {e}")
