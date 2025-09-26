import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

output_dir = os.path.join(PROJECT_ROOT ,".datasets/eda")

def eda(merged_df):
    print(f"Shape: {merged_df.shape}")
    print("------")

    print(f"Information: {merged_df.info()}")
    print("------")

    print(f"Describe: {merged_df.describe(include='all')}")
    print("------")

    print(f"\nNull values ? \n{merged_df.isnull().sum().sort_values(ascending=False)}")
    print("------")

    print(f"Duplicated values ? \n{merged_df.duplicated().sum()}")

    for col in merged_df.columns:
        print(f'{col}: ', merged_df[col].nunique())
 
    # save merged_df 
    os.makedirs(output_dir, exist_ok=True)
    merged_df.to_csv(f"{output_dir}/eda_df.csv", index_label=False)

    return merged_df
