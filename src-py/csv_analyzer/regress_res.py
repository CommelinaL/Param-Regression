import pandas as pd
import numpy as np
from csv_to_latex import csv_to_latex_pandas

# Read the CSV file
input_file = 'results_wo_npc - 250000.csv'
output_file = 'csv_analyzer\\regress_res_wo_npc - 250000.tex'
df = pd.read_csv(input_file)
df.drop(columns=df.columns[0], axis=1, inplace=True)
print(df)
df.drop(['RMSE','R2','MedAE', 'train_time', 'test_time'], axis=1, inplace=True)
df['R2 (normalized)'], df['MedAE (normalized)'] = df['MedAE (normalized)'], df['R2 (normalized)']
print(df)
# Convert to LaTeX
csv_to_latex_pandas(df, output_file)

# Read and print the output
with open(output_file, 'r') as f:
    print(f.read())