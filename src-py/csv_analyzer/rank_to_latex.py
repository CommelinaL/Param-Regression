import pandas as pd
from csv_to_latex import csv_to_latex_pandas
import os
from dataset import PROJECT_ROOT

seq_len = 30
dataset_size = 250000
del_feat = "npc"
model_name = "MLP"
input_file = os.path.join(PROJECT_ROOT, "Param-Regression", "src-cpp", "B-spline-curve-fitting", f"test_{seq_len}_{model_name}_wo_{del_feat}_{dataset_size}.csv")
output_file = 'csv_analyzer\\rank_30_MLP_wo_npc_250000.tex'
df = pd.read_csv(input_file)
df.drop(columns='Label_local', axis=1, inplace=True)
df.drop(columns='Regressor local', axis=1, inplace=True)
df.drop([0, 1, 2,3,4], axis=0, inplace=True)
df = df.transpose()
print(df)
print(df.sum(axis=0))
print(df.sum(axis=1))
df.to_csv('tmp.csv')
new_df = pd.read_csv('tmp.csv')
# Convert to LaTeX
csv_to_latex_pandas(new_df, output_file, is_int=True)