import pandas as pd
from csv_to_latex import csv_to_latex_pandas
import os
from dataset import PROJECT_ROOT
del_feat = 'npc'
dataset_size = 250000
seq_len = 15
input_path = os.path.join(PROJECT_ROOT, "Param-Regression", "src-cpp", "B-spline-curve-fitting", "model_"+str(dataset_size)+"_wo_"+del_feat+"_on_"+str(seq_len)+".csv")
output_file = "csv_analyzer\\model_"+str(dataset_size)+"_wo_"+del_feat+"_on_"+str(15)+".tex"
df = pd.read_csv(input_path)
df = df.transpose()
df = df.iloc[:,[1,0]]
df.columns = ['Top 1 rate','Top 3 rate']
df = df.sort_values('Top 1 rate', ascending=False) / 1000
print(df)
df.to_csv('tmp.csv')
new_df = pd.read_csv('tmp.csv')
csv_to_latex_pandas(new_df, output_file)
