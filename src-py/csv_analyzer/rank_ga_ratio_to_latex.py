import pandas as pd
import os
from csv_to_latex import csv_to_latex_pandas
from dataset import PROJECT_ROOT

dataset_size = 250000
del_feat = "npc"
model_name = "mlp"
input_root = os.path.join(PROJECT_ROOT, "Param-Regression", "src-cpp", "B-spline-curve-fitting")
base_list = {seq_len: "test_ga_{len}_{model}_wo_{feat}_{size}.csv".format(len=seq_len, model=model_name, feat=del_feat, size=dataset_size) for seq_len in range(10, 31, 5)}
input_file_list = {seq_len: os.path.join(input_root, base_list[seq_len]) for seq_len in range(10, 31, 5)}
rank_cnt = 3
output_file = 'csv_analyzer\\rank_ga_{cnt}_ratio_wo_{feat} - {size}-transpose.tex'.format(cnt=rank_cnt, feat=del_feat, size=dataset_size)
table = []
for seq_len, input_file in input_file_list.items():
    df = pd.read_csv(input_file)
    df.drop(columns='Label_local', axis=1, inplace=True)
    df.drop(columns='Regressor local', axis=1, inplace=True)
    df.drop([0, 1, 2,3,4], axis=0, inplace=True)
    df = df.transpose()
    print(df)
    print(df.cumsum(axis=1))
    row = {'Dataset': '100-' + str(seq_len), **dict(df.cumsum(axis=1)[rank_cnt+4] / 100)}
    table.append(row)

pd.DataFrame(table).transpose().to_csv('tmp.csv')
new_df = pd.read_csv('tmp.csv')
csv_to_latex_pandas(new_df, output_file)
