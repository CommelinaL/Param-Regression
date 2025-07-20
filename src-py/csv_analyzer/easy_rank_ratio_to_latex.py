import pandas as pd
import os
from csv_to_latex import csv_to_latex_pandas

dataset_size = 250000
del_feat = "npc"
model_name = "mlp"
input_root = r"D:\BSplineLearning\Param-Regression\src-cpp\B-spline-curve-fitting\easy"
base_list = {seq_len: "Label_test_{len}_{model}_wo_{feat}_{size}.csv".format(len=seq_len, model=model_name, feat=del_feat, size=dataset_size) for seq_len in range(10, 31, 5)}
input_file_list = {seq_len: os.path.join(input_root, base_list[seq_len]) for seq_len in range(10, 31, 5)}
rank_cnt = 3
output_file = 'csv_analyzer\\easy_rank_{cnt}_ratio_wo_{feat} - {size}.tex'.format(cnt=rank_cnt, feat=del_feat, size=dataset_size)
table = []
for seq_len, input_file in input_file_list.items():
    df = pd.read_csv(input_file)
    # df.drop(columns='Label_local', axis=1, inplace=True)
    # df.drop(columns='Regressor local', axis=1, inplace=True)
    df.drop([0], axis=0, inplace=True)
    df = df.transpose()
    row = {'Dataset': '10000-' + str(seq_len), **dict(df.cumsum(axis=1)[rank_cnt] / 10000)}
    table.append(row)

pd.DataFrame(table).to_csv('tmp.csv')
new_df = pd.read_csv('tmp.csv')
csv_to_latex_pandas(new_df, output_file)
