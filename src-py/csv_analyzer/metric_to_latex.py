import pandas as pd
import os
from csv_to_latex import csv_to_latex_pandas

def convert_currency(var):
    try:
        var = float(var)
    except:
        var = var
    return var

dataset_size = 250000
del_feat = "npc"
model_name = "mlp"
input_root = r"D:\BSplineLearning\Param-Regression\src-cpp\B-spline-curve-fitting"
base_list = {seq_len: "test_{len}_{model}_wo_{feat}_{size}.csv".format(len=seq_len, model=model_name, feat=del_feat, size=dataset_size) for seq_len in range(10, 31, 5)}
input_file_list = {seq_len: os.path.join(input_root, base_list[seq_len]) for seq_len in range(10, 31, 5)}
output_file = 'csv_analyzer\\metric_wo_cusp_{} - {}.tex'.format(del_feat, dataset_size)
table = []
for seq_len, input_file in input_file_list.items():
    df = pd.read_csv(input_file)
    # df.drop(columns='Label_local', axis=1, inplace=True)
    # df.drop(columns='Regressor local', axis=1, inplace=True)
    # df.drop(list(range(1, 14)), axis=0, inplace=True)
    row = {'Dataset': '10000-' + str(seq_len), **dict(df.iloc[0])}
    table.append(row)

pd.DataFrame(table).transpose().to_csv('metric_wo_cusp.csv')
new_df = pd.read_csv('metric_wo_cusp.csv')
for col in new_df.columns:
    for i in range(len(new_df)):
        try:
            new_df[i][col] = new_df[i][col].astype(float)
        except:
            pass
csv_to_latex_pandas(new_df, output_file)
