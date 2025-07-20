import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from csv_to_latex import csv_to_latex_pandas

feat_dict = {'none': [], 'npc': list(range(8)), 'rcl': [8, 9, 10], 'cvl': 11, 'san': [12, 13], 'daa': 14, 'dsa': 15}
path_dict = {del_feat: r"D:\BSplineLearning\Param-Regression\src-cpp\B-spline-curve-fitting\test_dataset_size_mlp_wo_{}_on_15.csv".format(del_feat)
             for del_feat in feat_dict}
cmp_feat_dict = {del_feat: 'wo {}'.format(del_feat) for del_feat in feat_dict}
cmp_feat_dict['none'] = 'all'
row_idx_dict = {'avg': 0, 'outlier_cnt':1, 'outlier_sum':2, 'avg_wo_outlier': 3, 'max':4}
row_chosen = 'max'
metrics = {del_feat: dict(pd.read_csv(path_dict[del_feat]).iloc[row_idx_dict[row_chosen]]) for del_feat in feat_dict}
for del_feat in feat_dict:
    metrics[del_feat].pop("label")
row_list = [{'selected feature': cmp_feat_dict[del_feat], **metrics[del_feat]} for del_feat in feat_dict]
df = pd.DataFrame(row_list)
csv_to_latex_pandas(df, "csv_analyzer\\datasize_feat_"+row_chosen+".tex")