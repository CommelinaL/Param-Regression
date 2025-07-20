import pandas as pd
import os
from dataset import QuaternaryData

heuristic_list = ["uniform", "chord", "centripetal", "universal", "foley", "fang", "modified_chord", "zcm"]
# Load the CSV files into DataFrames
# df = pd.read_csv(r"D:\BSplineLearning\Param-Regression\src-cpp\B-spline-curve-fitting\batch_regressor_test.csv")
# df=df.rename(columns={'Label':"classical_best"})
# df = df.transpose()
# df[[0,1]]=df[[1,0]]
# df.columns = ['avg_cost', 'surpass_label_rate', '2nd_cnt', '3rd_cnt'] + ["{}th_cnt".format(i) for i in range(4, 28)]
# df = df[['avg_cost', 'surpass_label_rate', '2nd_cnt', '3rd_cnt']]
# df['surpass_label_rate'] = df["surpass_label_rate"] / 1000
# print("\n")
# print("training dataset size: 140000")
# print(df)
df = None
for i in range(10, 31, 5):
    if df is None:
        df = pd.read_csv("PD1000-{}-cor1_time.csv".format(i))
        df = df.drop(df.columns[[0]], axis=1)
        df=df.rename(columns={'time':str(i)})
    else:
        tmp_df = pd.read_csv("PD1000-{}-cor1_time.csv".format(i))
        tmp_df = tmp_df.drop(tmp_df.columns[[0]], axis=1)
        tmp_df=tmp_df.rename(columns={'time': str(i)})
        df = pd.merge(df,tmp_df,how='inner',on='Model')
cls_df = pd.read_csv('classifier_time.csv')
new_row = pd.DataFrame({'Model': ['Classifier'], '10': cls_df['time'][0], '15': cls_df['time'][1],
                        '20': cls_df['time'][2], '25': cls_df['time'][3], '30': cls_df['time'][4]})
df = pd.concat([df, new_row])
print(df)
df.to_csv('time cost.csv')
