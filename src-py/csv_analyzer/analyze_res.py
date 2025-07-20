import pandas as pd
import os
from dataset import QuaternaryData

heuristic_list = ["uniform", "chord", "centripetal", "universal", "foley", "fang", "modified_chord", "zcm"]
# Load the CSV files into DataFrames
df_10 = pd.read_csv(r"D:\BSplineLearning\Param-Regression\src-cpp\B-spline-curve-fitting\PD1000-10-cor1_enlarged_heuristic_test.csv")
print((df_10.transpose())[::-1].diff(axis=0)[::-1])
df_30 = pd.read_csv(r"D:\BSplineLearning\Param-Regression\src-cpp\B-spline-curve-fitting\PD1000-30-cor1_enlarged_heuristic_test.csv")
print((df_30.transpose())[::-1].diff(axis=0)[::-1])
df_4 = pd.read_csv(r"D:\BSplineLearning\Param-Regression\src-cpp\B-spline-curve-fitting\enlarged_batch_regressor_test.csv")
print((df_4.transpose())[::-1].diff(axis=0)[::-1])