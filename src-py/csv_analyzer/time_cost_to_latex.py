import pandas as pd
from csv_to_latex import csv_to_latex_pandas

input_file = "total_time_cost.csv"
output_file = "csv_analyzer\\total_time_cost.tex"
df = pd.read_csv(input_file)
csv_to_latex_pandas(df, output_file)

input_file = "regress_time_cost.csv"
output_file = "csv_analyzer\\regress_time_cost.tex"
df = pd.read_csv(input_file)
csv_to_latex_pandas(df, output_file)