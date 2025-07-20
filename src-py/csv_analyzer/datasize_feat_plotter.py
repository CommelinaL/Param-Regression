import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

feat_dict = {'none': [], 'npc': list(range(8)), 'rcl': [8, 9, 10], 'cvl': 11, 'san': [12, 13],
            'dsa': 14, 'daa': 15, 
            # 'cvl_dsa': [11, 14]
            }
model_name = "mlp"
path_dict = {del_feat: r"D:\BSplineLearning\Param-Regression\src-cpp\B-spline-curve-fitting\test_dataset_size_{}_wo_{}_on_15.csv".format(model_name, del_feat)
             for del_feat in feat_dict}
x = np.arange(10000, 250001, 10000)
marker_dict = {'none': 'o-r', 'npc': 's-g', 'rcl': 'v-b', 'cvl': '^-c', 'san': '1-m', 'daa': '2-y', 'dsa': '3-k', 'cvl_dsa':'4:b'}
label_dict = {'none':'all', 'npc': 'no npc', 'rcl': 'no rcl', 'cvl': 'no cvl', 'san': 'no san', 'daa': 'no dsa', 'dsa': 'no daa', 'cvl_dsa': 'no cvl & dsa'}
# Create a figure and axis
plt.figure(figsize=(10, 6))
# plt.ylim(1.15, 1.22) # PLS
# plt.ylim(0.9, 1)
for del_feat in feat_dict:
    metric = list(pd.read_csv(path_dict[del_feat]).iloc[0]/1000.0)[:25]
    plt.plot(x, metric, marker_dict[del_feat], label=label_dict[del_feat])
plt.legend()
plt.xlabel('Size of training dataset')
# plt.ylabel('Average metric value of interpolation curves')
plt.ylabel('Top 1 ratio')
plt.tight_layout()
plt.show()