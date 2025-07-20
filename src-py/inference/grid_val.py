import os
import numpy as np
import pandas as pd
import sys
sys.path.append(r"D:\BSplineLearning\Param-Regression\src-py")
from tqdm import tqdm
import pickle
import shutil
# import time
from utils import read_pts_from_file, write_vector_to_file
from feat import extract_features_batch
from utils import normalize

if __name__ == "__main__":
    seq_len = 15
    model_name = "PLS"
    param_name = "n_components"
    print(model_name, param_name)
    model_dir = r"saved_model\\param_grid_test\\" + model_name + "_on_" + param_name
    data_dir = r"D:\BSplineLearning\variable_length\split_dataset_"+str(seq_len)+r"\test"
    output_root_dir = r"D:\BSplineLearning\variable_length\split_dataset_"+str(seq_len) + "\\param_grid_test\\" + model_name + "_on_" + param_name

    model_list = os.listdir(model_dir)
    # model_name_list = ["PLS", "Linear Regression", "Decision Tree", "Quadratic Regression", "MLP with manual feature", 
    #                    "Bayesian Regression", "XGBoost", "Gradient Boosting", "SVR"]
    # model_list = [model_name + ".pickle" for model_name in model_name_list]
    # time_dict = [{'Model': model_filename[:-7], 'time': 0} for model_filename in model_list]
    # feat_dict = {'none': [], 'npc': list(range(8)), 'rcl': [8, 9, 10], 'cvl': 11, 'san': [12, 13], 'dsa': 14, 'daa': 15,
    #              'rcl_cvl': [8,9,10,11], 'cvl_dsa': [11, 14]}
    for i, model_filename in enumerate(model_list):
        model_path = os.path.join(model_dir, model_filename)
        print(model_filename)
        total = 0
        for item_file in tqdm(os.listdir(data_dir)):
            total += 1
            if total > 1000:
                break
            item_path = os.path.join(data_dir, item_file, "point_data.txt")
            file_id = item_file
            try:
                int(file_id)
            except ValueError:
                continue
            output_item_dir = os.path.join(output_root_dir, file_id)
            pts = np.array(read_pts_from_file(item_path)).reshape(-1, 2)
            pts_gps = [pts[i:i+4] for i in range(len(pts) - 3)]
            X = extract_features_batch(pts_gps)
            # X = np.delete(X, feat_dict[del_feat], 1)
            with open(model_path, 'rb') as m:
                model = pickle.load(m)
                Y = model.predict(X)
                Y = normalize(Y)
                write_vector_to_file(os.path.join(output_item_dir, model_filename[:-7]+".bin"), Y.flatten())
                shutil.copyfile(item_path, os.path.join(output_item_dir, "point_data.txt"))

