import os
import numpy as np
import pandas as pd
import sys
from dataset import PROJECT_ROOT
sys.path.append(os.path.join(PROJECT_ROOT, "Param-Regression", "src-py"))
from tqdm import tqdm
import pickle
import shutil
import joblib
# import time
from utils import read_pts_from_file, write_vector_to_file
from feat import extract_features_batch
from utils import normalize

if __name__ == "__main__":
    dataset_size = 250000
    seq_len = 15
    del_feat = "npc"
    model_dir = os.path.join(PROJECT_ROOT, "Param-Regression", "src-py", "saved_model", f"data_{dataset_size}_wo_{del_feat}")
    data_dir = os.path.join(PROJECT_ROOT, "variable_length", "split_dataset_" + str(seq_len), "test")
    output_root_dir = os.path.join(PROJECT_ROOT, "variable_length", "split_dataset_" + str(seq_len), "model_test", str(dataset_size) + "_wo_" + del_feat)

    model_list = os.listdir(model_dir)
    # model_name_list = ["PLS", "Linear Regression", "Decision Tree", "Quadratic Regression", "MLP with manual feature", 
    #                    "Bayesian Regression", "XGBoost", "Gradient Boosting", "SVR"]
    # model_list = [model_name + ".pickle" for model_name in model_name_list]
    feat_dict = {'none': [], 'npc': list(range(8)), 'rcl': [8, 9, 10], 'cvl': 11, 'san': [12, 13], 'dsa': 14, 'daa': 15}
    # time_dict = [{'Model': model_filename[:-7], 'time': 0} for model_filename in model_list]
    for i, model_filename in enumerate(model_list):
        print("Testing", model_filename[:-7])
        if(model_filename[:-7]=="Random Forest"):
            continue
        if(model_filename[:-7]=="MLP with raw data"):
            continue
        model_path = os.path.join(model_dir, model_filename)
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
            X = np.delete(X, feat_dict[del_feat], 1)
            with open(model_path, 'rb') as m:
                model = pickle.load(m)
                Y = model.predict(X)
                Y = normalize(Y)
                write_vector_to_file(os.path.join(output_item_dir, model_filename[:-7]+".bin"), Y.flatten())
                shutil.copyfile(item_path, os.path.join(output_item_dir, "point_data.txt"))

