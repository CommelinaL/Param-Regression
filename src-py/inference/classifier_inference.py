import os
import numpy as np
import sys
sys.path.append(r"D:\BSplineLearning\Param-Regression\src-py")
import pandas as pd
from tqdm import tqdm
import pickle
import shutil
import time
from utils import read_pts_from_file, write_vector_to_file
from feat import extract_features_batch
from utils import normalize


if __name__ == "__main__":
    model_path = os.path.join('saved_model', "XGBClassifier200000-1-cor3.pickle")
    seq_len = 10
    del_feat = 'npc'
    data_dir = r"D:\BSplineLearning\sequential_data\test_" + str(seq_len)
    output_root_dir = r"D:\BSplineLearning\pseudo_label\seq_pred\class\test_" + str(seq_len)
    print(del_feat, data_dir)
    feat_dict = {'none':[], 'npc': list(range(8)), 'rcl': [8, 9, 10], 'cvl': 11, 'san': [12, 13], 'dsa': 14, 'daa': 15}
    # data_dir = r"D:\BSplineLearning\variable_length\split_dataset_15\test"
    # output_root_dir = r"D:\BSplineLearning\variable_length\split_dataset_15\class"
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)
    number = 0
    time_sum = 0
    for item_file in tqdm(os.listdir(data_dir)):
        item_path = os.path.join(data_dir, item_file)
        file_id = item_file[:-4]
        # item_path = os.path.join(data_dir, item_file, "point_data.txt")
        # file_id = item_file
        try:
            int(file_id)
        except ValueError:
            continue
        number += 1
        pts = np.array(read_pts_from_file(item_path)).reshape(-1, 2)
        pts_gps = [pts[i:i+4] for i in range(len(pts) - 3)]
        X = extract_features_batch(pts_gps, keep_shape=True)
        with open(model_path, 'rb') as m:
            model = pickle.load(m)
            X_input = np.delete(X, feat_dict[del_feat], 1)
            start_time = time.time()
            Y = model.predict(X_input)
            test_time = time.time() - start_time
            time_sum += test_time
            f=open(os.path.join(output_root_dir, file_id + '-r.txt'),"w")
            for i in list(Y.astype(int)):
                f.write(str(i)+' ')
            f.close()
    print(time_sum / number * 1e3)