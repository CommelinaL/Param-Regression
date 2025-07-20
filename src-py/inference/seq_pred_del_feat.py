import os
import numpy as np
import pandas as pd
import sys
sys.path.append(r"D:\BSplineLearning\Param-Regression\src-py")
from tqdm import tqdm
import pickle
import shutil
import time
from utils import read_pts_from_file, write_vector_to_file
from feat import extract_features_batch
from utils import normalize

if __name__ == "__main__":
    del_feat = 'npc'
    dataset_size = 250000
    seq_len = 15
    print(dataset_size, seq_len)
    model_dir = r"D:\BSplineLearning\Param-Regression\src-py\saved_model\data_"+str(dataset_size)+"_wo_"+del_feat
    data_dir = r"D:\BSplineLearning\sequential_data\test_"+str(seq_len)
    output_root_dir = r"D:\BSplineLearning\pseudo_label\seq_pred\data_"+str(dataset_size)+"_wo_"+del_feat+r"\test_"+str(seq_len)
    number = 0
    model_list = os.listdir(model_dir)
    # model_name_list = ["PLS", "Linear Regression", "Decision Tree", "Quadratic Regression", "MLP with manual feature", 
    #                    "Bayesian Regression", "XGBoost", "Gradient Boosting", "SVR"]
    # model_list = [model_name + ".pickle" for model_name in model_name_list]
    # model_list = ['MLP with manual feature.pickle']
    feat_dict = {'none':[], 'npc': list(range(8)), 'rcl': [8, 9, 10], 'cvl': 11, 'san': [12, 13], 'dsa': 14, 'daa': 15}
    time_dict = [{'Model': model_filename[:-7], 'time': 0} for model_filename in model_list]
    for item_file in tqdm(os.listdir(data_dir)):
        item_path = os.path.join(data_dir, item_file)
        file_id = item_file[:-4]
        try:
            int(file_id)
        except ValueError:
            continue
        if number > 1000:
            break
        number += 1
        output_item_dir = os.path.join(output_root_dir, file_id)
        pts = np.array(read_pts_from_file(item_path)).reshape(-1, 2)
        pts_gps = [pts[i:i+4] for i in range(len(pts) - 3)]
        X = extract_features_batch(pts_gps)
        X = np.delete(X, feat_dict[del_feat], 1)
        for i, model_filename in enumerate(model_list):
            model_path = os.path.join(model_dir, model_filename)
            with open(model_path, 'rb') as m:
                model = pickle.load(m)
                X_input = X if "raw" not in model_filename else X[:,:8]
                start_time = time.time()
                Y = model.predict(X_input)
                test_time = time.time() - start_time
                time_dict[i]['time'] += test_time
                Y = normalize(Y)
                write_vector_to_file(os.path.join(output_item_dir, model_filename[:-7]+".bin"), Y.flatten())
                shutil.copyfile(item_path, os.path.join(output_item_dir, "point_data.txt"))
    for row in time_dict:
        row['time'] /= number
        row['time'] *= 1e3
    results_df = pd.DataFrame(time_dict)
    results_df = results_df.sort_values('time')
    print(results_df)
    results_df.to_csv(os.path.basename(data_dir)+'_wo_'+del_feat+'_time.csv')


