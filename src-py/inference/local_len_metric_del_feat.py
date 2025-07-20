import os
import numpy as np
from tqdm import tqdm
import pickle
import sys
sys.path.append(r"D:\BSplineLearning\Param-Regression\src-py")
from utils import read_pts_from_file, write_vector_to_file, normalize
from feat import extract_features_batch

if __name__ == "__main__":
    del_feat = 'npc'
    print(del_feat)
    dataset_size = 250000
    model_name = "mlp"
    seq_len = 15
    data_dir = r"D:\BSplineLearning\variable_length\split_dataset_"+str(seq_len)+"\\test"
    output_root = "D:\\BSplineLearning\\variable_length\\" + model_name + "_wo_" + del_feat + "\\test_local_len_on" + str(seq_len) + "\\" + str(dataset_size)
    local_4_model_path = "saved_model\\dataset_size_test\\" + model_name + "_wo_" + del_feat + "\\" + str(dataset_size) + ".pickle"
    feat_dict = {'none':[], 'npc': list(range(8)), 'rcl': [8, 9, 10], 'cvl': 11, 'san': [12, 13], 'dsa': 14, 'daa': 15,
                 'cvl_dsa': [11, 14]}
    file_name_dict = {"mlp": "MLP with manual feature.pickle", "PLS": "PLS.pickle", "Linear": "Linear Regression.pickle",
                      "Ridge": "Ridge Regression.pickle", "SVR": "SVR.pickle"}
    for item_file in tqdm(os.listdir(data_dir)):
        item_path = os.path.join(data_dir, item_file, 'point_data.txt')
        output_item_dir = os.path.join(output_root, item_file)
        pts = np.array(read_pts_from_file(item_path)).reshape(-1, 2)
        pts_gps = [pts[i:i+4] for i in range(len(pts) - 3)]
        X = extract_features_batch(pts_gps)
        X = np.delete(X, feat_dict[del_feat], 1)
        with open(local_4_model_path, 'rb') as m:
            model = pickle.load(m)
            Y = model.predict(X)
            Y = normalize(Y)
            write_vector_to_file(os.path.join(output_item_dir, "4.bin"), Y.flatten())
    for local_len in [5, 6, 10, 15]:
        print(local_len)
        local_n_model_path = r"saved_model\seq_"+str(local_len)+"_wo_"+del_feat + "\\" + str(dataset_size) + "\\" + file_name_dict[model_name]
        feat_variable_dict = {'none': [], 'npc': list(range(2*local_len)), 'rcl': list(range(2*local_len, 3*local_len-1)),
                    'cvl': 3*local_len-1, 'san': list(range(3*local_len, 4*local_len-2)), 
                    'daa': list(range(4*local_len-2, 5*local_len-5)), 'dsa':list(range(5*local_len-5, 6*local_len-8))}
        for item_file in tqdm(os.listdir(data_dir)):
            item_path = os.path.join(data_dir, item_file, 'point_data.txt')
            output_item_dir = os.path.join(output_root, item_file)
            pts = np.array(read_pts_from_file(item_path)).reshape(-1, 2)
            pts_gps = [pts[i:i+local_len] for i in range(len(pts) - local_len + 1)]
            X = extract_features_batch(pts_gps, local_len=local_len)
            X = np.delete(X, feat_variable_dict[del_feat], 1)
            with open(local_n_model_path, 'rb') as m:
                model = pickle.load(m)
                Y = model.predict(X)
                Y = normalize(Y).squeeze()
                write_vector_to_file(os.path.join(output_item_dir, str(local_len)+".bin"), Y.flatten())