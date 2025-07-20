import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from feat import extract_features_batch
from utils import read_pts_from_file
import time
from custom_models import MyPolyRegressor, MyPCR, MultiLinearGAM

if __name__ == "__main__":
    raw_dir = r"D:\BSplineLearning\HybridParameterization\data\PointData"
    # Read raw data not in the training set
    raw_pt_list = []
    print("Reading raw data")
    i = 0
    for raw_filename in os.listdir(raw_dir):
        raw_pt_list.append(read_pts_from_file(os.path.join(raw_dir, raw_filename)))
        i += 1
        if i > 27:
            break
    # Extract feature
    raw_pt_list = np.array(raw_pt_list)
    feat = extract_features_batch(raw_pt_list)
    # Prediction
    model_dir = os.path.join('saved_model', 'GA')
    model_list = os.listdir(model_dir)
    results = [{"Model": model_filename[:-7]} for model_filename in model_list]
    for pt_num in range(10, 31, 5):
        print("Point number is", pt_num)
        for i, model_filename in enumerate(model_list):
            model_filename = model_list[i]
            print("Testing", model_filename[:-7])
            model_path = os.path.join(model_dir, model_filename)
            with open(model_path, 'rb') as m:
                model = pickle.load(m)
                if 'raw' not in model_filename:
                    x_in = feat[:pt_num - 3]
                else:
                    x_in = feat[:pt_num - 3, :8]
                start_test = time.time()
                y = model.predict(x_in)
                test_time = time.time() - start_test
                results[i][str(pt_num)] = test_time * 1e3
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('30')
    print(results_df)
    results_df.to_csv('test_time.csv')
