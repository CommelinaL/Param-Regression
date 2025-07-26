import os
import numpy as np
import pickle
import sys
from dataset import PROJECT_ROOT
sys.path.append(os.path.join(PROJECT_ROOT, "Param-Regression", "src-py"))
from utils import read_pts_from_file, write_vector_to_file
from feat import extract_features_batch
from utils import normalize
if __name__ == "__main__":
    del_feat = 'npc'
    dataset_size = 250000
    regress_model_path = os.path.join(PROJECT_ROOT, "Param-Regression", "src-py", "saved_model", f"data_{dataset_size}_wo_{del_feat}", "MLP with manual feature.pickle")
    class_model_path = os.path.join(PROJECT_ROOT, "Param-Regression", "src-py", 'saved_model', "XGBClassifier200000-1-cor3.pickle")
    base = "clash"
    item_path = os.path.join(PROJECT_ROOT, "Param-Regression", "meaningful_examples", f"{base}.txt")
    regress_dir = os.path.join(PROJECT_ROOT, "Param-Regression", "meaningful_examples", "regress")
    class_dir = os.path.join(PROJECT_ROOT, "Param-Regression", "meaningful_examples", "class")
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    feat_dict = {'none':[], 'npc': list(range(8)), 'rcl': [8, 9, 10], 'cvl': 11, 'san': [12, 13], 'dsa': 14, 'daa': 15}
    pts = np.array(read_pts_from_file(item_path)).reshape(-1, 2)
    pts_gps = [pts[i:i+4] for i in range(len(pts) - 3)]
    X = extract_features_batch(pts_gps)
    X = np.delete(X, feat_dict[del_feat], 1)
    with open(regress_model_path, 'rb') as m:
        model = pickle.load(m)
        Y = model.predict(X)
        Y = normalize(Y)
        write_vector_to_file(os.path.join(regress_dir, base+".bin"), Y.flatten())
    with open(class_model_path, 'rb') as m:
        model = pickle.load(m)
        Y = model.predict(X)
        f=open(os.path.join(class_dir, base + '-r.txt'),"w")
        for i in list(Y.astype(int)):
            f.write(str(i)+' ')
        f.close()