import os
import numpy as np
from tqdm import tqdm
from utils import read_vector_from_file, read_pts_from_file

PROJECT_ROOT = "D:\\BSplineLearning"


class QuaternaryData:
    def __init__(self, root_dir, mode = "train"):
        self.root_dir = root_dir
        self.mode = mode
        self.point_list, self.param_list, self.target_list, self.path_list = [], [], [], []
        for rel_path in tqdm(os.listdir(root_dir)):
            abs_path = os.path.join(root_dir, rel_path)
            for item_path in os.listdir(abs_path):
                item_abs_path = os.path.join(abs_path, item_path)
                if os.path.exists(os.path.join(item_abs_path, "point_data.txt")) == False:
                    continue
                point, param, target = self.read_item(item_abs_path)
                if np.any(np.isnan(param)):
                    continue
                self.point_list.append(point)
                self.param_list.append(param)
                self.target_list.append(target)
                self.path_list.append([rel_path, item_path])
    
    def read_item(self, item_path):
        point_path = os.path.join(item_path, "point_data.txt")
        param_path = os.path.join(item_path, "param.bin")
        points = read_pts_from_file(point_path)
        param = read_vector_from_file(param_path)
        target = [param[i+1]-param[i] for i in range(len(param) - 1)]
        return points, param, target
    
    def __len__(self):
        return len(self.param_list)

    def __getitem__(self, item):
        return self.point_list[item], self.param_list[item], self.target_list[item]

class FlatData(QuaternaryData):
    def __init__(self, root_dir, mode = "train"):
        self.root_dir = root_dir
        self.mode = mode
        self.point_list, self.param_list, self.target_list, self.path_list = [], [], [], []
        for item_path in tqdm(os.listdir(root_dir)):
            item_abs_path = os.path.join(root_dir, item_path)
            if os.path.exists(os.path.join(item_abs_path, "point_data.txt")) == False:
                continue
            point, param, target = self.read_item(item_abs_path)
            if np.any(np.isnan(param)):
                continue
            self.point_list.append(point)
            self.param_list.append(param)
            self.target_list.append(target)
            self.path_list.append(item_path)
    
    def read_item(self, item_path):
        point_path = os.path.join(item_path, "point_data.txt")
        param_path = os.path.join(item_path, "heuristic", "best.bin")
        if not os.path.exists(param_path):
            param_path = os.path.join(item_path, "best.bin")
        points = read_pts_from_file(point_path)
        param = read_vector_from_file(param_path)
        target = [param[i+1]-param[i] for i in range(len(param) - 1)]
        return points, param, target

class ClassData:
    def __init__(self, root_dir, mode = "train"):
        self.root_dir = root_dir
        self.mode = mode
        self.point_list, self.label_list = [], []
        for item_path in tqdm(os.listdir(root_dir)):
            item_abs_path = os.path.join(root_dir, item_path)
            self.point_list.append(read_pts_from_file(item_abs_path))
            self.label_list.append(int(item_path[0]) - 3)
    
    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, item):
        return self.point_list[item], self.label_list[item]