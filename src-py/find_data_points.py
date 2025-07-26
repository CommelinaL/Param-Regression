import os
import numpy as np
from tqdm import tqdm
from utils import read_pts_from_file
import shutil
from dataset import PROJECT_ROOT

def find_data_point(item_path, root_dir):
    train_item_path = os.path.join(root_dir, "train", item_path, "point_data.txt")
    test_item_path = os.path.join(root_dir, "test", item_path, "point_data.txt")
    if os.path.exists(train_item_path):
        return train_item_path
    else:
        return test_item_path

def add_data_point(source_root_dir, target_root_dir):
    for item_path in tqdm(os.listdir(target_root_dir)):
        point_target_path = os.path.join(target_root_dir, item_path, "point_data.txt")
        point_source_path = find_data_point(item_path, source_root_dir)
        shutil.copyfile(point_source_path, point_target_path)

if __name__ == "__main__":
    source_root_dir = os.path.join(PROJECT_ROOT, "pseudo_label", "supplement")
    target_root_dir = os.path.join(PROJECT_ROOT, "pseudo_label", "supplement_label", "test")
    add_data_point(source_root_dir, target_root_dir)