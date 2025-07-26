import os
import random
import shutil
import math
from tqdm import tqdm
from dataset import PROJECT_ROOT

def process_files(source_dir, target_dir, test_ratio=0.2):
    train_dir = os.path.join(target_dir, "train")
    test_dir = os.path.join(target_dir, "test")
    # Ensure output directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Walk through the source directory
    for rel_path in tqdm(os.listdir(source_dir)):
        if rel_path == "record":
            continue
        # Get the absolute path
        abs_path = os.path.join(source_dir, rel_path)
        # Create corresponding subdirectories in train and test directories
        train_subdir = os.path.join(train_dir, rel_path)
        test_subdir = os.path.join(test_dir, rel_path)
        os.makedirs(train_subdir, exist_ok=True)
        os.makedirs(test_subdir, exist_ok=True)

        # Calculate the number of files to move to the test set
        items = [f for f in os.listdir(abs_path)]
        num_test_files = math.ceil(len(items) * test_ratio)

        # Randomly select files for the test set
        test_files = random.sample(items, num_test_files)

        # Move files to appropriate directories
        for file in items:
            source_file = os.path.join(abs_path, file)
            if os.path.exists(os.path.join(source_file, "point_data.txt")) == False:
                continue
            if file in test_files:
                dest_file = os.path.join(test_subdir, file)
            else:
                dest_file = os.path.join(train_subdir, file)
            if os.path.exists(dest_file) == False:
                shutil.copytree(source_file, dest_file)

    print("File processing completed.")

def process_flat_files(source_dir, target_dir, test_ratio=0.2):
    train_dir = os.path.join(target_dir, "train")
    test_dir = os.path.join(target_dir, "test")
    # Ensure output directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    # Calculate the number of files to move to the test set
    items = [f for f in os.listdir(source_dir)]
    num_test_files = math.ceil(len(items) / 6)
    # Randomly select files for the test set
    test_files = random.sample(items, num_test_files)
    # Move files to appropriate directories
    for file in tqdm(items):
        source_file = os.path.join(source_dir, file)
        if os.path.exists(os.path.join(source_file, "point_data.txt")) == False:
            continue
        if file in test_files:
            dest_file = os.path.join(test_dir, file)
        else:
            dest_file = os.path.join(train_dir, file)
        os.makedirs(dest_file, exist_ok=True)
        shutil.copyfile(os.path.join(source_file, "point_data.txt"), os.path.join(dest_file, "point_data.txt"))
        # if os.path.exists(os.path.join(dest_file, "heuristic")) == False:
        #     shutil.copytree(os.path.join(source_file, "heuristic"), os.path.join(dest_file, "heuristic"))
        if os.path.exists(os.path.join(dest_file, "best.bin")) == False:
            shutil.copyfile(os.path.join(source_file, "best.bin"), os.path.join(dest_file, "best.bin"))
    

if __name__ == "__main__":
    source_dir = os.path.join(PROJECT_ROOT, "variable_length", "Label_15")
    target_dir = os.path.join(PROJECT_ROOT, "variable_length", "split_dataset_15")
    process_flat_files(source_dir, target_dir)
    # from dataset import QuaternaryData
    # data = QuaternaryData(os.path.join(target_dir, 'test'), "test")
    # print(data.point_list[0])
    # print(len(data.param_list))