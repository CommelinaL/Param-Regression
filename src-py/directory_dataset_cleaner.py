import os
import shutil
from collections import defaultdict

def get_subdirectories(directory):
    """Get all subdirectories in the given directory."""
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def remove_duplicates(train_dir, test_dir):
    """Remove duplicate subdirectories from train and test directories."""
    train_subdirs = set(get_subdirectories(train_dir))
    test_subdirs = set(get_subdirectories(test_dir))
    
    duplicates = train_subdirs.intersection(test_subdirs)
    
    print(f"Found {len(duplicates)} duplicates.")
    
    # Remove duplicates from test set (you can change this to remove from train set if preferred)
    for duplicate in duplicates:
        test_subdir_path = os.path.join(test_dir, duplicate)
        shutil.rmtree(test_subdir_path)
        print(f"Removed duplicate subdirectory from test set: {duplicate}")

def main():
    train_dir = "train"
    test_dir = "test"
    
    print(f"Original train set size: {len(get_subdirectories(train_dir))}")
    print(f"Original test set size: {len(get_subdirectories(test_dir))}")
    
    remove_duplicates(train_dir, test_dir)
    
    print(f"Cleaned train set size: {len(get_subdirectories(train_dir))}")
    print(f"Cleaned test set size: {len(get_subdirectories(test_dir))}")
    
    print("Duplicate removal complete.")

if __name__ == "__main__":
    main()