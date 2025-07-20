import os
import random
import shutil

def get_subdirectories(directory):
    """Get all subdirectories in the given directory."""
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def move_random_items(source_dir, dest_dir, num_items):
    """Randomly select and move items from source to destination directory."""
    items = get_subdirectories(source_dir)
    
    if len(items) < num_items:
        print(f"Warning: Only {len(items)} items available in the test set. Moving all of them.")
        num_items = len(items)
    
    selected_items = random.sample(items, num_items)
    
    for item in selected_items:
        source_path = os.path.join(source_dir, item)
        dest_path = os.path.join(dest_dir, item)
        
        try:
            shutil.move(source_path, dest_path)
            print(f"Moved {item} from test to train set.")
        except Exception as e:
            print(f"Error moving {item}: {str(e)}")
    
    return len(selected_items)

def main():
    train_dir = "train"
    test_dir = "test"
    num_items_to_move = 785

    print(f"Original train set size: {len(get_subdirectories(train_dir))}")
    print(f"Original test set size: {len(get_subdirectories(test_dir))}")

    moved_items = move_random_items(test_dir, train_dir, num_items_to_move)

    print(f"\nMoved {moved_items} items from test to train set.")
    print(f"Updated train set size: {len(get_subdirectories(train_dir))}")
    print(f"Updated test set size: {len(get_subdirectories(test_dir))}")

if __name__ == "__main__":
    main()