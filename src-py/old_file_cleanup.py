import os
import datetime
import argparse

def delete_old_files(directory, cutoff_date):
    deleted_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            modified_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
            
            if modified_time < cutoff_date:
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
    
    print(f"Total files deleted: {deleted_count}")

def main():
    parser = argparse.ArgumentParser(description="Delete files older than a specified date.")
    parser.add_argument("directory", help="Directory to clean up")
    parser.add_argument("cutoff_date", help="Cutoff date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    try:
        cutoff_date = datetime.datetime.strptime(args.cutoff_date, "%Y-%m-%d")
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD.")
        return
    
    if not os.path.isdir(args.directory):
        print("Invalid directory path.")
        return
    
    delete_old_files(args.directory, cutoff_date)

if __name__ == "__main__":
    # python old_file_cleanup.py /path/to/directory 2023-01-01
    main()