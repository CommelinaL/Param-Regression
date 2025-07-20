import os
import shutil
from tqdm import tqdm
seq_len = 30
dst_crt_num = 2000
src_dir = r"D:\BSplineLearning\sequential_data\8000_sup\test_"+str(seq_len)
dst_dir = r"D:\BSplineLearning\sequential_data\test_"+str(seq_len)
#src_cost_dir = r"D:\BSplineLearning\pseudo_label\cost\PD1000-"+str(seq_len)+"-cor1"
# dst_cost_dir = r"D:\BSplineLearning\pseudo_label\cost\test_"+str(seq_len)
# correct_list = ["uniform", "chord", "centripetal", "universal", "fang"]

for item_file in tqdm(os.listdir(src_dir)):
    src_item_path = os.path.join(src_dir, item_file)
    try:
        src_id = int(item_file[:-4])
    except ValueError:
        continue
    dst_id = src_id + dst_crt_num
    dst_item_path = os.path.join(dst_dir, str(dst_id)+".txt")
    shutil.copyfile(src_item_path, dst_item_path)
    # for method in correct_list:
    #     src_cost_item_path = os.path.join(src_cost_dir, method, str(src_id)+".bin")
    #     dst_cost_item_path = os.path.join(dst_cost_dir, method, str(dst_id)+".bin")
    #     shutil.copyfile(src_cost_item_path, dst_cost_item_path)