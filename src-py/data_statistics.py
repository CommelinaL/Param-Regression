import os
from tqdm import tqdm
import numpy as np
from dataset import QuaternaryData
from sklearn.metrics import root_mean_squared_error, r2_score, median_absolute_error


class ErrorCalculator(QuaternaryData):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.pred_list, self.target_list = [], []
        for rel_path in tqdm(os.listdir(root_dir)):
            abs_path = os.path.join(root_dir, rel_path)
            for item_path in os.listdir(abs_path):
                item_abs_path = os.path.join(abs_path, item_path)
                if os.path.exists(os.path.join(item_abs_path, "point_data.txt")) == False:
                    continue
                point_best, param_best, target = self.read_item(item_abs_path)
                if np.any(np.isnan(param_best)):
                    continue
                target = np.array(target)
                pred_group = []
                error_group = []
                for i in range(5):
                    i_path = os.path.join(item_abs_path, str(i))
                    point_i, param_i, pred_i = self.read_item(i_path)
                    pred_group.append(np.array(pred_i))
                    error_group.append(np.sum(np.abs(pred_i - target)))
                error_min = min(error_group)
                idx_min = error_group.index(error_min)
                del pred_group[idx_min]
                self.pred_list += pred_group
                self.target_list += [target for _ in range(4)]
    
    def calc_error(self):
        rmse = root_mean_squared_error(self.target_list, self.pred_list)
        r2 = r2_score(self.target_list, self.pred_list)
        medae = median_absolute_error(self.target_list, self.pred_list)
        return {'RMSE': rmse, 'R2': r2, 'MedAE': medae}

if __name__ == "__main__":
    calc = ErrorCalculator("split_dataset/test")
    print(calc.calc_error())