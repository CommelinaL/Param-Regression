from xgboost import XGBClassifier
import numpy as np
import pickle
import os
from sklearn.metrics import roc_auc_score
from dataset import ClassData
from feat import extract_features_batch

if __name__ == '__main__':
    # Data preprocessing
    root_dir = r"D:\BSplineLearning\HybridParameterization\data\TD4-80000-3-cor1"

    train_dataset = ClassData(os.path.join(root_dir, "traincrv"), "train")
    test_dataset = ClassData(os.path.join(root_dir, "testcrv"), "test")

    X_train = extract_features_batch(train_dataset.point_list)
    X_test = extract_features_batch(test_dataset.point_list)
    y_train = np.array(train_dataset.label_list)
    y_test = np.array(test_dataset.label_list)
    # The parameters of the model are from Tianyu Song
    model = XGBClassifier(n_estimators=450,
                               learning_rate=0.1,  # 步长
                               max_depth=10,  # 树的最大深度
                               min_child_weight=1,  # 决定最小叶子节点样本权重和
                               silent=1,  # 输出运行信息
                               subsample=0.8,  # 每个决策树所用的子样本占总样本的比例（作用于样本）
                               colsample_bytree=0.8,  # 建立树时对特征随机采样的比例（作用于特征）典型值：0.5-1
                               objective='multi:softmax',  # 多分类
                               #num_class=3,
                               nthread=20,
                               eval_metric="mlogloss",
                               early_stopping_rounds=10,
                               seed=27)
    save_path = 'saved_model'
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    model_name = "XGBClassifier"
    file_path = os.path.join(save_path, model_name +'.pickle')
    print("Training", model_name)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)