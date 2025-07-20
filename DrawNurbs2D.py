import matplotlib
import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import CheckButtons
import os


# 从txt文件中读取控制点和节点
def read_control_points(file_path):
    return np.loadtxt(file_path)
def read_knots(file_path):
    return np.loadtxt(file_path)

# 读取多组控制点和节点
draw_id = [8,9]  #取值是[0-8]
sample_id = "9247"
crv_dir = "crv\\test_20"
curves = []
for i in draw_id:
    control_points = read_control_points(os.path.join(crv_dir, sample_id, str(i), "control_points.txt"))  # 假设文件名格式为 control_points_1.txt, control_points_2.txt, ...
    knots = read_knots(os.path.join(crv_dir, sample_id, str(i), "knots.txt"))  # 假设文件名格式为 knots_1.txt, knots_2.txt, ...
    curves.append((control_points, knots))

matrix_res = np.loadtxt(os.path.join(crv_dir, sample_id,'metric_res.txt'))
print(matrix_res.dtype)
print(matrix_res)

label_name = ["Uniform ", "Chord length ", "Centripetal ", "Universal ", "Foley ", "Fang ", "Xu ", "ZCM ", "Classifier ", "Regressor ", "Label ", "Cls Label "]
colors_9 = ['#FF0000',
          '#00FF00',
          '#0000FF',
          '#00FFFF',
          '#FF00FF',
          '#FFFF00',
          '#800080',
          '#FFA500',
          '#008080' ]

colors_5 = ['#0000FF',
          '#00FF00',
          '#FF0080',
          '#800000',
          '#8000FF',
          '#00FF00',
          '#800080',
          '#FFA500',
          '#FF0000' ]

colors_rgb = ['#FF0000',
          '#00FF00',
          '#0000FF',
          '#FF0000',
          '#00FF00',
          '#0000FF',]

colors_2 = ['#0000FF',
            '#FF0000',]

colors_3 = ['#0000FF',
            '#FF0000',
            '#008080',]

# 创建并绘制多条B样条曲线
degree = 3  # B样条的阶数
fig, ax = plt.subplots()
lines = []
index = 0
for control_points, knots in curves:
    bspline = BSpline(knots, control_points, degree)
    t = np.linspace(0, 1, 200000)
    curve = bspline(t)
    w=1
    ls='--'
    if draw_id[index]==9:
        w=1.2
        ls='-'

    ax.plot(curve[:, 0], curve[:, 1], label=label_name[draw_id[index]] + "{:.4f}".format(matrix_res[draw_id[index]]), color=colors_3[index], linewidth=w, linestyle=ls)

    index += 1


#从txt文件中读取额外的数据点
data_points = np.loadtxt(os.path.join(crv_dir, sample_id,'data_points.txt'))  # 假设文件名为 extra_points.txt
ax.scatter(data_points[:, 0], data_points[:, 1], color='black', label='DataPoints', s = 30)

font2 = {'family': 'Times New Roman',
'weight': 'normal',
'size': 30,
}

ax.legend(fontsize="large", loc='upper right')

# plt.legend(loc='upper right')




# 连接额外数据点的折线
ax.plot(data_points[:, 0], data_points[:, 1], color='black', linestyle='--', linewidth=0.7, label='Connected Line')
# ax.invert_xaxis()
# ax.invert_yaxis()
if len(draw_id)==0:
    ax.set_xlabel('x')
    ax.set_ylabel('y')
else:
    plt.xticks([])  # 隐藏 x 轴坐标
    plt.yticks([])  # 隐藏 y 轴坐标
    plt.axis('off')
plt.show()