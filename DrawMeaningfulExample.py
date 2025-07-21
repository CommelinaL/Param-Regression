import matplotlib
import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import os


# 从txt文件中读取控制点和节点
def read_control_points(file_path):
    return np.loadtxt(file_path)
def read_knots(file_path):
    return np.loadtxt(file_path)

# 读取多组控制点和节点
#draw_id = [4, 5, 6, 7, 8]  #取值是[0-8]
draw_id = [8,9]
sample_id = "clash"
crv_dir = os.path.join(r"D:\BSplineLearning\ParamNet\crv\PD1000-20-cor1", sample_id)
crv_dir = "crv\\meaningful_examples\\" + sample_id
curves = []
for i in draw_id:
    control_points = read_control_points(os.path.join(crv_dir, str(i), "control_points.txt")) 
    knots = read_knots(os.path.join(crv_dir, str(i), "knots.txt")) 
    curves.append((control_points, knots))

matrix_res = np.loadtxt(os.path.join(crv_dir,'metric_res.txt'))
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

# 创建并绘制多条B样条曲线
degree = 3  # B样条的阶数
fig, ax = plt.subplots()
if len(draw_id)==0:
    # Load and display the background image
    try:
        background_img = plt.imread(os.path.join("meaningful_examples", sample_id + ".png"))
        # You'll need to adjust these extent values based on your data range
        # Get the approximate range of your curve data first
        ax.imshow(background_img, extent=[0.135,0.93,0.05,0.96], aspect='auto', alpha=0.7, zorder=0) # clash
        # ax.imshow(background_img, extent=[-6,102,-15,35], aspect='auto', alpha=0.7, zorder=0) # rolling door slat
    except FileNotFoundError:
        print("Background image not found. Continuing without background image.")

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

    ax.plot(curve[:, 0], -curve[:, 1], label=label_name[draw_id[index]] + "{:.4f}".format(matrix_res[draw_id[index]]), color=colors_2[index], linewidth=w, linestyle=ls)

    index += 1


#从txt文件中读取额外的数据点
data_points = np.loadtxt(os.path.join(crv_dir,'data_points.txt'))  # 假设文件名为 extra_points.txt
ax.scatter(data_points[:, 0], -data_points[:, 1], color='black', label='DataPoints', s = 30)
# ax.scatter(data_points[:, 0], data_points[:, 1], color='black', label='DataPoints', s = 15)

font2 = {'family': 'Times New Roman',
'weight': 'normal',
'size': 30,
}

ax.legend(fontsize="large", loc="upper right")
# ax.legend()

#ax.legend(prop=font2)

# plt.legend(loc='upper left')




# 连接额外数据点的折线
ax.plot(data_points[:, 0], -data_points[:, 1], color='black', linestyle='--', linewidth=0.7, label='Connected Line')

if sample_id == "rolling_door_slat":
    ax.invert_yaxis()

if len(draw_id)==0:
    ax.set_xlabel('x')
    ax.set_ylabel('y')
else:
    plt.xticks([])  # 隐藏 x 轴坐标
    plt.yticks([])  # 隐藏 y 轴坐标
    plt.axis('off')

plt.show()