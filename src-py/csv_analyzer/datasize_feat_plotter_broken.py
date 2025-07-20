import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

feat_dict = {'none': [], 'npc': list(range(8)), 'rcl': [8, 9, 10], 'cvl': 11, 'san': [12, 13],
            'dsa': 14, 'daa': 15, 
            # 'cvl_dsa': [11, 14]
            }
model_name = "mlp"
path_dict = {del_feat: r"D:\BSplineLearning\Param-Regression\src-cpp\B-spline-curve-fitting\test_dataset_size_{}_wo_{}_on_15.csv".format(model_name, del_feat)
             for del_feat in feat_dict}
x = np.arange(10000, 250001, 10000)
marker_dict = {'none': 'o-r', 'npc': 's-g', 'rcl': 'v-b', 'cvl': '^-c', 'san': '1-m', 'daa': '2-y', 'dsa': '3-k', 'cvl_dsa':'4:b'}
label_dict = {'none':'all', 'npc': 'no npc', 'rcl': 'no rcl', 'cvl': 'no cvl', 'san': 'no san', 'daa': 'no daa', 'dsa': 'no dsa', 'cvl_dsa': 'no cvl & dsa'}


# Create the figure and subplots
f, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

# Plot the data on both axes
for del_feat in feat_dict:
    # For demonstration, we'll use random data instead of reading from files
    metric = list(pd.read_csv(path_dict[del_feat]).iloc[1]/1000.0)[:25]
    ax.plot(x, metric, marker_dict[del_feat], label=label_dict[del_feat])
    ax2.plot(x, metric, marker_dict[del_feat], label=label_dict[del_feat])

# Set y-axis limits
ax.set_ylim(.9, 1.)  # outliers only
ax2.set_ylim(.2, .6)  # most of the data

# Hide the spines between ax and ax2
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

# Add diagonal lines
d = .015  # how big to make the diagonal lines in axes coordinates
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

ax.tick_params(labelsize="xx-large")
ax2.tick_params(labelsize="xx-large")
# Add legend and xlabel
plt.legend(fontsize="xx-large")
plt.xlabel('Size of training dataset', fontsize="xx-large")

# Add ylabel in the middle of the y-axis
f.text(0.02, 0.5, 'Top 3 ratio', va='center', rotation='vertical', fontsize="xx-large")

# Adjust the layout
plt.tight_layout()

# Adjust the subplot spacing to make room for the ylabel
plt.subplots_adjust(left=0.08)

plt.show()

# Save the figure (optional)
# plt.savefig('polyline_diagram.png', dpi=300, bbox_inches='tight')

print("The plot has been generated and displayed.")