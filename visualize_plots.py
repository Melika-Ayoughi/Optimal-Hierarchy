import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from matplotlib.cm import get_cmap

# Example data
height = [9, 9, 6, 5, 4, 6]#, 1, 15]  # Heights (x-axis)
# height_sorted_index = [4, 3, 2, 5, 1, 0]
height_sorted_index = [4, 3, 2, 1]
height_sorted_index_additional = [5, 0]

variance_depth_512 = [2.0, 0.0038909912109375, 0.2456463222710503, 0.24700927734375, 0.11279224777470548, 1.4777282121781679]#, 0.0, 0.0]
r = [2, 3, 4, 5]
max_degree = [9, 3, 4, 5, 6, 86]#, 512, 2]

distortion_method1_256 = [1.736, 0.880, 1.439, 2.129, 2.472, 3.444]#, 0.936, 0.837]  # Distortion for Poincare
distortion_method1_512 = [1.439, 0.459, 1.085, 1.471, 1.770, 2.791]#, 0.971, 0.883]  # Distortion for Poincare
distortion_method1_1024 = [0.988, 0.229, 0.752, 1.092, 1.385, 2.206]#, 0.965, 0.925]  # Distortion for Poincare

distortion_method2_256 = [0.717, 0.816, 0.742, 0.695, 0.657, 0.595]  # Distortion for Entailment
distortion_method2_512 = [0.863, 0.914, 0.878, 0.855, 0.837, 0.802]#, None, 0.986]  # Distortion for Entailment
distortion_method2_1024 = [0.932, 0.960, 0.940, 0.928, 0.919, 0.903]  # Distortion for Entailment

distortion_method3_256 = [0.207, 0.220, 0.124, 0.102, 0.115, 0.108]#, 0.015, 0.000]
distortion_method3_512 = [0.249, 0.259, 0.156, 0.133, 0.120, 0.140]#, 0.016, 0.000]  # Distortion for Construction1
distortion_method3_1024 = [0.298, 0.300, 0.160, 0.137, 0.156, 0.178]#, 0.009, 0.000]

distortion_method4_256 = [0.161, 0.176, 0.102, 0.079, 0.078, None]#, None, 0.962]
#todo: had to change None to 0 for bar plot:
distortion_method4_512 = [0.186, 0.207, 0.127, 0.103, 0.080, 0]#, None, 0.979]  # Distortion for Construction2
distortion_method4_1024 = [0.211, 0.240, 0.130, 0.105, 0.103, None]#, None, 0.988]


#dim=20, n=512 worst-case
wcdistortion_method1_512 = [69.530, 164.777, 183.974, 390.397, 336.711, 3607.953]  # Distortion for Poincare
wcdistortion_method2_512 = [224.731, 434.177, 316.338, 323.967, 383.626, 731.914]  # Distortion for Entailment
wcdistortion_method3_512 = [1.542, 1.539, 1.252, 1.201, 1.201, 1.329]  # Distortion for Construction1
wcdistortion_method4_512 = [1.257, 1.297, 1.155, 1.121, 1.092, None]  # Distortion for Construction2


#dim=20, n=512, map
map_method1_512 = [0.171, 0.866, 0.770, 0.671, 0.534, 0.020]  # MAP for Poincare
map_method2_512 = [0.304, 0.439, 0.217, 0.183, 0.169, 0.231]  # MAP for Entailment
map_method3_512 = [1, 1, 1, 1, 1, 1]  # MAP for Construction1
map_method4_512 = [1, 1, 1, 1, 1, None]  # MAP for Construction2

methods_512 = [
    {"name": "Poincaré", "distortion": distortion_method1_512, "wcd": wcdistortion_method1_512, "map": map_method1_512},
    {"name": "Entailment", "distortion": distortion_method2_512, "wcd": wcdistortion_method2_512, "map": map_method2_512},
    {"name": "Precomputed", "distortion": distortion_method3_512, "wcd": wcdistortion_method3_512, "map": map_method3_512},
    {"name": "Hadamard", "distortion": distortion_method4_512, "wcd": wcdistortion_method4_512, "map": map_method4_512},
]


def figure_distortion_vs_height(methods_512):

    unique_heights = sorted(set(height))
    bar_width = 0.1  # Width of individual bars
    x_positions = np.arange(len(unique_heights))  # X positions for the groups

    colors = ['darkred', 'red', 'darkblue', 'blue']
    # Start plotting
    plt.figure(figsize=(10, 6))
    # Loop through methods and plot bars
    for i, method in enumerate(methods_512):
        plt.bar(
            x_positions + i * bar_width,
            [method["distortion"][h] for h in height_sorted_index],  # Distortion for each height
            width=bar_width,
            label=method["name"],
            color=colors[i],
        )

    for i, method in enumerate(methods_512):
        plt.bar(
            [x_positions[unique_heights.index(height[h])] + (i + 4) * bar_width for h in height_sorted_index_additional],
            [method["distortion"][h] for h in height_sorted_index_additional],  # Distortion for each height
            width=bar_width,
            # label=method["name"],
            color=colors[i],
            alpha=0.7  # Differentiate extra methods with transparency
        )

    x_shift = 0.08
    # Loop through the bar groups
    for i, label in enumerate(["5-ary", "4-ary", "3-ary       Barabasi", "2-ary       Binomial"]):
        # Calculate the x-position for the label
        group_center = x_positions[i] + (bar_width * (4 if i < 2 else 8)) / 2 - x_shift
        # Place the label above the bar group
        plt.text(group_center,
                 # max(distortion_method1[i], distortion_method2[i]) + 0.02,  # Adjust height dynamically
                 plt.ylim()[1] * 0.95,
                 label,
                 ha='center',
                 va='bottom',
                 fontsize=12,
                 fontweight='bold')

    # Add labels, legend, and grid
    plt.xlabel("Height", fontsize=14)
    plt.ylabel("Average Distortion (log scale)", fontsize=14)
    plt.title("Average Distortion vs Height", fontsize=14)
    # plt.xticks(x_positions + 1.5 * bar_width, unique_heights)  # Center x-ticks under groups
    plt.xticks([0.15, 1.15, 2.35, 3.35], unique_heights, fontsize=12)
    plt.legend(title="Methods", fontsize=10, loc="center left", bbox_to_anchor=(0.1, 0.8))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.yscale('log')
    # Adjust layout and show plot
    plt.tight_layout()
    # plt.show()
    plt.savefig("./AvgDistortion_vs_Height_N512_barplot.pdf")

figure_distortion_vs_height(methods_512)
# # Create the plot
# plt.figure(figsize=(8, 6))
# markers = ['o', 'x', '^', 'D', 'p', 's']  # Different marker types
# tree_type_handles = []
# for m, label in zip(markers, ["Binomial", "2-ary", "3-ary", "4-ary", "5-ary", "Barabasi"]):
#     tree_type_handles.append(plt.scatter([], [], marker=m, color='black', label=label))
#
# # plt.legend()
# for x, y0, y1, y2, y3, m in zip(height, distortion_method1_512, distortion_method2_512, distortion_method3_512, distortion_method4_512, markers):
#     plt.scatter(x, y0, marker=m, color='blue')
#     plt.scatter(x, y1, marker=m, color='darkblue')
#     plt.scatter(x, y2, marker=m, color='red')
#     plt.scatter(x, y3, marker=m, color='darkred')
#
# color_legend = [
#     Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=15, label='Poincare'),
#     Line2D([0], [0], marker='s', color='w', markerfacecolor='darkblue', markersize=15, label='Entailment'),
#     Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=15, label='Construction1'),
#     Line2D([0], [0], marker='s', color='w', markerfacecolor='darkred', markersize=15, label='Construction2')
# ]
#
# # Add labels, title, and legend
# plt.xlabel("Height", fontsize=14)
# plt.ylabel("Average Distortion", fontsize=14)
# plt.title("Average Distortion vs Height N=512", fontsize=14)
# plt.legend(handles=tree_type_handles + color_legend, title="Tree Types & Methods", loc="best", fontsize=10)
#
#
# # plt.xlim(0, 20)
# # Add grid for better readability
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.yscale('log')
# # Show the plot
# plt.tight_layout()
# plt.show()
# # plt.savefig("./wc_distortion_vs_Height_N512.pdf")


def figure_metrics_vs_r(methods_512, r):
    plt.figure(figsize=(8, 6))
    # Plot data for each method
    x = np.arange(len(r))  # positions for groups
    bar_width = 0.2
    metrics = ["AVG Distortion", "WC Distortion", "MAP"]
    colors = ["darkred", "red", "darkblue", "blue"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)

    for idx, method in enumerate(methods_512):
        ax = axes[idx]
        sub_column_offset = [(i - (len(metrics) / 2 - 0.5)) * bar_width for i in range(len(metrics))]

        # Plot bars
        ax.bar(x + sub_column_offset[0], method["distortion"][1:5], bar_width, label=metrics[0], color=colors[idx], alpha=0.9)
        ax.bar(x + sub_column_offset[1], method["wcd"][1:5], bar_width, label=metrics[1], color=colors[idx], alpha=0.6)
        ax.bar(x + sub_column_offset[2], method["map"][1:5], bar_width, label=metrics[2], color=colors[idx], alpha=0.4)

        # Set titles and x-ticks
        ax.set_title(f'{method["name"]}', fontsize=17)
        ax.set_xticks(x)
        ax.set_xticklabels(r)
        ax.legend(metrics, fontsize=12, loc="upper center")

        # Add grid for readability
        ax.grid(True, linestyle='--', alpha=0.7)

    # Add common x and y labels
    fig.text(0.51, 0.0, 'r', ha='center', fontsize=15)
    fig.text(0.0, 0.5, 'Metrics (log scale)', va='center', rotation='vertical', fontsize=15)

    # Shared y-scale
    axes[0].set_yscale('log')

    # Add legend
    plt.tight_layout()
    plt.savefig("./Metrics_vs_r_N512_barplot.pdf")


# # plt.plot(r, distortion_method1_256[1:5], marker='o', color='blue', label='Poincare 256', alpha=0.4)
# plt.bar(r, distortion_method1_512[1:5], color='blue', label='Poincare Avg Distortion', alpha=0.7)
# plt.bar(r, wcdistortion_method1_512[1:5], color='blue', label='Poincare Wc Distortion', alpha=0.7)
# # plt.plot(r, distortion_method1_1024[1:5], marker='o', color='blue', label='Poincare 1024', alpha=1)
#
# plt.bar(r, distortion_method2_512[1:5], color='darkblue', label='Entailment Avg Distortion', alpha=0.7)
# plt.bar(r, wcdistortion_method2_512[1:5], color='darkblue', label='Entailment Wc Distortion', alpha=0.7)
# # plt.plot(r, distortion_method3_256[1:5], marker='^', color='red', label='Construction 1 256', alpha=0.4)
# plt.bar(r, distortion_method3_512[1:5], color='red', label='Construction 1 Avg Distortion', alpha=0.7)
# plt.bar(r, wcdistortion_method3_512[1:5], color='red', label='Construction 1 Wc Distortion', alpha=0.7)
# # plt.plot(r, distortion_method3_1024[1:5], marker='^', color='red', label='Construction 1 1024', alpha=1)
#
# # plt.plot(r, distortion_method4_256[1:5], marker='d', color='darkred', label='Construction 2 256', alpha=0.4)
# plt.bar(r, distortion_method4_512[1:5], color='darkred', label='Construction 2 Avg Distortion', alpha=0.7)
# plt.bar(r, wcdistortion_method4_512[1:5], color='darkred', label='Construction 2 Wc Distortion', alpha=0.7)
# # plt.plot(r, distortion_method4_1024[1:5], marker='d', color='darkred', label='Construction 2 1024', alpha=1)


