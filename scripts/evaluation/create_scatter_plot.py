######################################################################
#
# Copyright (c) 2024 ETHZ Autonomous Systems Lab. All rights reserved.
#
######################################################################

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.gridspec import GridSpec
from pathlib import Path
from PIL import Image

mpl.rcParams['font.size'] = 8
mpl.rcParams['figure.figsize'] = [9, 2.7]
mpl.rcParams['lines.linewidth'] = 0.6
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['axes.linewidth'] = 0.6
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

def create_scatter_plot(results_per_scenario_sample, config, output_dir, subsample=1):
    scenario_color = {
        'Industrial Hall': 'k',
        'Agricultural Field': 'green',
        'Rhône Glacier': 'navy'
    }
    label_dict = {
        "Metric Depth \\cite{depthanythingv2}-B": "Metric Depth-B",
        "Ours-B": "Ours-B",
        "Scaled Relative Depth \\cite{depthanythingv2}-B": "Scaled Relative-B"
    }
    samples = [['00000_rgb.jpg', '00000_dp.jpg'],
               ['00050_rgb.jpg', '00050_dp.jpg'],
               ['00250_rgb.jpg', '00250_dp.jpg']]

    colors = {
        "Ours-B": "#0072B2",
        "Metric Depth \\cite{depthanythingv2}-B": "#D55E00",
        "Scaled Relative Depth \\cite{depthanythingv2}-B": "#009E73",
    }

    sample_dir = Path('scripts/evaluation/samples')
    img_out = None
    for i, sample in enumerate(samples):
        rgb_file = sample_dir / sample[0]
        dp_file = sample_dir / sample[1]

        rgb = np.array(Image.open(rgb_file))
        dp = np.array(Image.open(dp_file))
        img = np.concatenate((rgb, dp), axis=1)
        if img_out is None:
            img_out = img
        else:
            img_out = np.concatenate((img_out, img), axis=0)
    border = 5
    img_out[:border, :] = (0.0, 0.0, 0.0)
    img_out[-border:, :] = (0.0, 0.0, 0.0)
    img_out[:, :border] = (0.0, 0.0, 0.0)
    img_out[:, -border:] = (0.0, 0.0, 0.0)

    # Update figure layout for 2 columns and 3 rows in the right column
    fig = plt.figure(figsize=(12, 6))  # Adjust width to fit two columns
    gs = GridSpec(3, 2, width_ratios=[2, 3], height_ratios=[1, 1, 1])  # 3 rows, 2 columns

    # Left column: Combined image
    ax0 = fig.add_subplot(gs[:, 0])  # Use all 3 rows of the left column for the image
    ax0.imshow(img_out)
    ax0.axis('off')

    # Right column: 3 rows of scatter plots
    axs = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, 1])]
    for i, scenario_key in enumerate(config['scenarios'].keys()):
        for network_key in config['networks'].keys():
            if '-B' in network_key: # and not 'Scaled' in network_key:
                average_depths = results_per_scenario_sample[scenario_key][network_key]['average_depth']
                abs_rel_values = results_per_scenario_sample[scenario_key][network_key]['abs_rel']
                average_depths_subsampled = average_depths[::subsample]
                abs_rel_values_subsampled = abs_rel_values[::subsample]
                label = label_dict[network_key] + ' ' + (scenario_key if 'Metric' in network_key else '')
                axs[i].scatter(average_depths_subsampled, abs_rel_values_subsampled, label=label, marker=config[network_key]['marker'], s=25, alpha=0.5, c=colors[network_key])
                if i == 2:
                    axs[i].set_xlabel('Average Scene Depth [m]')

                axs[i].set_ylabel('Absolute Relative Error [ ]')
                axs[i].legend(loc='upper right')
                axs[i].grid()

    plt.tight_layout()
    output_file = Path(output_dir) / 'results_overview.png'
    plt.savefig(str(output_file), transparent=False, bbox_inches='tight', dpi=400)
    plt.close()

    # Post-process the saved image to crop any unnecessary white space
    img = Image.open(str(output_file))
    img = img.convert("RGBA")
    bbox = img.getbbox()
    cropped_img = img.crop(bbox)
    cropped_img.save(str(output_file))
