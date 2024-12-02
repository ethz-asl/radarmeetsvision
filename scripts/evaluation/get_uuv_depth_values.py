from pathlib import Path
from scipy.signal import butter, filtfilt
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys

# This gives plot that can go in publications
import matplotlib as mpl
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

WINDOW_SIZE = 50

def process_npy_files(top_directory):
    folder_averages = {}  # To store averages for each folder
    initial_time = None

    # Try to read timestamps
    timestamp_dict = {}
    timestamps_file = Path(top_directory) / '..' / 'timestamps.txt'
    if timestamps_file.is_file():
        timestamps_lines = []
        with timestamps_file.open('r') as f:
            timestamps_lines = f.readlines()

        line_mask = r'([0-9]*)\: ([0-9\.]*)'

        for line in timestamps_lines:
            out = re.search(line_mask, line)
            if out is not None:
                if initial_time is None:
                    initial_time = float(out.group(2))
                timestamp_dict[int(out.group(1))] = (float(out.group(2)) - initial_time) / 1e9
    else:
        print("timestamps file does not exist!")

    # Try to load the dvl net distance
    dvl_net_file = Path(top_directory) / '..' / 'net_distances.txt'
    if dvl_net_file.is_file():
        dvl_lines = []
        with dvl_net_file.open('r') as f:
            dvl_lines = f.readlines()

        line_mask = r'([0-9]*)\: ([0-9\.]*)'
        distances = []
        for line in dvl_lines:
            out = re.search(line_mask, line)
            index = int(out.group(1))
            if out is not None and index in timestamp_dict.keys():
                distances.append((timestamp_dict[index], float(out.group(2))))

    folder_averages['DVL distance [m]'] = np.array(distances)

    # Try to load prior distance
    prior_file = Path(top_directory) / '..' / 'prior_distances_5.txt'
    if prior_file.is_file():
        prior_lines = []
        with prior_file.open('r') as f:
            prior_lines = f.readlines()

        line_mask = r'([0-9]*)\: ([0-9\.]*)'
        distances = []
        for line in prior_lines:
            out = re.search(line_mask, line)
            if out is not None:
                index = int(out.group(1))
                if index in timestamp_dict.keys() and out.group(2) != '':
                    distances.append((timestamp_dict[index], float(out.group(2))))
            else:
                print("Skipping", index, line)
                distances.append((0, 0.0))

        folder_averages['Avg. FFT Prior [m]'] = np.array(distances)

    # Iterate through all folders in the top-level directory
    averages = []
    averages_file = Path(top_directory) / '..' / 'averages.txt'
    for file_name in os.listdir(top_directory):
        if file_name.endswith(".npy"):
            index = int(file_name[:-4])
            file_path = os.path.join(top_directory, file_name)

            matrix = np.load(file_path)

            # Find the center 100x100 pixel field
            h, w = matrix.shape
            center_x, center_y = h // 2, w // 2
            start_x = max(center_x - WINDOW_SIZE//2, 0)
            start_y = max(center_y - WINDOW_SIZE//2, 0)

            # Adjust in case matrix dimensions are smaller than 100x100
            end_x = min(start_x + WINDOW_SIZE, h)
            end_y = min(start_y + WINDOW_SIZE, w)
            center_field = matrix[start_x:end_x, start_y:end_y]

            # Compute the average
            STEREO_TO_IMU_DEPTH_CALIB = 0.04
            avg_value = np.mean(center_field) + STEREO_TO_IMU_DEPTH_CALIB

            # Append the result as (index, average value)
            averages.append((timestamp_dict[index], avg_value))

            # Store averages for plotting
    averages_np = np.array(averages)
    averages_np = averages_np[averages_np[:, 0].argsort()]
    folder_averages['Avg. depth [m]'] = averages_np

    # Write the averages to a file in the same folder
    try:
        with open(averages_file, "w") as f:
            for index, avg in averages:
                f.write(f"{avg}\n")
        print(f"Processed folder: {top_directory}")
    except Exception as e:
        print(f"Error writing output file for folder {top_directory}: {e}")

    return folder_averages

def low_pass_filter(data, cutoff=0.025, order=2):
    """
    Apply a low-pass Butterworth filter to the data.
    
    Args:
        data: List or numpy array of values to filter.
        cutoff: Normalized cutoff frequency (0.1 = 10% of the Nyquist frequency).
        order: Order of the Butterworth filter.
    
    Returns:
        Filtered data as a numpy array.
    """
    b, a = butter(order, cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def plot_averages(folder_averages):
    # Define a list of marker styles
    marker_styles = itertools.cycle(['o', 's', 'D', '^', 'v', '*', '+', 'x'])

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    for folder_name, averages in folder_averages.items():
        # Plot original averages
        marker = next(marker_styles)
        averages_np = np.array(averages)
        axs[0].plot(
            averages_np[:, 0],
            averages_np[:, 1],
            label=folder_name,
            marker=marker,
            markersize=3
        )
        
        # Apply low-pass filter and plot
        filtered_averages = low_pass_filter(averages_np[:, 1])
        axs[1].plot(
            averages_np[:, 0],
            filtered_averages,
            label=f"{folder_name} (LP)",
            linewidth=2
        )

    # Add details to the original averages plot
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Average Depth [m]")
    # axs[0].set_xlim(400, 3500)
    # axs[0].set_ylim(0, 3.0)
    axs[0].legend(loc="best")
    axs[0].grid(True)

    # Add details to the filtered averages plot
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("LP Average Depth [m]")
    # axs[1].set_xlim(400, 3500)
    # axs[1].set_ylim(0, 3.0)
    axs[1].legend(loc="best")
    axs[1].grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    #plt.savefig("figure.png")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <top_directory_path>")
        sys.exit(1)
    
    top_directory = sys.argv[1]

    # Ensure the directory exists
    if not os.path.isdir(top_directory):
        print(f"Error: {top_directory} is not a valid directory.")
        sys.exit(1)

    # Process .npy files and compute averages
    folder_averages = process_npy_files(top_directory)

    # Plot the computed averages
    if folder_averages:
        plot_averages(folder_averages)
    else:
        print("No averages found to plot.")

