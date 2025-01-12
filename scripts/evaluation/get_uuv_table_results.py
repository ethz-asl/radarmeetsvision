from pathlib import Path
from scipy.signal import butter, filtfilt
import numpy as np
import os
import re
import sys

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
        distances = {}
        for line in dvl_lines:
            out = re.search(line_mask, line)
            index = int(out.group(1))
            if out is not None and index in timestamp_dict.keys():
                distances[index] = float(out.group(2))

    folder_averages['DVL'] = np.array(distances)

    # Iterate through all folders in the top-level directory
    averages_ap_gt = []
    averages_ap_pr = []
    averages_fft = []
    averages_dvl = []
    for file_name in os.listdir(top_directory):
        if file_name.endswith(".npy"):
            index = int(file_name[:-4])
            file_path = Path(top_directory) / file_name
            prediction = np.load(file_path)

            # Try to load the depth groundtruth from april tags
            depth_file = Path(top_directory) / '..' / 'depth' / f'{index:05d}_d.npy'
            fft_depth_file = Path(top_directory) / '..' / 'depth_fft' / f'{index:05d}_d.npy'
            if depth_file.is_file() and fft_depth_file.is_file() and index in distances.keys():
                depth = np.load(depth_file)
                depth_groundtruth_mask = depth > 0.0
                avg_pred = prediction[depth_groundtruth_mask].mean()
                avg_gt = depth[depth_groundtruth_mask].mean()
                averages_ap_pr.append((timestamp_dict[index], avg_pred))
                averages_ap_gt.append((timestamp_dict[index], avg_gt))

                fft_depth = np.load(fft_depth_file)
                fft_mask = fft_depth > 0.0
                avg_fft = fft_depth[fft_mask].mean()
                averages_fft.append((timestamp_dict[index], avg_fft))
                averages_dvl.append((timestamp_dict[index], distances[index]))

    # Store averages for plotting
    averages_fft_np = np.array(averages_fft)
    if len(averages_fft_np):
        averages_fft_np = averages_fft_np[averages_fft_np[:, 0].argsort()]
        folder_averages['FFT'] = averages_fft_np[:, 1]

    averages_ap_pr_np = np.array(averages_ap_pr)
    if len(averages_ap_pr_np):
        averages_ap_pr_np = averages_ap_pr_np[averages_ap_pr_np[:, 0].argsort()]
        folder_averages['RMV'] = averages_ap_pr_np[:, 1]

    averages_ap_gt_np = np.array(averages_ap_gt)
    if len(averages_ap_gt_np):
        averages_ap_gt_np = averages_ap_gt_np[averages_ap_gt_np[:, 0].argsort()]
        folder_averages['GT'] = averages_ap_gt_np[:, 1]

    averages_dvl_np = np.array(averages_dvl)
    if len(averages_dvl_np):
        averages_dvl_np = averages_dvl_np[averages_dvl_np[:, 0].argsort()]
        folder_averages['DVL'] = averages_dvl_np[:, 1]

    rmv_avg = (np.abs(folder_averages['RMV'] - folder_averages['GT']) / folder_averages['GT']).mean()
    fft_avg = (np.abs(folder_averages['FFT'] - folder_averages['GT']) / folder_averages['GT']).mean()
    dvl_avg = (np.abs(folder_averages['DVL'] - folder_averages['GT']) / folder_averages['GT']).mean()
    print(f"FFT: {fft_avg:.3f}, RMV: {rmv_avg:.3f}, DVL: {dvl_avg:.3f}, N: {len(folder_averages['GT'])}")
    return folder_averages

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
