import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

# Configure matplotlib to match the example style
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

def unnormalize_depth(depth, depth_min=0.19983673095703125, depth_max=120.49285888671875):
    """
    Unnormalizes a depth file given min and max depth.
    Assumes input is in range [0, 1] for valid (nonzero) values.
    Zeros remain unchanged as they encode no depth.
    """
    valid_mask = depth > 0  # Only unnormalize nonzero values
    unnormalized = np.zeros_like(depth)  # Start with zeros
    unnormalized[valid_mask] = depth[valid_mask] * (depth_max - depth_min) + depth_min
    return unnormalized


def load_npy_files(base_dir):
    # Define directories based on the base directory
    depth_dir = base_dir / "depth"
    prediction_rgb_dir = base_dir / "prediction_rgb"
    prediction_radar_dir = base_dir / "prediction_radar"

    # Check if directories exist
    if not all(d.is_dir() for d in [depth_dir, prediction_rgb_dir, prediction_radar_dir]):
        print("One or more directories (depth, prediction_rgb, prediction_radar) are missing.")
        return {}, {}, {}

    # Gather depth files
    depth_files = sorted(depth_dir.glob("*_d*.npy"))  # Match both *_d.npy and *_dn.npy

    # Initialize storage for loaded data
    depth_data = {}
    rgb_data = {}
    radar_data = {}

    for depth_file in depth_files:
        index = depth_file.stem.split("_")[0]
        rgb_file = prediction_rgb_dir / f"{index}_p.npy"
        radar_file = prediction_radar_dir / f"{index}_p.npy"

        # Check if corresponding files exist in both prediction directories
        if rgb_file.exists() and radar_file.exists():
            try:
                # Load the depth file and check for unnormalization
                depth = np.load(depth_file)
                if depth_file.stem.endswith("_dn"):
                    depth = unnormalize_depth(depth)

                # Load predictions
                rgb_data[index] = np.load(rgb_file)
                radar_data[index] = np.load(radar_file)
                depth_data[index] = depth
            except Exception as e:
                print(f"Error loading files for index {index}: {e}")
        else:
            print(f"Missing prediction files for index {index} (RGB or Radar).")

    print(f"Loaded {len(depth_data)} sets of files.")
    return depth_data, rgb_data, radar_data


def calculate_metrics_for_index(depth, rgb_pred, radar_pred):
    # Mask for valid depth pixels (non-zero depth values)
    valid_mask = depth > 0

    if np.any(valid_mask):
        # Calculate average scene depth
        avg_depth = np.mean(depth[valid_mask])

        # Calculate MARE for RGB predictions
        rgb_error = np.abs((rgb_pred[valid_mask] - depth[valid_mask]) / depth[valid_mask])
        mare_rgb = np.mean(rgb_error)

        # Calculate MARE for Radar predictions
        radar_error = np.abs((radar_pred[valid_mask] - depth[valid_mask]) / depth[valid_mask])
        mare_radar = np.mean(radar_error)

        return avg_depth, mare_rgb, mare_radar
    return None, None, None


def plot_metrics_step(avg_scene_depths, mare_rgb, mare_radar, output_dir, step):
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the scatter plot
    fig = plt.figure(figsize=(5.4, 2.7))  # Match the style and layout of the example
    gs = GridSpec(1, 1)  # Single grid for the scatter plot

    ax = fig.add_subplot(gs[0])

    # Scatter plots
    ax.scatter(avg_scene_depths, mare_rgb, label="RGB-Method-B", color="red", alpha=0.7, s=25, marker='o')
    ax.scatter(avg_scene_depths, mare_radar, label="Ours-B", color="blue", alpha=0.7, s=25, marker='o')

    # Axis labels, grid, and legend
    ax.set_xlabel("Average Scene Depth [m]", fontsize=8)
    ax.set_ylabel("Absolute Relative Error [ ]", fontsize=8)
    ax.grid(linewidth=0.5)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    # Save the figure for this step
    output_file = output_dir / f"{step:03d}.png"
    plt.savefig(str(output_file), bbox_inches="tight", dpi=400)
    plt.close()


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Load corresponding depth, prediction RGB, and prediction radar .npy files, and create step-by-step scatter plots."
    )
    parser.add_argument(
        "base_dir",
        type=str,
        help="Path to the base directory containing 'depth', 'prediction_rgb', and 'prediction_radar' subdirectories.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Resolve the base directory and create the output directory
    base_dir = Path(args.base_dir).resolve()
    output_dir = base_dir / "mare"  # Create 'mare' directory inside the base directory

    # Validate base directory
    if not base_dir.is_dir():
        print(f"Provided base_dir '{base_dir}' is not a valid directory.")
        return

    # Load files
    depth_data, rgb_data, radar_data = load_npy_files(base_dir)

    # Initialize storage for metrics
    avg_scene_depths = []
    mare_rgb = []
    mare_radar = []

    # Process files in ascending order of indices
    sorted_indices = sorted(depth_data.keys(), key=lambda x: int(x))  # Sort indices numerically
    for step, index in enumerate(sorted_indices):
        depth = depth_data[index]
        rgb_pred = rgb_data.get(index)
        radar_pred = radar_data.get(index)

        if rgb_pred is None or radar_pred is None:
            continue

        # Calculate metrics for the current index
        avg_depth, rgb_mare, radar_mare = calculate_metrics_for_index(depth, rgb_pred, radar_pred)

        if avg_depth is not None:
            avg_scene_depths.append(avg_depth)
            mare_rgb.append(rgb_mare)
            mare_radar.append(radar_mare)

            # Plot and save the current step
            plot_metrics_step(avg_scene_depths, mare_rgb, mare_radar, output_dir, step)

    print(f"Step-by-step plots saved in '{output_dir}'.")


if __name__ == "__main__":
    main()
