import argparse
import numpy as np
import pywavemap as wave
import radarmeetsvision as rmv

from pathlib import Path
from PIL import Image as PilImage

"""
        # Load transform
        pose_file = file_path_prefix + "_pose.txt"
        if not os.path.isfile(pose_file):
            print(f"Could not find pose file '{pose_file}'")
            current_index += 1
            raise SystemExit
        if os.path.isfile(pose_file):
            with open(pose_file) as f:
                pose_data = [float(x) for x in f.read().split()]
                transform = np.eye(4)
                for row in range(4):
                    for col in range(4):
                        transform[row, col] = pose_data[row * 4 + col]
        pose = wave.Pose(transform)
 """

class WavemapPipeline:
    def __init__(self):
        self.map = wave.Map.create({
            "type": "hashed_chunked_wavelet_octree",
            "min_cell_width": {
                "meters": 0.05
            }
        })

        # Create a measurement integration pipeline
        self.pipeline = wave.Pipeline(map)

        # Add map operations
        self.pipeline.add_operation({
            "type": "threshold_map",
            "once_every": {
                "seconds": 5.0
            }
        })

        # Add a measurement integrator
        # TODO: What about distortion
        self.pipeline.add_integrator(
            "rmv", {
                "projection_model": {
                    "type": "pinhole_camera_projector",
                    "width": 640,
                    "height": 480,
                    "fx": 600.215588472,
                    "fy": 600.915799608,
                    "cx": 317.051855594,
                    "cy": 216.108482975
                },
                "measurement_model": {
                    "type": "continuous_ray",
                    "range_sigma": {
                        "meters": 0.1 # TODO: What makes sense here? Relative possible?
                    },
                    "scaling_free": 0.2, # TODO: 0.5 / 0.5
                    "scaling_occupied": 0.4 # TODO
                },
                "integration_method": {
                    "type": "hashed_chunked_wavelet_integrator",
                    "min_range": {
                        "meters": 0.1
                    },
                    "max_range": {
                        "meters": 10.0
                    }
                },
            })

    def prediction_callback(self, index, depth_prediction):
        depth_prediction_np = depth_prediction.cpu().numpy()
        import pdb; pdb.set_trace()

        image = wave.Image(depth_prediction_np)
        pose = self.get_pose(index)

        # Integrate the depth image
        print(f"Integrating measurement {index}")
        self.pipeline.run_pipeline(["rmv"], wave.PosedImage(pose, image))

    def finalize(self):
        # Remove map nodes that are no longer needed
        self.map.prune()

        # Save the map
        print(f"Saving map of size {self.map.memory_usage} bytes")
        self.map.store('output.wvmp')

        del self.pipeline, self.map

def main(args):
    rmv.setup_global_logger()
    interface = rmv.Interface()
    interface.set_encoder('vitb')
    depth_min = 0.19983673095703125
    depth_max = 120.49285888671875
    interface.set_depth_range((depth_min, depth_max))
    interface.set_output_channels(2)

    interface.set_size(480, 640)
    interface.set_batch_size(1)
    interface.set_criterion()
    interface.set_use_depth_prior(True)

    interface.load_model(pretrained_from=args.network)
    loader = interface.get_single_dataset_loader(args.dataset, min_index=0, max_index=-1)

    wp = WavemapPipeline()
    interface.validate_epoch(0, loader, iteration_callback=wp.prediction_callback)
    wp.finalize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Purely evalute a network')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--network', type=str, help='Path to the network file')
    args = parser.parse_args()
    main()
