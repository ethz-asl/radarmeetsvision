import argparse
import matplotlib.pyplot as plt
import numpy as np
import pywavemap as wave
import radarmeetsvision as rmv
import re

from pathlib import Path

class WavemapPipeline:
    def __init__(self, dataset_dir):
        self.map = wave.Map.create({
            "type": "hashed_chunked_wavelet_octree",
            "min_cell_width": {
                "meters": 0.05
            }
        })

        # Create a measurement integration pipeline
        self.pipeline = wave.Pipeline(self.map)

        # Add map operations
        self.pipeline.add_operation({
            "type": "threshold_map",
            "once_every": {
                "seconds": 5.0 # TODO: What does this mean in this context?
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
                        "meters": 0.02 # TODO: What makes sense here? Relative possible?
                    },
                    "scaling_free": 0.4, # TODO: Was 0.2, 0.4
                    "scaling_occupied": 0.8 # TODO
                },
                "integration_method": {
                    "type": "hashed_chunked_wavelet_integrator",
                    "min_range": {
                        "meters": 0.1
                    },
                    "max_range": {
                        "meters": 20.0
                    }
                },
            })

        self.dataset_dir = Path(dataset_dir)
        self.pose_dict = self.read_pose_dict(dataset_dir)

        self.initial_pose = None

    def read_pose_dict(self, dataset_dir):
        pose_mask = r'-?\d+\.\d+(?:e[+-]?\d+)?|-?\d+'
        pose_dict = {}
        pose_file = Path(dataset_dir) / 'pose_file.txt'
        if pose_file.is_file():
            with pose_file.open('r') as f:
                lines = f.readlines()
                for line in lines:
                    out = re.findall(pose_mask, line)
                    if out is not None:
                        index = int(out[0])
                        pose_dict[index] = np.zeros((4, 4))
                        pose_dict[index][3, 3] = 1.0
                        for i in range(12):
                            row = i // 4
                            col = i % 4
                            pose_dict[index][row][col] = float(out[i + 1])
        else:
            print(f'Could not find {pose_file}')

        return pose_dict

    def integrate_wavemap(self, depth_prediction_np, pose_np):
        if self.initial_pose is None:
            self.initial_pose = self.pose_dict[index]

        # Pose in the frame of the first pose
        pose_np = np.linalg.inv(self.initial_pose) @ pose_np

        image = wave.Image(depth_prediction_np)
        pose = wave.Pose(pose_np)

        # Integrate the depth image
        self.pipeline.run_pipeline(["rmv"], wave.PosedImage(pose, image))

    def prediction_callback(self, index, depth_prediction):
        depth_prediction_np = depth_prediction.cpu().numpy().squeeze()
        file = self.dataset_dir / 'prediction' / f'{index:05d}_pred'
        np.save(str(file)+'.npy', depth_prediction_np)
        plt.imsave(str(file)+'.jpg', depth_prediction_np)
        pose_np = self.pose_dict[index]

        print(f"Integrating measurement {index}")
        self.integrate_wavemap(depth_prediction_np.T, pose_np)

    def prediction_reader(self, prediction_dir, dataset):
        filelist = dataset.filelist
        for i in range(len(dataset)):
            index = filelist[i]
            prediction_file = prediction_dir / f'{index:05d}_pred.npy'
            depth_prediction_np = np.load(str(prediction_file))
            pose_np = self.pose_dict[index]

            print(f"Integrating measurement {index}")
            self.integrate_wavemap(depth_prediction_np.T, pose_np)


    def finalize(self):
        # Remove map nodes that are no longer needed
        self.map.prune()

        # Save the map
        print(f"Saving map of size {self.map.memory_usage} bytes")
        self.map.store('/home/asl/Downloads/output.wvmp')

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
    dataset, loader = interface.get_single_dataset(args.dataset, min_index=0, max_index=-1)

    wp = WavemapPipeline(args.dataset)

    predictions_dir = Path(args.dataset) / 'prediction'
    if not predictions_dir.is_dir():
        predictions_dir.mkdir(exist_ok=True)
        interface.validate_epoch(0, loader, iteration_callback=wp.prediction_callback)
    else:
        wp.prediction_reader(predictions_dir, dataset)

    wp.finalize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Purely evalute a network')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--network', type=str, help='Path to the network file')
    args = parser.parse_args()
    main(args)
