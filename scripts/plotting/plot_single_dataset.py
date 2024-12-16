import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2
import radarmeetsvision as rmv

from pathlib import Path

class Plotter:
    def __init__(self, dataset_dir):
        dataset_dir = Path(dataset_dir)
        self.prediction_dir = dataset_dir / args.dir
        if not self.prediction_dir.is_dir():
            self.prediction_dir.mkdir(exist_ok=True)

    def callback(self, sample, prediction):
        index = int(sample['index'])
        prediction_file = self.prediction_dir / f'{index:05d}_p.jpg'
        prediction_np_file = self.prediction_dir / f'{index:05d}_p.npy'
        prediction_np = prediction.cpu().numpy().squeeze()
        np.save(prediction_np_file, prediction_np)
        plt.imsave(str(prediction_file), prediction_np, cmap='viridis')

def main(args):
    rmv.setup_global_logger()
    interface = rmv.Interface()
    interface.set_encoder('vitb')
    depth_min = 0.19983673095703125
    depth_max = 120.49285888671875
    interface.set_depth_range((depth_min, depth_max))
    interface.set_output_channels(args.outputchannels)

    interface.set_size(480, 640)
    interface.set_batch_size(1)
    interface.set_criterion()
    interface.set_use_depth_prior(bool(args.usedepthprior))

    interface.load_model(pretrained_from=args.network)
    _, loader = interface.get_single_dataset(args.dataset, min_index=args.min, max_index=args.max)

    plotter = Plotter(args.dataset)
    interface.validate_epoch(0, loader, iteration_callback=plotter.callback)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Purely evalute a network')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--network', type=str, help='Path to the network file')
    parser.add_argument('--dir', type=str, default='prediction')
    parser.add_argument('--min', type=int, default=0)
    parser.add_argument('--max', type=int, default=-1)
    parser.add_argument('--outputchannels', type=int, default=2)
    parser.add_argument('--usedepthprior', type=int, default=1)
    args = parser.parse_args()
    main(args)