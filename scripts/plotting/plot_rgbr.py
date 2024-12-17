import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2
import radarmeetsvision as rmv

from pathlib import Path

rgbr_dir = None

def prediction_callback(sample, prediction):
    image = sample['image'].cpu().numpy().squeeze()
    image = np.moveaxis(image, 0, -1)
    depth_prior = sample['depth_prior'].cpu().numpy().squeeze()

    image = cv2.resize(image, (640, 480))
    depth_prior = cv2.resize(depth_prior, (640, 480))

    nonzero_mask = (depth_prior > 0.0)
    ones = np.ones(nonzero_mask.sum())
    zeros = np.zeros(nonzero_mask.sum())
    red = np.vstack((ones, zeros, zeros)).T
    image[nonzero_mask] = red
    image = np.clip(image, 0.0, 1.0)
    plt.imsave(str(rgbr_dir / f"{int(sample['index']):05d}_rgbr.jpg"), image)

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
    _, loader = interface.get_single_dataset(args.dataset, min_index=0, max_index=-1)

    global rgbr_dir
    rgbr_dir = Path(args.dataset) / 'rgbr'
    if not rgbr_dir.is_dir():
        rgbr_dir.mkdir(exist_ok=True)

    interface.validate_epoch(0, loader, iteration_callback=prediction_callback)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Purely evalute a network')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--network', type=str, help='Path to the network file')
    args = parser.parse_args()
    main(args)
