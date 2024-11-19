######################################################################
#
# Copyright (c) 2024 ETHZ Autonomous Systems Lab. All rights reserved.
#
######################################################################

import argparse
import logging
import pickle
import radarmeetsvision as rmv

from pathlib import Path
from results_table_template import generate_tables
from create_scatter_plot import create_scatter_plot

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Purely evalute a network')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--output', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--network', type=str, help='Path to the network file')
    args = parser.parse_args()
    rmv.setup_global_logger()

    interface = rmv.Interface()
    interface.set_encoder('vitb')
    depth_min = 0.19983673095703125
    depth_max = 120.49285888671875
    interface.set_depth_range((depth_min, depth_max))
    interface.set_output_channels(2)
    interface.set_use_depth_prior(True)
    interface.load_model(pretrained_from=args.network)

    interface.set_size(480, 640)
    interface.set_batch_size(1)
    interface.set_criterion()

    loader = interface.get_single_dataset_loader(args.dataset, min_index=0, max_index=-1)
    interface.validate_epoch(0, loader, save_outputs=True)



if __name__ == "__main__":
    main()
