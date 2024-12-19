######################################################################
#
# Copyright (c) 2024 ETHZ Autonomous Systems Lab. All rights reserved.
#
######################################################################

import argparse
import json
import logging
import re
import radarmeetsvision as rmv

from pathlib import Path

logger = logging.getLogger(__name__)

# TODO: This is just copy pasta, not so nice
class Evaluation:
    def __init__(self, config, scenario_key, network_key, args):
        self.results_per_sample = {}
        self.results_dict = {}
        self.interface = rmv.Interface()
        self.networks_dir = None
        if args.network is not None:
            self.networks_dir = Path(args.network)

        self.datasets_dir = args.dataset
        self.results, self.results_per_sample = None, None
        self.setup_interface(config, scenario_key, network_key)
        self.run(network_key)

    def run(self, network_key):
        self.interface.validate_epoch(0, self.loader)
        self.results, self.results_per_sample = self.interface.get_results()
        self.results['method'] = network_key

    def setup_interface(self, config, scenario_key, network_key):
        self.interface.set_epochs(1)
        self.interface.set_encoder(config['encoder'])
        self.interface.set_depth_range((config['depth_min'], config['depth_max']))
        self.interface.set_output_channels(config['output_channels'])
        self.interface.set_use_depth_prior(config['use_depth_prior'])
        self.interface.load_model(pretrained_from=network_key)
        self.interface.set_size(config['height'], config['width'])
        self.interface.set_batch_size(1)
        self.interface.set_criterion()

        dataset_list = [config['scenarios'][scenario_key]]
        index_list = [config['index'][config["scenarios"][scenario_key]]]
        self.loader, _ = self.interface.get_dataset_loader('val_all', self.datasets_dir, dataset_list, index_list)

    def get_results_per_sample(self):
        return self.results_per_sample

    def get_results(self):
        return self.results

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config


def main():
    parser = argparse.ArgumentParser(description='Create results table for paper')
    parser.add_argument('--config', type=str, default='scripts/evaluation/config_best.json', help='Path to the JSON config file.')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--output', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--network', type=str, help='Path to the network directory')

    args = parser.parse_args()

    rmv.setup_global_logger(args.output)

    if not Path(args.config).is_file():
        raise FileNotFoundError(f'The config file {args.config} does not exist.')

    if not Path(args.dataset).is_dir():
        raise FileNotFoundError(f'The dataset dir {args.dataset} does not exist.')

    config = load_config(args.config)
    network_dir = Path(args.network)
    pth_files = list(network_dir.rglob("*.pth"))

    best_abs_rel, best_abs_rel_file = 1000.0, None
    file_mask = r"latest_([0-9]*).pth"
    for pth_file in pth_files:
        out = re.search(file_mask, str(pth_file))
        if out is not None:

            # NOTE: Consider only 25 epochs for all, no improvements observed after that
            index = int(out.group(1))
            if index < 24:
                abs_rel = 0.0
                for scenario_key in config['scenarios'].keys():
                    logger.info(f'Evaluation: {scenario_key} {pth_file}')
                    eval_obj = Evaluation(config, scenario_key, pth_file, args)
                    results_dict = eval_obj.get_results_dict()
                    abs_rel += results_dict['abs_rel']

                abs_rel /= len(config['scenarios'].keys())
                if abs_rel < best_abs_rel:
                    best_abs_rel = abs_rel
                    best_abs_rel_file = pth_file
                    logger.info("==========================================")
                    logger.info(f"{pth_file}: {abs_rel:.3f}")
                    logger.info("==========================================")


if __name__ == "__main__":
    main()