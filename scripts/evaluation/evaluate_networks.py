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


class Evaluation:
    def __init__(self, config, scenario_key, network_key, args):
        self.results_per_sample = {}
        self.results_dict = {}
        self.interface = rmv.Interface()
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

        network_config = config[network_key]
        self.interface.set_encoder(network_config['encoder'])

        depth_min, depth_max = network_config['depth_min'], network_config['depth_max']
        self.interface.set_depth_range((depth_min, depth_max))
        self.interface.set_output_channels(network_config['output_channels'])
        self.interface.set_use_depth_prior(network_config['use_depth_prior'])

        network_file = config['networks'][network_key]
        if network_file is not None:
            network_file = self.networks_dir / network_file
        self.interface.load_model(pretrained_from=network_file)

        self.interface.set_size(config['height'], config['width'])
        self.interface.set_batch_size(1)
        self.interface.set_criterion()

        dataset_list = [config['scenarios'][scenario_key]]
        self.loader, _ = self.interface.get_dataset_loader('val_all', self.datasets_dir, dataset_list)

    def get_results_per_sample(self):
        return self.results_per_sample

    def get_results(self):
        return self.results


def main():
    parser = argparse.ArgumentParser(description='Create evaluation results for paper')
    parser.add_argument('--config', type=str, default='scripts/evaluation/config.json', help='Path to the JSON config file.')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--output', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--network', type=str, help='Path to the network directory')
    args = parser.parse_args()

    rmv.setup_global_logger()

    config = rmv.load_config(args.config)

    results_per_scenario = {}
    results_per_scenario_sample = {}
    results_per_scenario_file = Path(args.output) / "results_per_scenario.pkl"
    results_per_scenario_sample_file = Path(args.output) / "results_per_scenario_sample.pkl"
    if not results_per_scenario_file.is_file() or not results_per_scenario_sample_file.is_file():
        for scenario_key in config['scenarios'].keys():
            results_per_scenario[scenario_key] = []
            results_per_sample = {}

            for network_key in config['networks'].keys():
                logger.info(f'Evaluation: {scenario_key} {network_key}')
                eval_obj = Evaluation(config, scenario_key, network_key, args)
                results_per_sample[network_key] = eval_obj.get_results_per_sample()

                results_dict = eval_obj.get_results()
                if results_dict:
                    # Used to generate main results table
                    results_per_scenario[scenario_key].append(results_dict.copy())
                    logger.info(f'{scenario_key} {network_key} {results_dict["abs_rel"]:.3f}')

            # Used to generate visual grid
            results_per_scenario_sample[scenario_key] = results_per_sample

        with results_per_scenario_file.open('wb') as f:
            pickle.dump(results_per_scenario, f)

        with results_per_scenario_sample_file.open('wb') as f:
            pickle.dump(results_per_scenario_sample, f)

    else:
        with results_per_scenario_file.open('rb') as f:
            results_per_scenario = pickle.load(f)

        with results_per_scenario_sample_file.open('rb') as f:
            results_per_scenario_sample = pickle.load(f)

    # Generate the main results table
    if results_per_scenario:
        generate_tables(args.output, results_per_scenario)

    create_scatter_plot(results_per_scenario_sample, config, args.output)


if __name__ == "__main__":
    main()
