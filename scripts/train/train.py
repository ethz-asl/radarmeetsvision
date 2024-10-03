import argparse
import radarmeetsvision as rmv

from pathlib import Path

def main(config, checkpoints, datasets, results):
    interface = rmv.Interface()

    interface.set_epochs(config['epochs'])
    interface.set_encoder(config['encoder'])

    depth_min, depth_max = config['depth_min'], config['depth_max']
    interface.set_depth_range((depth_min, depth_max))
    interface.set_output_channels(config['output_channels'])
    interface.set_use_depth_prior(config['use_depth_prior'])

    pretrained_from = None
    if config['pretrained_from'] is not None:
        pretrained_from = Path(checkpoints) / config['pretrained_from']
    interface.load_model(pretrained_from=pretrained_from)

    interface.set_results_path(results)
    interface.set_optimizer()

    interface.set_size(config['height'], config['width'])
    interface.set_batch_size(1)
    interface.set_criterion()

    loaders = {}
    for task in config['task'].keys():
        dataset_list = config['task'][task]['datasets']
        datasets_dir = Path(datasets) / config['task'][task]['dir']
        loader, _ = interface.get_dataset_loader(task, str(datasets_dir), dataset_list)
        loaders[task] = loader

    for epoch in range(config['epochs']):
        if loaders.get('train_all', None) is not None:
            interface.train_epoch(epoch, loaders['train_all'])

        if loaders.get('val_all', None) is not None:
            interface.validate_epoch(epoch, loaders['val_all'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train metric depth prediction')
    parser.add_argument('--checkpoints', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--datasets', type=str, required=True)
    parser.add_argument('--results', type=str, required=True)
    args = parser.parse_args()

    rmv.setup_global_logger(args.results)

    config = rmv.load_config(args.config)
    main(config, args.checkpoints, args.datasets, args.results)