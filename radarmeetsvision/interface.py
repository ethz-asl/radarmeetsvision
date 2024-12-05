######################################################################
#
# Copyright (c) 2024 ETHZ Autonomous Systems Lab. All rights reserved.
#
######################################################################

import torch
import logging

from .dataset_helper import MultiDatasetLoader
from .metric_depth_network import *
from .utils import get_device

from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class Interface:
    def __init__(self):
        self.batch_size = None
        self.criterion = None
        self.depth_min_max = None
        self.depth_prior_dir = None
        self.device = get_device()
        self.encoder = None
        self.lr = None
        self.max_depth = None
        self.min_depth = None
        self.model = None
        self.optimizer = None
        self.output_channels = None
        self.previous_best = self.reset_previous_best()
        self.results = None
        self.results_path = None
        self.use_depth_prior = None

    def reset_previous_best(self):
        return {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100,
        'average_depth': 0.0}

    def set_use_depth_prior(self, use, depth_prior_dir=None):
        if depth_prior_dir is not None and Path(depth_prior_dir).is_dir():
            logger.info(f"Overwriting depth prior dir: {depth_prior_dir}")
            self.depth_prior_dir = Path(depth_prior_dir)

        self.use_depth_prior = use

    def set_encoder(self, encoder):
        self.encoder = encoder

    def set_depth_range(self, depth_min_max):
        self.depth_min_max = depth_min_max
        self.min_depth = self.depth_min_max[0]
        self.max_depth = self.depth_min_max[1]

    def set_output_channels(self, output_channels):
        self.output_channels = output_channels

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_criterion(self):
        self.criterion = SiLogLoss().to(self.device)

    def set_size(self, height, width):
        self.size = (height, width)

    def set_epochs(self, epochs):
        self.epochs = epochs

    def set_results_path(self, results_path):
        results_path = Path(results_path)
        if results_path.is_dir:
            self.results_path = Path(results_path)

        else:
            logger.error(f'{self.results_path} does not exist')

    def get_results(self):
        return self.results, self.results_per_sample

    def load_model(self, pretrained_from=None):
        if self.encoder is not None and self.max_depth is not None and self.output_channels is not None and self.use_depth_prior is not None:
            logger.info(f'Using pretrained from: {pretrained_from}')
            logger.info(f'Using depth prior: {self.use_depth_prior}')
            logger.info(f'Using encoder: {self.encoder}')
            logger.info(f'Using max depth: {self.max_depth}')
            logger.info(f'Using output channels: {self.output_channels}')

            self.model = get_model(pretrained_from, self.use_depth_prior, self.encoder, self.max_depth, output_channels=self.output_channels)
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model.to(self.device)
        else:
            logger.error(f"One or multiple undefined: {self.encoder}, {self.max_depth}, {self.output_channels}, {self.use_depth_prior}")

        return self.model


    def set_optimizer(self, lr=0.000005):
        self.lr = lr
        param_groups = [
            {'params': [], 'lr': lr},
            {'params': [], 'lr': lr * 10.0}]

        for name, param in self.model.named_parameters(): 
            if 'pretrained' in name:
                if self.use_depth_prior and name == 'pretrained.patch_embed.proj.weight':
                    # This is randomly initialized and needs higher lr as well
                    param_groups[1]['params'].append(param)
                else:
                    # Normal lr
                    param_groups[0]['params'].append(param)
            else:
                # Increased lr
                param_groups[1]['params'].append(param)

        self.optimizer = AdamW(param_groups, lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
        return self.optimizer


    def get_dataset_loader(self, task, datasets_dir, dataset_list, index_list=None):
        datasets = []
        datasets_dir = Path(datasets_dir)
        for i, dataset_name in enumerate(dataset_list):
            dataset_dir = datasets_dir / dataset_name

            index_min, index_max = 0, -1
            if index_list != None:
                index_min, index_max = index_list[i][0], index_list[i][1]

            dataset = BlearnDataset(dataset_dir, task, self.size, index_min, index_max, self.depth_prior_dir)

            if len(dataset) > 0:
                datasets.append(dataset)

        # Convert to MultiDataset (also ok for one)
        dataset = MultiDatasetLoader(datasets, self.depth_min_max)
        if 'train' in task:
            loader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, drop_last=True, shuffle=True)

        else:
            loader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, drop_last=True)

        return loader, dataset

    def get_single_dataset_loader(self, dataset_dir, min_index=0, max_index=-1):
        dataset = BlearnDataset(dataset_dir, 'all', self.size, min_index, max_index, self.depth_prior_dir)
        return DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, drop_last=True)

    def update_best_result(self, results, nsamples):
        if nsamples:
            logger.info('==========================================================================================')
            logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
            logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / nsamples).item() for v in results.values()])))
            logger.info('==========================================================================================')

            for k in results.keys():
                if k in ['d1', 'd2', 'd3']:
                    self.previous_best[k] = max(self.previous_best[k], (results[k] / nsamples).item())
                else:
                    self.previous_best[k] = min(self.previous_best[k], (results[k] / nsamples).item())


    def prepare_sample(self, sample, random_flip=False):
        image = sample['image'].to(self.device)

        depth_prior = sample['depth_prior'].to(self.device)
        if self.use_depth_prior:
            depth_prior = depth_prior.unsqueeze(1)
            image = torch.cat((image, depth_prior), axis=1)

        if 'depth' in sample.keys() and 'valid_mask' in sample.keys():
            depth_target = sample['depth'].to(self.device)
            valid_mask = sample['valid_mask'].to(self.device)
            mask = (valid_mask == 1) & (depth_target >= self.min_depth) & (depth_target <= self.max_depth)

            if random_flip:
                image, depth_target, mask = randomly_flip(image, depth_target, mask)

        else:
            depth_target, mask = None, None


        return image, depth_prior, depth_target, mask


    def train_epoch(self, epoch, train_loader):
        print_epoch_summary(epoch, self.epochs, self.previous_best)
        self.model.train()
        total_loss = 0
        total_iters = self.epochs * len(train_loader)

        for i, sample in enumerate(train_loader):
            self.optimizer.zero_grad()
            image, _, depth_target, mask = self.prepare_sample(sample, random_flip=True)

            prediction = self.model(image)
            depth_prediction = get_depth_from_prediction(prediction, image)
            if depth_prediction is not None and mask.sum() > 0:
                loss = self.criterion(depth_prediction, depth_target, mask)
                if loss is not None:
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

            # Adaptive learning rate based on iterations
            iters = epoch * len(train_loader) + i
            lr = self.lr * (1 - iters / total_iters) ** 0.9
            self.optimizer.param_groups[0]["lr"] = lr
            self.optimizer.param_groups[1]["lr"] = lr * 10.0

            if i % 100 == 0:
                logger.info('Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(train_loader), self.optimizer.param_groups[0]['lr'], total_loss/(i + 1.0)))


    def validate_epoch(self, epoch, val_loader, iteration_callback=None):
        self.model.eval()

        self.results, self.results_per_sample, nsamples = get_empty_results(self.device)
        for i, sample in enumerate(val_loader):
            image, _, depth_target, mask = self.prepare_sample(sample, random_flip=False)

            # TODO: Maybe not hardcode 10 here?
            if mask is None or mask.sum() > 10:
                with torch.no_grad():
                    prediction = self.model(image)
                    prediction = interpolate_shape(prediction, depth_target)
                    depth_prediction = get_depth_from_prediction(prediction, image)

                    # TODO: Expand on this interface
                    if iteration_callback is not None:
                        iteration_callback(int(sample['index']), depth_prediction)

                    if mask is not None:
                        current_results = eval_depth(depth_prediction[mask], depth_target[mask])
                        if current_results is not None:
                            for k in self.results.keys():
                                self.results[k] += current_results[k]
                                self.results_per_sample[k].append(current_results[k])
                            nsamples += 1

                if i % 10 == 0:
                    abs_rel = (self.results["abs_rel"]/nsamples).item()
                    logger.info(f'Iter: {i}/{len(val_loader)}, Absrel: {abs_rel:.3f}')

        self.update_best_result(self.results, nsamples)
        self.save_checkpoint(epoch)


    def save_checkpoint(self, epoch):
        if self.results_path is not None and len(str(self.results_path)) > 1:
            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': self.previous_best,
            }
            # TODO: How to check properly if current path is not .
            torch.save(checkpoint, self.results_path / f'latest_{epoch}.pth')
