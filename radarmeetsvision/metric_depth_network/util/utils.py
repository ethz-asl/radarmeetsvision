######################################################################
#
# Copyright (c) 2024 ETHZ Autonomous Systems Lab. All rights reserved.
#
######################################################################

import logging
import random
import torch

from datetime import datetime

logger = logging.getLogger(__name__)

def print_epoch_summary(epoch, epochs, result_dict):
    d1 = result_dict['d1']
    d2 = result_dict['d2']
    d3 = result_dict['d3']
    abs_rel = result_dict['abs_rel']
    sq_rel = result_dict['sq_rel']
    rmse = result_dict['rmse']
    rmse_log = result_dict['rmse_log']
    log10 = result_dict['log10']
    silog = result_dict['silog']
    logger.info(f'===========> Epoch: {epoch}/{epochs}, d1: {d1:.3f}, d2: {d2:.3f}, d3: {d3:.3f}')
    logger.info(f'===========> Epoch: {epoch}/{epochs}, abs_rel: {abs_rel:.3f}, sq_rel: {sq_rel:.3f},'
                f'rmse: {rmse:.3f}, rmse_log: {rmse_log:.3f}, log10: {log10:.3f}, silog: {silog:.3f}')

def get_empty_results(device):
    results = {
        'd1': 0.0,
        'd2': 0.0,
        'd3': 0.0,
        'abs_rel': 0.0,
        'sq_rel': 0.0,
        'rmse': 0.0,
        'rmse_log': 0.0,
        'log10': 0.0,
        'silog': 0.0
    }
    results_per_sample = {
        'd1': [],
        'd2': [],
        'd3': [],
        'abs_rel': [],
        'sq_rel': [],
        'rmse': [],
        'rmse_log': [],
        'log10': [],
        'silog': []
    }
    nsamples = torch.tensor([0.0]).to(device)
    return results, results_per_sample, nsamples

def randomly_flip(img, target, valid_mask):
    if random.random() < 0.5:
        img = img.flip(-1)
        target = target.flip(-1)
        valid_mask = valid_mask.flip(-1)
    return img, target, valid_mask
