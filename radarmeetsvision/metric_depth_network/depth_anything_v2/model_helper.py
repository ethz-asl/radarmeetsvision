######################################################################
#
# Copyright (c) 2024 ETHZ Autonomous Systems Lab. All rights reserved.
#
######################################################################

import logging
import torch
from .dpt import DepthAnythingV2

logger = logging.getLogger(__name__)

model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

def get_model(pretrained_from, use_depth_prior, encoder, max_depth, output_channels):
    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth, 'use_depth_prior': use_depth_prior, 'output_channels': output_channels})

    state_dict = model.state_dict()
    if pretrained_from:
        logger.info("Loading pretrained model")
        state_dict = torch.load(pretrained_from, map_location='cpu')
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']

    if use_depth_prior:
        logger.info("Using depth prior")

        weights = state_dict['pretrained.patch_embed.proj.weight']
        if weights.shape[1] < 4:
            logger.info("Appending weights to network for input channels")
            new_channel_weights = torch.randn(weights.shape[0], 1, weights.shape[2], weights.shape[3]) * 0.01
            new_weights = torch.cat((weights, new_channel_weights), dim=1)
            state_dict['pretrained.patch_embed.proj.weight'] = new_weights

    if output_channels > 1:
        logger.info(f"Using {output_channels} output channels")
        weights = state_dict['depth_head.scratch.output_conv2.2.weight']
        if weights.shape[0] < output_channels:
            logger.info("Appending weights to pretrained network for output channels")
            new_channel_weights = torch.randn(output_channels-1, state_dict.shape[1], state_dict.shape[2], state_dict.shape[3]) * 0.01
            new_weights = torch.cat((state_dict, new_channel_weights), dim=0)
            state_dict['depth_head.scratch.output_conv2.2.weight'] = new_weights

            weights = state_dict['depth_head.scratch.output_conv2.2.bias']
            new_bias = torch.randn(output_channels-1) * 0.01
            new_bias_weights = torch.cat((state_dict, new_bias), dim=0)
            state_dict['depth_head.scratch.output_conv2.2.bias'] = new_bias_weights

    model.load_state_dict(state_dict, strict=False)

    return model
