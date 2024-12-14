######################################################################
#
# Copyright (c) 2024 ETHZ Autonomous Systems Lab. All rights reserved.
#
######################################################################

import logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def get_depth_from_prediction(prediction, input):
    depth = prediction[:, 0, :, :]
    if prediction.shape[1] > 1 and input.shape[1] > 3:
        prior = input[:, 3, :, :]
        prior_mask = prior > 0
        if prior_mask.sum():
            prior_mean = prior[prior_mask].mean()
            confidence = prediction[:, 1, :, :]
            depth = depth * confidence + prior_mean * (1.0 - confidence)
        else:
            depth = None

    return depth

def get_confidence_from_prediction(prediction):
    return prediction[:, 1:, :, :]

def interpolate_shape(prediction, target, mode='bilinear'):
    # TODO: I don't like this default output size specification
    interp_shape = (480, 640)
    if target is not None:
        if len(target.shape) > 2:
            target = target.squeeze()

        interp_shape = (target.shape[0], target.shape[1])

    if mode == 'nearest':
        if len(prediction.shape) < 4:
            prediction = prediction.unsqueeze(0)
        prediction = F.interpolate(prediction, interp_shape, mode=mode)
    else:
        prediction = F.interpolate(prediction, interp_shape, mode=mode, align_corners=True)

    return prediction
