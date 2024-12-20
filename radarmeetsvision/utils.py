######################################################################
#
# Copyright (c) 2024 ETHZ Autonomous Systems Lab. All rights reserved.
#
######################################################################

import colorlog
import json
import logging
import torch
from datetime import datetime

def get_device(min_memory_gb=8):
    device_str = 'cpu'
    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        total_memory_gb = device.total_memory / (1024 ** 3)
        if total_memory_gb > min_memory_gb:
            device_str = 'cuda'
    return device_str

def setup_global_logger(output_dir=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set to INFO, DEBUG, etc.

    # Remove any existing handlers (to prevent duplicate logs if called multiple times)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create console handler with colorized output using `colorlog`
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a colorized formatter for the console
    color_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )
    console_handler.setFormatter(color_formatter)
    logger.addHandler(console_handler)

    if output_dir is not None and len(output_dir) > 0:
        log_file = datetime.now().strftime(f"{output_dir}/log_%Y-%m-%d_%H-%M-%S") + '.txt'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config
