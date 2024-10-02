#!/bin/bash

checkpoints=tests/resources
save_path=tests/resources
datasets=tests/resources

config_path=tests/resources
config_radar=$config_path/test_train_radar.json
config_metric=$config_path/test_train_metric.json
config_relative=$config_path/test_train_relative.json

# Install radarmeetsvision
pip install -e .

# RADAR TRAINING (depth prior + 2 output channels)
python3 scripts/train.py \
--checkpoints $checkpoints \
--config $config_radar \
--datasets $datasets \
--results ""

if [ $? -eq 0 ]; then
    echo "Radar training script successful"
else
    echo "Training script failed"
    exit 1
fi

# RGB TRAINING (no depth prior + 1 output channel)
python3 scripts/train.py \
--checkpoints $checkpoints \
--config $config_metric \
--datasets $datasets \
--results ""

if [ $? -eq 0 ]; then
    echo "RGB training script successful"
else
    echo "Training script failed"
    exit 1
fi

# Relative RGB TRAINING (no depth prior + 1 output channel)
python3 scripts/train.py \
--checkpoints $checkpoints \
--config $config_relative \
--datasets $datasets \
--results ""

if [ $? -eq 0 ]; then
    echo "Relative RGB training script successful"
else
    echo "Training script failed"
    exit 1
fi
