DATASET_DIR='/home/asl/Storage/publish/validation/'
NETWORK_DIR='/home/asl/Storage/training_for_publication/'
CONFIG_DIR='scripts/evaluation/'
OUTPUT_DIR='/home/asl/Storage/output'

# Radar
mkdir $OUTPUT_DIR/radar_b
mkdir $OUTPUT_DIR/radar_s
python3 scripts/evaluation/find_best.py --dataset $DATASET_DIR --output $OUTPUT_DIR/radar_b --network $NETWORK_DIR/training_base/radar --config $CONFIG_DIR/config_best_radar_b.json &
python3 scripts/evaluation/find_best.py --dataset $DATASET_DIR --output $OUTPUT_DIR/radar_s --network $NETWORK_DIR/training_small2/radar --config $CONFIG_DIR/config_best_radar_s.json &

# Metric RGB
mkdir $OUTPUT_DIR/metric_b
mkdir $OUTPUT_DIR/metric_s
python3 scripts/evaluation/find_best.py --dataset $DATASET_DIR --output $OUTPUT_DIR/metric_b --network $NETWORK_DIR/training_base2/metric_rgb --config $CONFIG_DIR/config_best_metric_b.json &
python3 scripts/evaluation/find_best.py --dataset $DATASET_DIR --output $OUTPUT_DIR/metric_s --network $NETWORK_DIR/training_small/metric_rgb --config $CONFIG_DIR/config_best_metric_s.json &

# Relative
mkdir $OUTPUT_DIR/relative_b
mkdir $OUTPUT_DIR/relative_s
python3 scripts/evaluation/find_best.py --dataset $DATASET_DIR --output $OUTPUT_DIR/relative_b --network $NETWORK_DIR/training_base/relative_rgb --config $CONFIG_DIR/config_best_relative_b.json &
python3 scripts/evaluation/find_best.py --dataset $DATASET_DIR --output $OUTPUT_DIR/relative_s --network $NETWORK_DIR/training_small/relative_rgb/ --config $CONFIG_DIR/config_best_relative_s.json &
