#!/bin/bash

# Install radarmeetsvision
pip install -e .

RESULTS_DIR=tests/resources/results
if [ -d $RESULTS_DIR ]; then
    rm -r $RESULTS_DIR
fi
rm -rf tests/resources/*.pkl

# Run the evaluation script
python3 scripts/evaluation/evaluate_networks.py --dataset tests/resources --config tests/resources/test_evaluation.json --output tests/resources --network tests/resources

TEX_FILE="tests/resources/results/results_table0.tex"
if [[ -f "$TEX_FILE" && -s "$TEX_FILE" ]]; then
    echo "Evaluation script successful, .tex table exists and is not empty."
else
    echo "Evaluation script failed, .tex table does not exist or is empty."
    exit 1
fi
