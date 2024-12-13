#!/bin/bash

# Base command with common parameters
BASE_CMD="python3.11 pbnn.py --activation silu --exploration_steps 200 --sgd_epochs 2000 --sgd_lr 5e-3"


# # Run experiment with Dense3 and Dense4
# echo "Running experiment with Dense3, Dense4..."
# $BASE_CMD --probabilistic-layer-names Dense3 Dense4

# # Run experiment with Dense0 and Dense4
# echo "Running experiment with Dense0, Dense4..."
# $BASE_CMD --probabilistic-layer-names Dense0 Dense4

# Run experiment with Dense1 and Dense4
echo "Running experiment with Dense1, Dense4..."
$BASE_CMD --probabilistic-layer-names Dense1 Dense4

# Run experiment with Dense2 and Dense4
echo "Running experiment with Dense2, Dense4..."
$BASE_CMD --probabilistic-layer-names Dense2 Dense4

echo "All experiments completed!"