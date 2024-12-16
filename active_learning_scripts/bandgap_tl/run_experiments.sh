#!/bin/bash

# Base command with common parameters
BASE_CMD="python3.11 pbnn.py --activation silu --exploration_steps 200 --pretrain-epochs 500 --pretrain-lr 5e-3 --pretrain-batch-size 32 --hidden-dims 32 16 8 8"

# Different prior sigma values to test

echo "Running experiment with sigma = 0.1..."
$BASE_CMD --priors-sigma 0.1 --probabilistic-layer-names Dense2 Dense3 Dense4

echo "Running experiment with sigma = 0.5..."
$BASE_CMD --priors-sigma 0.5 --probabilistic-layer-names Dense2 Dense3 Dense4

echo "Running experiment with sigma = 1.0..."
$BASE_CMD --priors-sigma 1.0 --probabilistic-layer-names Dense2 Dense3 Dense4