#!/bin/bash

# Base command with common parameters
BASE_CMD="python3.11 dkl.py"

# Run experiment with tanh
echo "Running experiment with tanh..."
$BASE_CMD --activation tanh

# Run experiment with silu ('swish')
echo "Running experiment with silu..."
$BASE_CMD --activation silu

# Run experiment with relu
echo "Running experiment with relu..."
$BASE_CMD --activation relu

echo "All experiments completed!"