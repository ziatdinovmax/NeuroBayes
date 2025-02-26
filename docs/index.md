---
layout: default
title: NeuroBayes Documentation
nav_order: 1
---

# NeuroBayes Documentation

NeuroBayes is a Python library that provides easy-to-use implementations of fully and partially Bayesian neural networks, along with standard Gaussian processes and deep kernel learning, offering a flexible framework for probabilistic modeling of black box functions.

## Table of Contents

- [Installation](installation.md)
- [Overview](#overview)
- [Main Components](#main-components)
- [Models](#models)
- [Networks](#networks)
- [Examples](#examples)

## Overview

Machine learning, at its core, is about approximating unknown functions – mapping inputs to outputs based on observed data. In scientific and engineering applications, this often means modeling complex relationships between structural or process parameters and target properties. Traditionally, Gaussian Processes (GPs) have been favored in these domains for their ability to provide robust uncertainty estimates. However, GPs struggle with systems featuring discontinuities and non-stationarities, common in physical science problems, as well as with high dimensional data. **NeuroBayes** bridges this gap by combining the flexibility and scalability of neural networks with the rigorous uncertainty quantification of Bayesian methods. This repository enables the use of full BNNs and partial BNNs with the No-U-Turn Sampler for intermediate size datasets, making it a powerful tool for a wide range of scientific and engineering applications.


## Models

NeuroBayes offers several types of models:

- [Bayesian Neural Networks](models/bnn.md) - Fully Bayesian treatment of neural networks
- [Partially Bayesian Neural Networks](models/partial_bnn.md) - Selective Bayesian treatment for efficiency
- [Heteroskedastic Models](models/heteroskedastic.md) - Models with input-dependent noise
- [Gaussian Processes](models/gp.md) - For GP regression with uncertainty estimation
- [Deep Kernel Learning](models/dkl.md) - Combining neural networks with Gaussian processes


## Networks

Currently, NeuroBayes supports multiple neural network architectures, with more architectures on their way:

- [MLP](networks/mlp.md) - Multi-layer perceptrons for tabular data
- [ConvNet](networks/convnet.md) - Convolutional networks for structured data
- [Transformer](networks/transformer.md) - Transformer architectures for sequential data