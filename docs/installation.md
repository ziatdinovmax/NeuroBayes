---
layout: default
title: Installation
nav_order: 2
---

# Installation

NeuroBayes requires Python 3.9 or later and is built on JAX, NumPyro, and Flax. For the latest version or to contribute to the project, you can install from source:

```bash
git clone https://github.com/ziatdinovmax/NeuroBayes.git
cd NeuroBayes
pip install -e .
```

or, simply:
```bash
pip install git+https://github.com/ziatdinovmax/NeuroBayes.git
```

## GPU Support

For GPU acceleration, make sure to install the appropriate version of JAX and JAXlib with CUDA support. Follow the [official JAX installation guide](https://github.com/google/jax#installation) for GPU support.
