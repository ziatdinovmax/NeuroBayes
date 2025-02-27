---
layout: default
title: Diagnostics
parent: Utils
grand_parent: API Reference
---

<a id="neurobayes.utils.diagnostics"></a>

# neurobayes.utils.diagnostics

<a id="neurobayes.utils.diagnostics.MCMCDiagnostics"></a>

## MCMCDiagnostics Objects

```python
class MCMCDiagnostics()
```

Lightweight diagnostics for BNN MCMC samples with layer analysis

<a id="neurobayes.utils.diagnostics.MCMCDiagnostics.analyze_samples"></a>

#### analyze\_samples

```python
def analyze_samples(mcmc_samples: Dict[str, jnp.ndarray]) -> Dict
```

Analyze MCMC samples layer by layer using NumPyro's split_gelman_rubin

<a id="neurobayes.utils.diagnostics.MCMCDiagnostics.print_summary"></a>

#### print\_summary

```python
def print_summary()
```

Print formatted diagnostic summary

<a id="neurobayes.utils.diagnostics.MCMCDiagnostics.run_diagnostics"></a>

#### run\_diagnostics

```python
def run_diagnostics(mcmc_samples: Dict[str, jnp.ndarray]) -> Dict
```

Analyze and print MCMC diagnostics summary in one go


