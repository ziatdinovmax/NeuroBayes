---
layout: default
title: Priors
parent: Utils
grand_parent: API Reference
---

<a id="neurobayes.utils.priors"></a>

# neurobayes.utils.priors

<a id="neurobayes.utils.priors.place_normal_prior"></a>

#### place\_normal\_prior

```python
def place_normal_prior(param_name: str, loc: float = 0.0, scale: float = 1.0)
```

Samples a value from a normal distribution with the specified mean (loc) and standard deviation (scale),
and assigns it to a named random variable in the probabilistic model. Can be useful for defining prior mean functions
in structured Gaussian processes.

<a id="neurobayes.utils.priors.place_lognormal_prior"></a>

#### place\_lognormal\_prior

```python
def place_lognormal_prior(param_name: str,
                          loc: float = 0.0,
                          scale: float = 1.0)
```

Samples a value from a log-normal distribution with the specified mean (loc) and standard deviation (scale),
and assigns it to a named random variable in the probabilistic model. Can be useful for defining prior mean functions
in structured Gaussian processes.

<a id="neurobayes.utils.priors.place_halfnormal_prior"></a>

#### place\_halfnormal\_prior

```python
def place_halfnormal_prior(param_name: str, scale: float = 1.0)
```

Samples a value from a half-normal distribution with the specified standard deviation (scale),
and assigns it to a named random variable in the probabilistic model. Can be useful for defining prior mean functions
in structured Gaussian processes.

<a id="neurobayes.utils.priors.place_uniform_prior"></a>

#### place\_uniform\_prior

```python
def place_uniform_prior(param_name: str,
                        low: float = None,
                        high: float = None,
                        X: jnp.ndarray = None)
```

Samples a value from a uniform distribution with the specified low and high values,
and assigns it to a named random variable in the probabilistic model. Can be useful for defining prior mean functions
in structured Gaussian processes.

<a id="neurobayes.utils.priors.place_gamma_prior"></a>

#### place\_gamma\_prior

```python
def place_gamma_prior(param_name: str,
                      c: float = None,
                      r: float = None,
                      X: jnp.ndarray = None)
```

Samples a value from a uniform distribution with the specified concentration (c) and rate (r) values,
and assigns it to a named random variable in the probabilistic model. Can be useful for defining prior mean functions
in structured Gaussian processes.

<a id="neurobayes.utils.priors.normal_dist"></a>

#### normal\_dist

```python
def normal_dist(loc: float = None,
                scale: float = None) -> numpyro.distributions.Distribution
```

Generate a Normal distribution based on provided center (loc) and standard deviation (scale) parameters.
If neither are provided, uses 0 and 1 by default. It can be used to pass custom priors to GP models.

<a id="neurobayes.utils.priors.lognormal_dist"></a>

#### lognormal\_dist

```python
def lognormal_dist(loc: float = None,
                   scale: float = None) -> numpyro.distributions.Distribution
```

Generate a LogNormal distribution based on provided center (loc) and standard deviation (scale) parameters.
If neither are provided, uses 0 and 1 by default. It can be used to pass custom priors to GP models.

<a id="neurobayes.utils.priors.halfnormal_dist"></a>

#### halfnormal\_dist

```python
def halfnormal_dist(scale: float = None) -> numpyro.distributions.Distribution
```

Generate a half-normal distribution based on provided standard deviation (scale).
If none is provided, uses 1.0 by default. It can be used to pass custom priors to GP models.

<a id="neurobayes.utils.priors.gamma_dist"></a>

#### gamma\_dist

```python
def gamma_dist(
        c: float = None,
        r: float = None,
        input_vec: jnp.ndarray = None) -> numpyro.distributions.Distribution
```

Generate a Gamma distribution based on provided shape (c) and rate (r) parameters. If the shape (c) is not provided,
it attempts to infer it using the range of the input vector divided by 2. The rate parameter defaults to 1.0 if not provided.
It can be used to pass custom priors to GP models.

<a id="neurobayes.utils.priors.uniform_dist"></a>

#### uniform\_dist

```python
def uniform_dist(
        low: float = None,
        high: float = None,
        input_vec: jnp.ndarray = None) -> numpyro.distributions.Distribution
```

Generate a Uniform distribution based on provided low and high bounds. If one of the bounds is not provided,
it attempts to infer the missing bound(s) using the minimum or maximum value from the input vector.
It can be used to pass custom priors to GP models.

:

    Assign custom prior to kernel lengthscale during GP model initialization

    >>> model = gpax.ExactGP(input_dm, kernel, lengthscale_prior_dist=gpax.priors.uniform_dist(1, 3))

    Train as usual

    >>> model.fit(rng_key, X, y)

<a id="neurobayes.utils.priors.auto_priors"></a>

#### auto\_priors

```python
def auto_priors(func: Callable,
                params_begin_with: int,
                dist_type: str = 'normal',
                loc: float = 0.0,
                scale: float = 1.0) -> Callable
```

Generates a function that, when invoked, samples from normal or log-normal distributions
for each parameter of the given deterministic function, except the first one.

**Arguments**:

- `func` _Callable_ - The deterministic function for which to set normal or log-normal priors.
- `params_begin_with` _int_ - Parameters to account for start from this number.
- `loc` _float, optional_ - Mean of the normal or log-normal distribution. Defaults to 0.0.
- `scale` _float, optional_ - Standard deviation of the normal or log-normal distribution. Defaults to 1.0.
  

**Returns**:

  A function that, when invoked, returns a dictionary of sampled values
  from normal or log-normal distributions for each parameter of the original function.

<a id="neurobayes.utils.priors.auto_normal_priors"></a>

#### auto\_normal\_priors

```python
def auto_normal_priors(func: Callable,
                       loc: float = 0.0,
                       scale: float = 1.0) -> Callable
```

Places normal priors over function parameters.

**Arguments**:

- `func` _Callable_ - The deterministic function for which to set normal priors.
- `loc` _float, optional_ - Mean of the normal distribution. Defaults to 0.0.
- `scale` _float, optional_ - Standard deviation of the normal distribution. Defaults to 1.0.
  

**Returns**:

  A function that, when invoked, returns a dictionary of sampled values
  from normal distributions for each parameter of the original function.

<a id="neurobayes.utils.priors.auto_lognormal_priors"></a>

#### auto\_lognormal\_priors

```python
def auto_lognormal_priors(func: Callable,
                          loc: float = 0.0,
                          scale: float = 1.0) -> Callable
```

Places log-normal priors over function parameters.

**Arguments**:

- `func` _Callable_ - The deterministic function for which to set log-normal priors.
- `loc` _float, optional_ - Mean of the log-normal distribution. Defaults to 0.0.
- `scale` _float, optional_ - Standard deviation of the log-normal distribution. Defaults to 1.0.
  

**Returns**:

  A function that, when invoked, returns a dictionary of sampled values
  from log-normal distributions for each parameter of the original function.


