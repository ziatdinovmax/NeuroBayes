from typing import Dict, Tuple, Callable, Optional, Union
import jax
from jax import vmap
import jax.random as jra
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median

from .priors import GPPriors
from .utils import put_on_device, split_in_batches

kernel_fn_type = Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray],  jnp.ndarray]


class GP:
    """
    Fully Bayesian exact Gaussian process
    """

    def __init__(self,
                 input_dim: int, kernel: kernel_fn_type,
                 priors: Optional[GPPriors] = None,
                 jitter: float = 1e-6
                 ) -> None:
        if priors is None:
            priors = GPPriors()
        self.kernel = kernel
        self.kernel_dim = input_dim
        self.priors = priors
        self.jitter = jitter
        self.X_train = None
        self.y_train = None

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None) -> None:
        """GP probabilistic model with inputs X and targets y"""
        # Initialize mean function at zeros
        f_loc = jnp.zeros(X.shape[0])
        # Sample kernel parameters
        kernel_params = self.sample_kernel_params()
        # Sample observational noise variance
        noise = self.sample_noise()
        # Compute kernel
        k = self.kernel(X, X, kernel_params, noise, self.jitter)
        # Sample y according to the standard Gaussian process formula
        numpyro.sample(
            "y",
            dist.MultivariateNormal(loc=f_loc, covariance_matrix=k),
            obs=y,
        )

    def fit(self,
            X: jnp.ndarray,
            y: jnp.ndarray,
            num_warmup: int = 2000,
            num_samples: int = 2000,
            num_chains: int = 1,
            chain_method: str = "sequential",
            progress_bar: bool = True,
            print_summary: bool = True,
            device: str = None,
            rng_key: jnp.array = None
            ) -> None:
        """
        Run Hamiltonian Monter Carlo to infer the GP parameters

        Args:
            X: 2D feature vector
            y: 1D target vector
            num_warmup: number of HMC warmup states
            num_samples: number of HMC samples
            num_chains: number of HMC chains
            chain_method: 'sequential', 'parallel' or 'vectorized'
            progress_bar: show progress bar
            print_summary: print summary at the end of sampling
            device:
                The device (e.g. "cpu" or "gpu") perform computation on ('cpu', 'gpu'). If None, computation
                is performed on the JAX default device.
            rng_key: random number generator key
        """
        key = rng_key if rng_key is not None else jra.PRNGKey(0)
        X, y = self.set_data(X, y)
        X, y = put_on_device(device, X, y)
        self.X_train = X
        self.y_train = y

        init_strategy = init_to_median(num_samples=10)
        kernel = NUTS(self.model, init_strategy=init_strategy)
        self.mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=progress_bar,
            jit_model_args=False,
        )
        self.mcmc.run(key, X, y)

        if print_summary:
            self.print_summary()

    def sample_noise(self) -> jnp.ndarray:
        """
        Sample observational noise variance
        """
        return numpyro.sample("noise", self.priors.noise_prior)

    def sample_kernel_params(self) -> Dict[str, jnp.ndarray]:
        """
        Sample kernel parameters
        """
        with numpyro.plate("ard", self.kernel_dim):
            length = numpyro.sample("k_length", self.priors.lengthscale_prior)
        scale = numpyro.sample("k_scale", self.priors.output_scale_prior)
        return {"k_length": length, "k_scale": scale}

    def compute_gp_posterior(self, X_new: jnp.ndarray,
                             X_train: jnp.ndarray, y_train: jnp.ndarray,
                             params: Dict[str, jnp.ndarray],
                             noiseless: bool = True,
                             ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns mean and covariance of multivariate normal
        posterior for a single sample of trained GP parameters
        """
        noise = params["noise"]
        noise_p = noise * (1 - jnp.array(noiseless, int))
        # compute kernel matrices for train and new/test data
        k_XX = self.kernel(X_train, X_train, params, noise, self.jitter)
        k_pp = self.kernel(X_new, X_new, params, noise_p, self.jitter)
        k_pX = self.kernel(X_new, X_train, params)
        # compute predictive mean covariance
        K_xx_inv = jnp.linalg.inv(k_XX)
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, y_train))
        cov = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
        return mean, cov

    def predict(self,
                X_new: jnp.ndarray,
                noiseless: bool = True,
                device: str = None
                ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Make prediction at X_new points a trained GP model

        Args:
            X_new:
                New inputs with *(number of points, number of features)* dimensions
            noiseless:
                Noise-free prediction. It is set to False by default as new/unseen data is assumed
                to follow the same distribution as the training data. Hence, since we introduce a model noise
                by default for the training data, we also want to include that noise in our prediction.
            device:
                The device (e.g. "cpu" or "gpu") perform computation on ('cpu', 'gpu'). If None, computation
                is performed on the JAX default device.

        Returns:
            Posterior mean and variance
        """
        X_new = self.set_data(X_new)
        samples = self.get_samples(chain_dim=False)
        self.X, self.y, X_new, samples = put_on_device(
            device, self.X_train, self.y_train, X_new, samples)

        predictive = lambda p: self.compute_gp_posterior(
            X_new, self.X_train, self.y_train, p, noiseless)
        # Compute predictive mean and covariance for all HMC samples
        mu_all, cov_all = vmap(predictive)(samples)
        # Return predictive mean and variance averaged over the HMC samples
        return mu_all.mean(0), cov_all.mean(0).diagonal()

    def predict_in_batches(self, X_new: jnp.ndarray,
                           batch_size: int = 200,
                           noiseless: bool = True,
                           device: str = None
                           ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Make prediction in batches (to avoid memory overflow) 
        at X_new points a trained GP model
        """
        mean, var = [], []
        for x in split_in_batches(X_new, batch_size):
            mean_i, var_i = self.predict(x, noiseless, device)
            mean_i = jax.device_put(mean_i, jax.devices("cpu")[0])
            var_i = jax.device_put(var_i, jax.devices("cpu")[0])
            mean.append(mean_i)
            var.append(var_i)
        return jnp.concatenate(mean), jnp.concatenate(var)

    def draw_from_mvn(self,
                      rng_key: jnp.ndarray,
                      X_new: jnp.ndarray,
                      params: Dict[str, jnp.ndarray],
                      n_draws: int,
                      noiseless: bool
                      ) -> jnp.ndarray:
        """
        Draws predictive samples from multivariate normal distribution
        at X_new for a single estimate of GP posterior parameters
        """
        mu, cov = self.compute_gp_posterior(
            X_new, self.X_train, self.y_train, params, noiseless)
        mvn = dist.MultivariateNormal(mu, cov)
        return mvn.sample(rng_key, sample_shape=(n_draws,))

    def sample_from_posterior(self,
                              X_new: jnp.ndarray,
                              noiseless: bool = True,
                              n_draws: int = 100,
                              device: str = None,
                              rng_key: jnp.ndarray = None
                              ) -> jnp.ndarray:
        """
        Sample from the posterior predictive distribution at X_new

        Args:
            X_new:
                New inputs with *(number of points, number of features)* dimensions
            noiseless:
                Noise-free prediction. It is set to False by default as new/unseen data is assumed
                to follow the same distribution as the training data. Hence, since we introduce a model noise
                by default for the training data, we also want to include that noise in our prediction.
            n_draws:
                Number of MVN distribution samples to draw for each sample with GP parameters
            device:
                The device (e.g. "cpu" or "gpu") perform computation on ('cpu', 'gpu'). If None, computation
                is performed on the JAX default device.
            rng_key:
                Optional random number generator key

        Returns:
            A set of samples from the posterior predictive distribution.

        """
        key = rng_key if rng_key is not None else jra.PRNGKey(0)
        X_new = self.set_data(X_new)
        samples = self.get_samples(chain_dim=False)
        self.X, self.y, X_new, samples = put_on_device(
            device, self.X, self.y, X_new, samples)

        num_samples = len(next(iter(samples.values())))
        vmap_args = (jra.split(key, num_samples), samples)
        predictive = lambda p1, p2: self.draw_from_mvn(p1, X_new, p2, n_draws, noiseless)
        return vmap(predictive)(*vmap_args)

    def get_samples(self, chain_dim: bool = False) -> Dict[str, jnp.ndarray]:
        """Get posterior samples (after running the MCMC chains)"""
        return self.mcmc.get_samples(group_by_chain=chain_dim)

    def set_data(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None
                  ) -> Union[Tuple[jnp.ndarray], jnp.ndarray]:
        X = X if X.ndim > 1 else X[:, None]
        if y is not None:
            return X, y.squeeze()
        return X

    def print_summary(self) -> None:
        samples = self.get_samples(1)
        numpyro.diagnostics.print_summary(samples)

