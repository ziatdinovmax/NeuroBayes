from typing import Dict, Tuple, Callable, Optional, Union, Type
import jax.numpy as jnp
import flax
import numpyro
import numpyro.distributions as dist

from .gp import GP
from .deterministic_nn import DeterministicNN
from .priors import GPPriors

kernel_fn_type = Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray],  jnp.ndarray]


class DMFGP(GP):
    """
    Gaussian Process with Deep Mean Function
    """

    def __init__(self,
                 input_dim: int, kernel: kernel_fn_type,
                 nn: Type[flax.linen.Module],
                 nn_weights: Optional[Dict[str, jnp.ndarray]] = None,
                 priors: Optional[GPPriors] = None,
                 jitter: float = 1e-6):
        super().__init__(input_dim, kernel, priors, jitter)
        self.nn = nn
        self.nn_weights = nn_weights

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None) -> None:
        """DMFGP model with inputs X and targets y"""
        # Deep mean function
        f_loc = self.nn.apply({'params': self.nn_weights}, X).squeeze()
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
            sgd_epochs: Optional[int] = None,
            sgd_lr: Optional[float] = 0.01,
            sgd_batch_size: Optional[int] = None,
            sgd_wa_epochs: Optional[int] = 10,
            map_sigma: float = 1.0,
            progress_bar: bool = True,
            print_summary: bool = True,
            device: str = None,
            rng_key: jnp.array = None,
            extra_fields: Optional[Tuple[str]] = ()
            ) -> None:
        """
        Infers the DMFGP parameters

        Args:
            X: 2D feature vector
            y: 1D target vector
            num_warmup: number of HMC warmup states
            num_samples: number of HMC samples
            num_chains: number of HMC chains
            chain_method: choose between 'sequential', 'vectorized', and 'parallel'
            sgd_swa_epochs:
                number of SGD training epochs for NN (if trained weights are not provided at the initialization stage)
            sgd_lr:
                SGD learning rate (if trained weights are not provided at the initialization stage)
            sgd_batch_size:
                Batch size for SGD training (if trained weights are not provided at the initialization stage).
                Defaults to None, meaning that an entire dataset is passed through an NN.
            sgd_wa_epochs: Number of epochs for stochastic weight averaging at the end of SGD training trajectory for NN (defautls to 10)
            map_sigma: sigma in gaussian prior for regularized SGD training of NN
            progress_bar: show progress bar
            print_summary: print summary at the end of sampling
            device:
                The device (e.g. "cpu" or "gpu") perform computation on ('cpu', 'gpu'). If None, computation
                is performed on the JAX default device.
            rng_key: random number generator key
            extra_fields:
                Extra fields (e.g. 'accept_prob') to collect during the HMC run.
                The extra fields are accessible from model.mcmc.get_extra_fields() after model training.
        """
        if not self.nn_weights:
            print("Training deterministic NN...")
            det_nn = DeterministicNN(
                self.nn, self.kernel_dim,
                learning_rate=sgd_lr, swa_epochs=sgd_wa_epochs, sigma=map_sigma)
            det_nn.train(X, y, 500 if sgd_epochs is None else sgd_epochs, sgd_batch_size)
            self.nn_weights = det_nn.state.params
        super().fit(
            X, y, num_warmup, num_samples, num_chains, chain_method, progress_bar, print_summary, device, rng_key, extra_fields)
       
    def compute_gp_posterior(self, X_new: jnp.ndarray,
                             X_train: jnp.ndarray, y_train: jnp.ndarray,
                             params: Dict[str, jnp.ndarray],
                             noiseless: bool = True,
                             ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns mean and covariance of multivariate normal
        posterior for a single sample of trained DMFGP parameters
        """
        noise = params["noise"]
        noise_p = noise * (1 - jnp.array(noiseless, int))
        # Apply deep mean function
        y_residual = y_train - self.nn.apply({'params': self.nn_weights}, X_train).squeeze()
        # compute kernel matrices for train and new/test data
        k_XX = self.kernel(X_train, X_train, params, noise, self.jitter)
        k_pp = self.kernel(X_new, X_new, params, noise_p, self.jitter)
        k_pX = self.kernel(X_new, X_train, params)
        # compute predictive mean covariance
        K_xx_inv = jnp.linalg.inv(k_XX)
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, y_residual))
        cov = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
        # Apply deep mean function
        mean += self.nn.apply({'params': self.nn_weights}, X_new).squeeze()
        return mean, cov