from typing import Dict, Callable, Optional, List
import jax.numpy as jnp
import jax.random as jra
import numpyro
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta

from .dkl import DKL
from .priors import GPPriors
from .utils import put_on_device


kernel_fn_type = Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray],  jnp.ndarray]


class VIDKL(DKL):

    """
    Variational Inference-based Deep Kernel Learning
    """

    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 base_kernel: kernel_fn_type,
                 priors: Optional[GPPriors] = None,
                 hidden_dim: List[int] = None,
                 activation: str = 'tanh',
                 jitter: float = 1e-6
                 ) -> None:
        super(VIDKL, self).__init__(
            input_dim, latent_dim, base_kernel,
            priors, hidden_dim, activation, jitter)

    def fit(self, X: jnp.ndarray, y: jnp.ndarray,
            num_steps: int = 1000, step_size: float = 5e-3,
            progress_bar: bool = True,
            rng_key: jnp.array = None,
            **kwargs: float
            ) -> None:
        """
        Run variational inference to learn DKL (hyper)parameters

        Args:
            rng_key: random number generator key
            X: 2D feature vector with *(number of points, number of features)* dimensions
            y: 1D target vector with *(n,)* dimensions
            num_steps: number of SVI steps
            step_size: step size schedule for Adam optimizer
            progress_bar: show progress bar
            print_summary: print summary at the end of training
            rng_key: random number generator key
        """
        key = rng_key if rng_key is not None else jra.PRNGKey(0)
        X, y = self.set_data(X, y)
        X, y = put_on_device(X, y)
        self.X_train = X
        self.y_train = y

        optim = numpyro.optim.Adam(step_size=step_size, b1=0.5)
        self.svi = SVI(
            self.model,
            guide=AutoDelta(self.model),
            optim=optim,
            loss=Trace_ELBO(),
            X=X,
            y=y,
            **kwargs
        )

        params = self.svi.run(
            key, num_steps, progress_bar=progress_bar)[0]

        self.params = self.svi.guide.median(params)

    def get_samples(self, **kwargs):
        return {k: v[None] for (k, v) in self.params.items()}

    def print_summary(self, print_nn_weights: bool = False) -> None:
        samples = self.get_samples(1)
        if print_nn_weights:
            numpyro.diagnostics.print_summary(samples)
        else:
            list_of_keys = ["k_scale", "k_length", "noise"]
        numpyro.diagnostics.print_summary(
            {k: v for (k, v) in samples.items() if k in list_of_keys})
