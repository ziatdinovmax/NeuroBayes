from typing import Dict, Type, Any
import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
import optax
from functools import partial
from tqdm import tqdm

from .utils import split_in_batches


class TrainState(train_state.TrainState):
    batch_stats: Any

class DeterministicNN:

    """
    Class for training and predicting with deterministic (as in non-stochastic)
    neural network.

    Args:
        architecture: a Flax model
        input_dim: input dimensions (aka number of features)
        loss: type of loss, 'homoskedastic' (default) or 'heteroskedastic'
        learning_rate: SGD learning rate
        map: Uses maximum a posteriori approximation with a zero-centered Gaussian prior
        sigma: Standard deviation for a Gaussian prior
        swa_epochs: Number of epochs for stochastic weight averaging at the end of training trajectory (defautls to 10)
    """

    def __init__(self,
                 architecture: Type[flax.linen.Module],
                 input_dim: int,
                 loss: str = 'homoskedastic',
                 learning_rate: float = 0.01,
                 map: bool = True,
                 sigma: float = 1.0,
                 swa_epochs: int = 10) -> None:
        
        self.model = architecture

        if loss not in ['homoskedastic', 'heteroskedastic']:
            raise ValueError("Select between 'homoskedastic' or 'heteroskedastic' loss")
        self.loss = loss
        key = jax.random.PRNGKey(0)
        params = self.model.init(key, jnp.ones((1, input_dim)))['params']
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optax.adam(learning_rate),
            batch_stats=None
        )
        self.map = map
        self.sigma = sigma
        self.average_last_n_weights = swa_epochs
        self.params_history = []

    def mse_loss(self, params: Dict, inputs: jnp.ndarray,
                 targets: jnp.ndarray) -> jnp.ndarray:
        predictions = self.model.apply({'params': params}, inputs)
        return jnp.mean((predictions - targets) ** 2)
    
    def heteroskedastic_loss(self, params: Dict, inputs: jnp.ndarray,
                             targets: jnp.ndarray) -> jnp.ndarray:
        y_pred, y_var = self.model.apply({'params': params}, inputs)
        return jnp.mean(0.5 * jnp.log(y_var) + 0.5 * (targets - y_pred)**2 / y_var)
    
    def gaussian_prior(self, params: Dict) -> jnp.ndarray:
        l2_norm = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
        return l2_norm / (2 * self.sigma**2)  # Regularization term
    
    def total_loss(self, params: Dict, inputs: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        # Compute the base loss
        loss_fn = self.mse_loss if self.loss == 'homoskedastic' else self.heteroskedastic_loss
        loss = loss_fn(params, inputs, targets)
        # Optionally add Gaussian prior to the loss
        if self.map:
            prior_loss = self.gaussian_prior(params) / len(inputs)
            loss += prior_loss
        return loss

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, state, inputs, targets):
        loss, grads = jax.value_and_grad(self.total_loss)(state.params, inputs, targets)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def train(self, X_train: jnp.ndarray, y_train: jnp.ndarray, epochs: int, batch_size: int = None) -> None:
        X_train, y_train = self.set_data(X_train, y_train)
        
        if batch_size is None or batch_size >= len(X_train):
            batch_size = len(X_train)
        
        X_batches = split_in_batches(X_train, batch_size)
        y_batches = split_in_batches(y_train, batch_size)
        num_batches = len(X_batches)
        
        with tqdm(total=epochs, desc="Training Progress", leave=True, disable=None) as pbar:  # Progress bar tracks epochs now, but only with TTY
            for epoch in range(epochs):
                epoch_loss = 0.0
                for i, (X_batch, y_batch) in enumerate(zip(X_batches, y_batches)):
                    self.state, batch_loss = self.train_step(self.state, X_batch, y_batch)
                    epoch_loss += batch_loss

                # Start storing parameters in the last n epochs
                if epochs - epoch <= self.average_last_n_weights:
                    self._store_params(self.state.params)
                
                avg_epoch_loss = epoch_loss / num_batches
                pbar.set_postfix_str(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_epoch_loss:.4f}")
                pbar.update(1)

        if self.params_history:  # Ensure there is something to average
            self.state = self.state.replace(params=self.average_params())

    @partial(jax.jit, static_argnums=(0,))
    def _predict(self, state, X):
        return state.apply_fn({'params': state.params}, X)

    def predict(self, X):
        X = self.set_data(X)
        return self._predict(self.state, X)
    
    def set_data(self, X: jnp.ndarray, y: jnp.ndarray = None)  -> jnp.ndarray:
        X = X if X.ndim > 1 else X[:, None]
        if y is not None:
            y = y[:, None] if y.ndim < 2 else y
            return X, y
        return X
    
    def _store_params(self, params: Dict) -> None:
        self.params_history.append(params)

    def average_params(self) -> Dict:
        if not self.params_history:
            return self.state.params
        # Compute the element-wise average of all stored parameters
        avg_params = jax.tree_util.tree_map(
            lambda *param_trees: jnp.mean(jnp.stack(param_trees), axis=0),
            *self.params_history
        )
        return avg_params
