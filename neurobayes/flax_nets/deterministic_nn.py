from typing import Dict, Type, Any, Union, Tuple, Optional
import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
import optax
from functools import partial
from tqdm import tqdm
import numpy as np

from .train_utils import build_swa_schedule
from ..utils.utils import split_in_batches, monitor_dnn_loss


class TrainState(train_state.TrainState):
    batch_stats: Any

class DeterministicNN:
    def __init__(self,
                 architecture: Type[flax.linen.Module],
                 input_shape: Union[int, Tuple[int]],
                 loss: str = 'homoskedastic',
                 learning_rate: float = 0.01,
                 map: bool = True,
                 sigma: float = 1.0,
                 swa_protocol: Optional[Dict] = None) -> None:
        """
        Args:
            architecture: a Flax model
            input_shape: (n_samples, n_features) or (n_samples, *dims, n_channels)
            loss: type of loss, 'homoskedastic' (default) or 'heteroskedastic'
            learning_rate: Initial learning rate
            map: Uses maximum a posteriori approximation
            sigma: Standard deviation for Gaussian prior
            swa_protocol: SWA configuration dictionary
        """
        input_shape = (input_shape,) if isinstance(input_shape, int) else input_shape
        self.model = architecture

        if loss not in ['homoskedastic', 'heteroskedastic']:
            raise ValueError("Select between 'homoskedastic' or 'heteroskedastic' loss")
        
        # Initialize model and optimizer
        key = jax.random.PRNGKey(0)
        params = self.model.init(key, jnp.ones((1, *input_shape)))['params']
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optax.adam(learning_rate),
            batch_stats=None
        )
        
        # Store configurations
        self.loss_type = loss
        self.map = map
        self.sigma = sigma
        
        # Configure SWA
        protocol = {} if swa_protocol is None else swa_protocol.copy()
        if 'lr' not in protocol:
            protocol['lr'] = {'type': 'constant', 'value': learning_rate}
        elif 'value' not in protocol['lr']:
            protocol['lr']['value'] = learning_rate
            
        self.swa_schedule = build_swa_schedule(protocol)
        self.params_history = []

    def mse_loss(self, params: Dict, inputs: jnp.ndarray,
                 targets: jnp.ndarray) -> jnp.ndarray:
        """Compute mean squared error loss."""
        predictions = self.model.apply({'params': params}, inputs)
        return jnp.mean((predictions - targets) ** 2)
    
    def heteroskedastic_loss(self, params: Dict, inputs: jnp.ndarray,
                             targets: jnp.ndarray) -> jnp.ndarray:
        """Compute heteroskedastic loss with predicted variance."""
        y_pred, y_var = self.model.apply({'params': params}, inputs)
        return jnp.mean(0.5 * jnp.log(y_var) + 0.5 * (targets - y_pred)**2 / y_var)
    
    def gaussian_prior(self, params: Dict) -> jnp.ndarray:
        """Compute Gaussian prior loss term."""
        l2_norm = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
        return l2_norm / (2 * self.sigma**2)  # Regularization term
    
    def total_loss(self, params: Dict, inputs: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        """Compute total loss including base loss and prior if MAP is used."""
        # Compute the base loss
        loss_fn = self.mse_loss if self.loss_type == 'homoskedastic' else self.heteroskedastic_loss
        loss = loss_fn(params, inputs, targets)
        # Optionally add Gaussian prior to the loss
        if self.map:
            prior_loss = self.gaussian_prior(params) / len(inputs)
            loss += prior_loss
        return loss

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, state, batch, learning_rate):
        """Single training step with dynamic learning rate"""
        inputs, targets = batch
        loss, grads = jax.value_and_grad(self.total_loss)(state.params, inputs, targets)
        
        # Update with current learning rate
        if learning_rate != state.tx.learning_rate:
            state = state.replace(tx=optax.adam(learning_rate))
        state = state.apply_gradients(grads=grads)
        
        return state, loss

    def train(self, X_train: jnp.ndarray, y_train: jnp.ndarray, 
              epochs: int, batch_size: int = None) -> None:
        """Train the model with SWA support"""
        X_train, y_train = self._process_data(X_train, y_train)
        batch_size = len(X_train) if batch_size is None or batch_size >= len(X_train) else batch_size
        
        # Create batches
        X_batches = split_in_batches(X_train, batch_size)
        y_batches = split_in_batches(y_train, batch_size)
        num_batches = len(X_batches)
        
        with tqdm(total=epochs, desc="Training Progress") as pbar:
            for epoch in range(epochs):
                # Get current learning rate and collection decision
                lr, should_collect = self.swa_schedule(epoch, epochs)
                
                # Train for one epoch
                epoch_loss = 0.0
                for X_batch, y_batch in zip(X_batches, y_batches):
                    self.state, batch_loss = self.train_step(
                        self.state, (X_batch, y_batch), lr
                    )
                    epoch_loss += batch_loss
                
                # Collect weights if scheduled
                if should_collect:
                    self._store_params(self.state.params)
                
                # Update progress bar
                avg_epoch_loss = epoch_loss / num_batches
                pbar.set_postfix_str(
                    f"Loss: {avg_epoch_loss:.4f}, "
                    f"LR: {lr:.4f}"
                )
                pbar.update(1)
        
        # Average collected weights if any
        if self.params_history:
            self.state = self.state.replace(params=self._average_params())

    def _store_params(self, params: Dict) -> None:
        """Store parameters as numpy arrays for memory efficiency"""
        params_np = jax.tree_map(lambda x: np.array(x), params)
        self.params_history.append(params_np)

    def _average_params(self) -> Dict:
        """Compute SWA parameters"""
        if not self.params_history:
            return self.state.params
        
        # Convert back to jax arrays and compute average
        params_jax = [jax.tree_map(lambda x: jnp.array(x), p) 
                     for p in self.params_history]
        avg_params = jax.tree_map(
            lambda *trees: jnp.mean(jnp.stack(trees), axis=0),
            *params_jax
        )
        return avg_params

    def _process_data(self, X: jnp.ndarray, y: jnp.ndarray = None) -> jnp.ndarray:
        """Ensure data has correct shape"""
        X = X if X.ndim > 1 else X[:, None]
        if y is not None:
            y = y[:, None] if y.ndim < 2 else y
            return X, y
        return X

    @partial(jax.jit, static_argnums=(0,))
    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """Make predictions with the model"""
        X = self._process_data(X)
        return self.model.apply({'params': self.state.params}, X)