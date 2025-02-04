from typing import Dict, Type, Any, Union, Tuple, List, Optional, Callable
import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
import optax
from functools import partial
from tqdm import tqdm

from .train_utils import create_swa_schedule
from ..utils.utils import split_in_batches


class TrainState(train_state.TrainState):
    batch_stats: Any


class DeterministicNN:
    """
    Args:
        architecture: a Flax model
        input_shape: (n_samples, n_features) or (n_samples, *dims, n_channels)
        loss: type of loss, 'homoskedastic' (default) or 'heteroskedastic'
        learning_rate: Initial learning rate
        map: Uses maximum a posteriori approximation
        sigma: Standard deviation for Gaussian prior
        swa_config: Dictionary configuring the Stochastic Weight Averaging behavior:
            - 'schedule': Type of learning rate schedule and weight collection strategy.
                Options:
                - 'constant': Uses constant learning rate, collects weights after start_pct
                - 'linear': Linearly decays learning rate to swa_lr, then collects weights
                - 'cyclic': Cycles learning rate between swa_lr and a peak value, collecting at cycle starts
            - 'start_pct': When to start SWA as fraction of total epochs (default: 0.95)
            - 'swa_lr': Final/SWA learning rate (default: same as initial learning_rate)
            Additional parameters based on schedule type:
            - For 'linear' and 'cyclic':
                - 'decay_fraction': Fraction of total epochs for decay period (default: 0.05)
            - For 'cyclic' only:
                - 'cycle_length': Number of epochs per cycle (required)
    """
    def __init__(self,
                 architecture: Type[flax.linen.Module],
                 input_shape: Union[int, Tuple[int]],
                 loss: str = 'homoskedastic',
                 learning_rate: float = 0.01,
                 map: bool = True,
                 sigma: float = 1.0,
                 swa_config: Optional[Dict] = None) -> None:
        
        input_shape = (input_shape,) if isinstance(input_shape, int) else input_shape
        self.model = architecture

        is_transformer = 'transformer' in self.model.__class__.__name__.lower()
        input_dtype = jnp.int32 if is_transformer else jnp.float32
        
        if loss not in ['homoskedastic', 'heteroskedastic', 'classification']:
            raise ValueError("Select between 'homoskedastic', 'heteroskedastic', or 'classification' loss")
        self.loss = loss
        
        # Initialize model
        key = jax.random.PRNGKey(0)
        params = self.model.init(
            key,
            jnp.ones((1, *input_shape), dtype=input_dtype)
        )['params']
        
        # Default SWA configuration with all required parameters
        self.default_swa_config = {
            'schedule': 'constant',
            'start_pct': 0.95,
            'swa_lr': learning_rate,  # Same as initial for constant schedule
        }
        
        # Update with user config if provided
        self.swa_config = {**self.default_swa_config, **(swa_config or {})}
        
        self.current_lr = learning_rate
        self.optimizer = optax.adam(learning_rate)
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=self.optimizer,
            batch_stats=None,
        )

        self.learning_rate = learning_rate
        self.map = map
        self.sigma = sigma

        self.params_history = []

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, state, inputs, targets):
        """JIT-compiled training step"""
        dropout_key = jax.random.PRNGKey(state.step)
        loss, grads = jax.value_and_grad(self.total_loss)(
            state.params, 
            inputs, 
            targets, 
            rngs={'dropout': dropout_key}
        )
        state = state.apply_gradients(grads=grads)
        return state, loss
    
    def update_learning_rate(self, learning_rate: float):
        """Update the optimizer with a new learning rate"""
        if learning_rate != self.current_lr:
            self.current_lr = learning_rate
            self.state = self.state.replace(tx=optax.adam(learning_rate))

    def train(self, X_train: jnp.ndarray, y_train: jnp.ndarray, epochs: int, batch_size: int = None) -> None:
        X_train, y_train = self.set_data(X_train, y_train)
        
        if batch_size is None or batch_size >= len(X_train):
            batch_size = len(X_train)

        # Calculate SWA start epoch
        start_epoch = int(epochs * self.swa_config['start_pct'])
        
        # Create learning rate schedule
        lr_schedule = create_swa_schedule(
            schedule_type=self.swa_config['schedule'],
            initial_lr=self.learning_rate,
            swa_lr=self.swa_config['swa_lr'],
            start_epoch=start_epoch,
            total_epochs=epochs,
            cycle_length=self.swa_config.get('cycle_length'),
            decay_fraction=self.swa_config.get('decay_fraction', 0.05)
        )
        
        X_batches = split_in_batches(X_train, batch_size)
        y_batches = split_in_batches(y_train, batch_size)
        num_batches = len(X_batches)
        
        with tqdm(total=epochs, desc="Training Progress", leave=True) as pbar:
            for epoch in range(epochs):
                # Get learning rate and collection decision from schedule
                learning_rate, should_collect = lr_schedule(epoch)
                # Update learning rate if needed 
                self.update_learning_rate(learning_rate)
                
                epoch_loss = 0.0
                for i, (X_batch, y_batch) in enumerate(zip(X_batches, y_batches)):
                    self.state, batch_loss = self.train_step(self.state, X_batch, y_batch)
                    epoch_loss += batch_loss
                
                # Collect weights if scheduled
                if should_collect:
                    self._store_params(self.state.params)
                
                avg_epoch_loss = epoch_loss / num_batches
                pbar.set_postfix_str(
                    f"Epoch {epoch+1}/{epochs}, "
                    f"LR: {learning_rate:.6f}, "
                    f"Loss: {avg_epoch_loss:.4f} "
                )
                pbar.update(1)
                
        # Average collected weights if any were collected
        if self.params_history:
            self.state = self.state.replace(params=self.average_params())

    def _store_params(self, params: Dict) -> None:
        self.params_history.append(params)

    def average_params(self) -> Dict:
        """Average model parameters, excluding normalization layers"""
        if not self.params_history:
            return self.state.params
            
        def should_average(path_tuple):
            """Check if parameter should be averaged based on its path"""
            path_str = '/'.join(str(p) for p in path_tuple)
            skip_patterns = ['LayerNorm', 'BatchNorm', 'embedding/norm']
            return not any(pattern in path_str for pattern in skip_patterns)
        
        def average_leaves(*leaves, path=()):
            """Average parameters if not in normalization layers"""
            if should_average(path):
                return jnp.mean(jnp.stack(leaves), axis=0)
            else:
                return leaves[0]  # Keep original parameters
        
        # Apply averaging with path information
        averaged_params = jax.tree_util.tree_map_with_path(
            lambda path, *values: average_leaves(*values, path=path),
            self.params_history[0],
            *self.params_history[1:]
        )
        
        return averaged_params

    def reset_swa(self):
        """Reset SWA collections"""
        self.params_history = []
    
    def gaussian_prior(self, params: Dict) -> jnp.ndarray:
        l2_norm = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
        return l2_norm / (2 * self.sigma**2)
    
    def total_loss(self, params: Dict, inputs: jnp.ndarray, targets: jnp.ndarray, 
                rngs: Dict) -> jnp.ndarray:
        if self.loss == 'classification':
            loss = self.cross_entropy_loss(params, inputs, targets, rngs)
        else:
            loss_fn = self.mse_loss if self.loss == 'homoskedastic' else self.heteroskedastic_loss
            loss = loss_fn(params, inputs, targets, rngs)
        if self.map:
            prior_loss = self.gaussian_prior(params) / len(inputs)
            loss += prior_loss
        return loss

    def mse_loss(self, params: Dict, inputs: jnp.ndarray,
                targets: jnp.ndarray, rngs: Dict) -> jnp.ndarray:
        predictions = self.model.apply(
            {'params': params}, 
            inputs, 
            enable_dropout=True,
            rngs=rngs
        )
        return jnp.mean((predictions - targets) ** 2)
        
    def heteroskedastic_loss(self, params: Dict, inputs: jnp.ndarray,
                            targets: jnp.ndarray, rngs: Dict) -> jnp.ndarray:
        y_pred, y_var = self.model.apply(
            {'params': params}, 
            inputs, 
            enable_dropout=True,
            rngs=rngs
        )
        return jnp.mean(0.5 * jnp.log(y_var) + 0.5 * (targets - y_pred)**2 / y_var)
        
    def cross_entropy_loss(self, params: Dict, inputs: jnp.ndarray,
                        targets: jnp.ndarray, rngs: Dict) -> jnp.ndarray:
        logits = self.model.apply(
            {'params': params}, 
            inputs, 
            enable_dropout=True,
            rngs=rngs
        )
        return -jnp.mean(jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1))

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        X = self.set_data(X)
        return self._predict(self.state, X)

    @partial(jax.jit, static_argnums=(0,))
    def _predict(self, state, X):
        predictions = state.apply_fn({'params': state.params}, X, enable_dropout=False)
        if self.loss == 'classification':
            return jax.nn.softmax(predictions)
        return predictions
    
    def set_data(self, X: jnp.ndarray, y: jnp.ndarray = None) -> jnp.ndarray:
        X = X if X.ndim > 1 else X[:, None]
        
        if y is not None:
            if self.loss == 'classification':
                y = y.reshape(-1)
                y = jax.nn.one_hot(y, num_classes=self.model.target_dim)
            else:
                y = y[:, None] if y.ndim < 2 else y  # Regression 
            return X, y
        return X

    def get_params(self) -> Dict:
        return self.state.params