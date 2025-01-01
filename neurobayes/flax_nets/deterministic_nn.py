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
    collected_weights: List

class DeterministicNN:
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
        
        if loss not in ['homoskedastic', 'heteroskedastic']:
            raise ValueError("Select between 'homoskedastic' or 'heteroskedastic' loss")
        self.loss = loss
        
        # Initialize model
        key = jax.random.PRNGKey(0)
        params = self.model.init(key, jnp.ones((1, *input_shape)))['params']
        
        # Default SWA configuration
        self.default_swa_config = {
            'schedule': 'constant',
            'start_pct': 0.95,
        }
        
        # Update with user config if provided
        self.swa_config = {**self.default_swa_config, **(swa_config or {})}
        
        # Create initial state
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optax.adam(learning_rate),
            batch_stats=None,
            collected_weights=[]
        )
        
        self.learning_rate = learning_rate
        self.map = map
        self.sigma = sigma

    def train(self, X_train: jnp.ndarray, y_train: jnp.ndarray, 
              epochs: int, batch_size: int = None) -> None:
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
            cycle_length=self.swa_config['cycle_length']
        )
        
        X_batches = split_in_batches(X_train, batch_size)
        y_batches = split_in_batches(y_train, batch_size)
        num_batches = len(X_batches)
        
        collected_weights = []
        
        with tqdm(total=epochs, desc="Training Progress", leave=True) as pbar:
            for epoch in range(1, epochs + 1):
                # Get learning rate and collection decision for current epoch
                current_lr, should_collect = lr_schedule(epoch)
                
                epoch_loss = 0.0
                for X_batch, y_batch in zip(X_batches, y_batches):
                    self.state, batch_loss = self.train_step(
                        self.state, X_batch, y_batch, current_lr
                    )
                    epoch_loss += batch_loss
                
                # Collect weights if schedule indicates
                if should_collect:
                    collected_weights.append(self.state.params)
                
                # Update progress bar
                avg_epoch_loss = epoch_loss / num_batches
                status = f"Epoch {epoch}/{epochs}, Loss: {avg_epoch_loss:.4f}, LR: {current_lr:.6f}"
                if should_collect:
                    status += " (collected)"
                pbar.set_postfix_str(status)
                pbar.update(1)
        
        # Calculate final averaged weights if we collected any
        if collected_weights:
            self.state = self.state.replace(
                params=self.average_params(collected_weights),
                collected_weights=collected_weights
            )

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, state, inputs, targets, learning_rate):
        """Single training step with configurable learning rate"""
        loss, grads = jax.value_and_grad(self.total_loss)(state.params, inputs, targets)
        
        # Update optimizer learning rate
        new_tx = optax.chain(
            optax.scale_by_adam(),
            optax.scale_by_schedule(lambda _: learning_rate)
        )
        state = state.replace(tx=new_tx)
        
        # Apply gradients
        state = state.apply_gradients(grads=grads)
        return state, loss

    def average_params(self, params_list: List[Dict]) -> Dict:
        """Average a list of parameter dictionaries"""
        return jax.tree_util.tree_map(
            lambda *params: jnp.mean(jnp.stack(params), axis=0),
            *params_list
        )

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
        return l2_norm / (2 * self.sigma**2)
    
    def total_loss(self, params: Dict, inputs: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        loss_fn = self.mse_loss if self.loss == 'homoskedastic' else self.heteroskedastic_loss
        loss = loss_fn(params, inputs, targets)
        if self.map:
            prior_loss = self.gaussian_prior(params) / len(inputs)
            loss += prior_loss
        return loss

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        X = self.set_data(X)
        return self._predict(self.state, X)

    @partial(jax.jit, static_argnums=(0,))
    def _predict(self, state, X):
        return state.apply_fn({'params': state.params}, X)
    
    def set_data(self, X: jnp.ndarray, y: jnp.ndarray = None) -> jnp.ndarray:
        X = X if X.ndim > 1 else X[:, None]
        if y is not None:
            y = y[:, None] if y.ndim < 2 else y
            return X, y
        return X

    def get_params(self) -> Dict:
        return self.state.params