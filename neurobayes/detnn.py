from typing import Dict, Type, Any
import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
import optax
from functools import partial
from tqdm import tqdm


class TrainState(train_state.TrainState):
    batch_stats: Any

class DeterministicNN:

    def __init__(self,
                 architecture: Type[flax.linen.Module],
                 input_dim: int,
                 loss: str = 'homoskedastic',
                 learning_rate: float = 0.01) -> None:
        
        self.model = architecture
        self.loss = loss
        key = jax.random.PRNGKey(0)
        params = self.model.init(key, jnp.ones((1, input_dim)))['params']
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optax.adam(learning_rate),
            batch_stats=None
        )

    def mse_loss(self, params: Dict, inputs: jnp.ndarray,
                 targets: jnp.ndarray) -> jnp.ndarray:
        predictions = self.model.apply({'params': params}, inputs)
        return jnp.mean((predictions - targets) ** 2)
    
    def heteroskedastic_loss(self, params: Dict, inputs: jnp.ndarray,
                             targets: jnp.ndarray) -> jnp.ndarray:
        y_pred, y_var = self.model.apply({'params': params}, inputs)
        return jnp.mean(0.5 * jnp.log(y_var) + 0.5 * (targets - y_pred)**2 / y_var)

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, state, inputs, targets):
        loss_fn = self.mse_loss if self.loss == 'homoskedastic' else self.heteroskedastic_loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params, inputs, targets)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def train(self, X_train: jnp.ndarray, y_train: jnp.ndarray, epochs: int) -> None:
        X_train, y_train = self.set_data(X_train, y_train)
        with tqdm(total=epochs, desc="Training Progress", leave=True) as pbar:
            for epoch in range(epochs):
                self.state, loss = self.train_step(self.state, X_train, y_train)
                pbar.update(1)
                pbar.set_postfix_str(f"Epoch {epoch+1}, Loss: {loss:.4f}")

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

