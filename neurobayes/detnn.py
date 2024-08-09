from typing import Dict, Type
import jax
import jax.numpy as jnp
import flax
import optax
from tqdm import tqdm


class DeterministicNN:

    def __init__(self,
                 architecture: Type[flax.linen.Module],
                 input_dim: int,
                 learning_rate: float = 0.01) -> None:
        self.model = architecture
        self.params = self.model.init(
            jax.random.PRNGKey(0), jnp.ones((1, input_dim)))['params']
        self.optimizer = optax.adam(learning_rate=learning_rate)
        self.opt_state = self.optimizer.init(self.params)

    def mse_loss(self, params: Dict, inputs: jnp.ndarray,
                 targets: jnp.ndarray) -> jnp.ndarray:
        predictions = self.model.apply({'params': params}, inputs)
        return jnp.mean((predictions - targets) ** 2)

    def train_step(self, inputs: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        # Calculate gradients
        loss_value, grads = jax.value_and_grad(self.mse_loss)(self.params, inputs, targets)
        # Update parameters and optimizer state
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        return loss_value

    def train(self, X_train: jnp.ndarray, y_train: jnp.ndarray, epochs: int) -> None:
        X_train, y_train = self.set_data(X_train, y_train)
        with tqdm(total=epochs, desc="Training Progress", leave=True) as pbar:
            for epoch in range(epochs):
                loss = self.train_step(X_train, y_train)
                pbar.update(1)
                pbar.set_postfix_str(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        X = self.set_data(X)
        return self.model.apply({'params': self.params}, X)
    
    def set_data(self, X: jnp.ndarray, y: jnp.ndarray = None)  -> jnp.ndarray:
        X = X if X.ndim > 1 else X[:, None]
        if y is not None:
            y = y[:, None] if y.ndim < 2 else y
            return X, y
        return X

