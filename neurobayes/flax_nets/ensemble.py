from typing import Dict, Type, Any, Union, Tuple, List, Optional, Callable
import jax
import jax.numpy as jnp
import flax
import optax
import numpy as np
from tqdm import tqdm

from .deterministic_nn import DeterministicNN, TrainState
from ..utils.utils import put_on_device, split_in_batches


class EnsembleDeterministicNN:
    """
    An ensemble of DeterministicNN models for improved predictions through model averaging.
    
    Args:
        architecture: a Flax model
        input_shape: (n_features,) or (*dims, n_channels)
        loss: type of loss, 'homoskedastic' (default), 'heteroskedastic', or 'classification'
        num_models: number of models in the ensemble
        learning_rate: Initial learning rate for each model
        map: Uses maximum a posteriori approximation for each model
        sigma: Standard deviation for Gaussian prior
        swa_config: Dictionary configuring the Stochastic Weight Averaging behavior (see DeterministicNN)
        collect_gradients: Whether to collect gradients during training
        init_random_seeds: List of random seeds for initializing each model. If None, seeds will be generated.
    """
    def __init__(self,
                 architecture: Type[flax.linen.Module],
                 input_shape: Union[int, Tuple[int]],
                 loss: str = 'homoskedastic',
                 num_models: int = 5,
                 learning_rate: float = 0.01,
                 map: bool = True,
                 sigma: float = 1.0,
                 swa_config: Optional[Dict] = None,
                 collect_gradients: bool = False,
                 init_random_seeds: Optional[List[int]] = None
                 ) -> None:
        
        self.architecture = architecture
        self.input_shape = input_shape
        self.loss = loss
        self.num_models = num_models
        self.learning_rate = learning_rate
        self.map = map
        self.sigma = sigma
        self.swa_config = swa_config
        self.collect_gradients = collect_gradients
        
        # Generate random seeds if not provided
        if init_random_seeds is None:
            init_random_seeds = list(range(num_models))
        elif len(init_random_seeds) < num_models:
            # If too few seeds provided, extend the list
            init_random_seeds = init_random_seeds + list(range(len(init_random_seeds), num_models))
        
        # Create ensemble members
        self.models = []
        for i in range(num_models):
            model = DeterministicNN(
                architecture=architecture,
                input_shape=input_shape,
                loss=loss,
                learning_rate=learning_rate,
                map=map,
                sigma=sigma,
                swa_config=swa_config,
                collect_gradients=collect_gradients,
                init_random_state=init_random_seeds[i]
            )
            self.models.append(model)
        
        self.trained = False
    
    def train(self, 
              X_train: jnp.ndarray, 
              y_train: jnp.ndarray, 
              epochs: int, 
              batch_size: int = None,
              bootstrap: bool = True,
              bootstrap_fraction: float = 1.0) -> None:
        """
        Train all models in the ensemble.
        
        Args:
            X_train: Input features
            y_train: Target values
            epochs: Number of epochs to train each model
            batch_size: Batch size for training (None means full batch)
            bootstrap: Whether to use bootstrap sampling for ensemble diversity
            bootstrap_fraction: Fraction of data to sample for each bootstrap if bootstrap=True
        """
        # Convert inputs to correct shapes if needed
        X_train, y_train = self._set_data(X_train, y_train)
        
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{self.num_models}...")
            
            if bootstrap:
                # Create a bootstrap sample
                n_samples = len(X_train)
                n_bootstrap = int(n_samples * bootstrap_fraction)
                indices = np.random.choice(n_samples, n_bootstrap, replace=True)
                X_boot = X_train[indices]
                y_boot = y_train[indices]
                
                # Train on bootstrap sample
                model.train(X_boot, y_boot, epochs, batch_size)
            else:
                # Train on full dataset
                model.train(X_train, y_train, epochs, batch_size)
        
        self.trained = True
    
    def predict(self, X: jnp.ndarray, return_individual: bool = False) -> Union[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray]]]:
        """
        Generate ensemble predictions by averaging individual model predictions.
        
        Args:
            X: Input features
            return_individual: Whether to return individual model predictions alongside ensemble predictions
            
        Returns:
            For homoskedastic and classification:
                If return_individual=False: (ensemble_mean, ensemble_variance)
                If return_individual=True: (ensemble_mean, ensemble_variance, individual_predictions)
            For heteroskedastic:
                If return_individual=False: (ensemble_mean, ensemble_total_variance)
                If return_individual=True: (ensemble_mean, ensemble_total_variance, individual_means, individual_variances)
        """
        if not self.trained:
            raise ValueError("Models must be trained before prediction")
        
        X = self._set_data(X)
        
        # Get predictions from each model
        individual_preds = []
        
        for model in self.models:
            pred = model.predict(X)
            individual_preds.append(pred)
        
        # For heteroskedastic models, we need to handle mean and variance separately
        if self.loss == 'heteroskedastic':
            individual_means = []
            individual_variances = []
            
            for pred in individual_preds:
                individual_means.append(pred[0])
                individual_variances.append(pred[1])
            
            # Convert to arrays for easier manipulation
            individual_means = jnp.array(individual_means)
            individual_variances = jnp.array(individual_variances)
            
            # Calculate ensemble mean and variance
            ensemble_mean = jnp.mean(individual_means, axis=0)
            
            # Total variance = mean of individual variances (aleatoric) + variance of means (epistemic)
            aleatoric_variance = jnp.mean(individual_variances, axis=0)
            epistemic_variance = jnp.var(individual_means, axis=0)
            ensemble_variance = aleatoric_variance + epistemic_variance
            
            if return_individual:
                return ensemble_mean, ensemble_variance, individual_means, individual_variances
            else:
                return ensemble_mean, ensemble_variance
        
        else:  # For homoskedastic regression and classification
            # Stack predictions
            all_preds = jnp.stack(individual_preds)
            
            # Calculate ensemble mean and variance
            ensemble_mean = jnp.mean(all_preds, axis=0)
            ensemble_variance = jnp.var(all_preds, axis=0)
            
            if return_individual:
                return ensemble_mean, ensemble_variance, individual_preds
            else:
                return ensemble_mean, ensemble_variance
    
    def predict_in_batches(self, X: jnp.ndarray, batch_size: int = 200, return_individual: bool = False):
        """
        Generate ensemble predictions in batches to avoid memory issues.
        
        Args:
            X: Input features
            batch_size: Batch size for prediction
            return_individual: Whether to return individual model predictions
            
        Returns:
            Similar to predict(), but processed in batches
        """
        X = self._set_data(X)
        X_batches = split_in_batches(X, batch_size)
        
        # For storing results
        ensemble_means = []
        ensemble_variances = []
        
        if return_individual:
            all_individual_preds = [[] for _ in range(self.num_models)]
        
        for X_batch in X_batches:
            if return_individual:
                batch_mean, batch_var, batch_individual = self.predict(X_batch, return_individual=True)
                
                if self.loss == 'heteroskedastic':
                    for i in range(self.num_models):
                        all_individual_preds[i].append((batch_individual[0][i], batch_individual[1][i]))
                else:
                    for i in range(self.num_models):
                        all_individual_preds[i].append(batch_individual[i])
            else:
                batch_mean, batch_var = self.predict(X_batch, return_individual=False)
            
            ensemble_means.append(batch_mean)
            ensemble_variances.append(batch_var)
        
        # Concatenate results
        ensemble_mean = jnp.concatenate(ensemble_means)
        ensemble_variance = jnp.concatenate(ensemble_variances)
        
        if return_individual:
            if self.loss == 'heteroskedastic':
                individual_means = []
                individual_variances = []
                
                for model_preds in all_individual_preds:
                    means = jnp.concatenate([p[0] for p in model_preds])
                    vars = jnp.concatenate([p[1] for p in model_preds])
                    individual_means.append(means)
                    individual_variances.append(vars)
                
                return ensemble_mean, ensemble_variance, individual_means, individual_variances
            else:
                individual_preds = [jnp.concatenate(model_preds) for model_preds in all_individual_preds]
                return ensemble_mean, ensemble_variance, individual_preds
        else:
            return ensemble_mean, ensemble_variance
    
    def _set_data(self, X: jnp.ndarray, y: jnp.ndarray = None) -> Union[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """Helper method to ensure data has correct shape"""
        X = X if X.ndim > 1 else X[:, None]
        
        if y is not None:
            if self.loss == 'classification':
                y = y.reshape(-1)
                # Note: no one-hot encoding here as it's handled by individual models
            else:
                y = y[:, None] if y.ndim < 2 else y  # Regression 
            return X, y
        return X
    
    def get_params(self) -> List[Dict]:
        """Get parameters for all models in the ensemble"""
        return [model.get_params() for model in self.models]
    
    def save_ensemble(self, filename: str) -> None:
        """Save the ensemble models to a file"""
        params_list = self.get_params()
        ensemble_data = {
            'params_list': params_list,
            'architecture': self.architecture,
            'input_shape': self.input_shape,
            'loss': self.loss,
            'num_models': self.num_models,
            'learning_rate': self.learning_rate,
            'map': self.map,
            'sigma': self.sigma,
            'swa_config': self.swa_config
        }
        np.save(filename, ensemble_data, allow_pickle=True)
        print(f"Ensemble saved to {filename}")
    
    @classmethod
    def load_ensemble(cls, filename: str) -> 'EnsembleDeterministicNN':
        """
        Load an ensemble from a file.
        
        Args:
            filename: Path to the saved ensemble file
            
        Returns:
            Loaded EnsembleDeterministicNN instance
        """
        ensemble_data = np.load(filename, allow_pickle=True).item()
        
        # Create ensemble with same configuration
        ensemble = cls(
            architecture=ensemble_data['architecture'],
            input_shape=ensemble_data['input_shape'],
            loss=ensemble_data['loss'],
            num_models=ensemble_data['num_models'],
            learning_rate=ensemble_data['learning_rate'],
            map=ensemble_data['map'],
            sigma=ensemble_data['sigma'],
            swa_config=ensemble_data['swa_config']
        )
        
        # Load parameters into each model
        params_list = ensemble_data['params_list']
        for i, params in enumerate(params_list):
            ensemble.models[i].state = ensemble.models[i].state.replace(params=params)
        
        ensemble.trained = True
        return ensemble