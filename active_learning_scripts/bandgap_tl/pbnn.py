#!/usr/bin/env python3

import sys
sys.path.append("../..")

from pathlib import Path
import logging
import datetime
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, NamedTuple
import numpy as np
import jax.numpy as jnp
from jax.random import PRNGKey
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import pickle

# Local imports
from neurobayes.flax_nets.mlp import FlaxMLP
from neurobayes import PartialBNN, DeterministicNN
from neurobayes.utils import utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ScaledData(NamedTuple):
    """Container for scaled data and corresponding scalers."""
    X_scaled: np.ndarray
    y_theory_scaled: np.ndarray
    y_exp_scaled: np.ndarray
    X_scaler: StandardScaler
    y_scaler: StandardScaler

@dataclass
class ExplorationConfig:
    """Configuration parameters for the exploration process."""
    seeds: List[int]
    exploration_steps: int
    pretrain_epochs: int
    pretrain_lr: float
    pretrain_batch_size: int
    probabilistic_layer_names: List[str]
    output_dir: Path
    file: Path
    hidden_dims: List[int]
    activation: str
    priors_sigma: float
    experiment_name: str = "bandgaps"
    
    def get_output_filename(self) -> Path:
        """Generate a descriptive output filename based on parameters."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        layer_str = "-".join(l.lower() for l in self.probabilistic_layer_names)
        params = f"pretrain_prob{layer_str}_steps{self.exploration_steps}_epochs{self.pretrain_epochs}_lr{self.pretrain_lr}"
        return self.output_dir / f"{self.experiment_name}_{params}_{timestamp}.pkl"

class DataProcessor:
    """Handles data loading and preprocessing."""
    
    def __init__(self):
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
    def load_and_preprocess(self, file: str) -> ScaledData:
        """Load and preprocess both theoretical and experimental data."""
        try:
            # Load theoretical and experimental data
            data = np.load(file)
            X = data['features']
            y_experimental = data['targets_exp']
            y_theory = data['targets_theory']
            
            self._validate_input_data(X, y_theory, y_experimental)
            
            # Scale features
            X_scaled = self.X_scaler.fit_transform(X)
            
            # Reshape y data for scaling
            y_theory = y_theory.reshape(-1, 1)
            y_experimental = y_experimental.reshape(-1, 1)
            
            # Fit scaler on theoretical data (which typically has more samples)
            y_theory_scaled = self.y_scaler.fit_transform(y_theory)
            # Transform experimental data using the same scaler
            y_exp_scaled = self.y_scaler.transform(y_experimental)
            
            # Convert back to 1D arrays
            y_theory_scaled = y_theory_scaled.ravel()
            y_exp_scaled = y_exp_scaled.ravel()
            
            return ScaledData(
                X_scaled=X_scaled,
                y_theory_scaled=y_theory_scaled,
                y_exp_scaled=y_exp_scaled,
                X_scaler=self.X_scaler,
                y_scaler=self.y_scaler
            )
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    @staticmethod
    def _validate_input_data(X: np.ndarray, y_theory: np.ndarray, y_exp: np.ndarray) -> None:
        """Validate input data types and shapes."""
        if not all(isinstance(arr, (np.ndarray, jnp.ndarray)) for arr in [X, y_theory, y_exp]):
            raise TypeError("All inputs must be NumPy or JAX NumPy arrays")
        if not (X.shape[0] == y_theory.shape[0] == y_exp.shape[0]):
            raise ValueError("X, y_theory, and y_exp must have same number of samples")

class ActiveLearner:
    """Handles the active learning process with pre-training."""
    
    def __init__(self, config: ExplorationConfig):
        self.config = config
        self.pretrained_net = None
        self.pretrained_params = None
        self.y_scaler = None
        
    def pretrain(self, scaled_data: ScaledData) -> None:
        """Pre-train the deterministic model on theoretical data."""
        logger.info("Starting pre-training phase...")
        
        # Store y_scaler for later use
        self.y_scaler = scaled_data.y_scaler
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_data.X_scaled, 
            scaled_data.y_theory_scaled,
            test_size=0.1, 
            random_state=self.config.seeds[0]
        )
        
        # Initialize the network
        net = FlaxMLP(
            hidden_dims=self.config.hidden_dims,
            target_dim=1,
            activation=self.config.activation
        )
        
        # Create deterministic model and train
        det_model = DeterministicNN(
            net, 
            scaled_data.X_scaled.shape[-1], 
            learning_rate=self.config.pretrain_lr,
            map=False
        )
        
        det_model.train(
            X_train, 
            y_train,
            epochs=self.config.pretrain_epochs,
            batch_size=self.config.pretrain_batch_size
        )
        
        # Evaluate deterministic model on test data
        y_pred_scaled = det_model.predict(X_test).squeeze()
        # Inverse transform predictions and true values for interpretable metrics
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_test_unscaled = self.y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
        det_mae = utils.mae(y_pred, y_test_unscaled)
        logger.info(f"Deterministic model MAE on test data (unscaled): {det_mae:.4f}")
        
        # Store pre-trained model and parameters
        self.pretrained_net = det_model.model
        self.pretrained_params = det_model.state.params
        logger.info("Pre-training completed")

    def run_exploration(
        self, 
        scaled_data: ScaledData,
        seed: int
    ) -> Dict[str, List[float]]:
        """Run the active learning exploration process."""
        np.random.seed(seed)
        
        X_measured, X_unmeasured, y_measured, y_unmeasured = train_test_split(
            scaled_data.X_scaled, 
            scaled_data.y_exp_scaled,
            test_size=0.95, 
            random_state=seed
        )
        
        metrics = {metric: [] for metric in ['mse', 'mae', 'nlpd', 'coverage']}
        
        for step in range(self.config.exploration_steps):
            logger.info(f'Seed {seed}, Step {step}')
                
            try:
                results = self._exploration_step(
                    X_measured, y_measured, X_unmeasured, y_unmeasured
                )
                
                X_measured, y_measured = results['new_measured']
                X_unmeasured, y_unmeasured = results['new_unmeasured']
                
                for metric, value in results['metrics'].items():
                    metrics[metric].append(value)
                    
                # Log metrics (already in unscaled space)
                logger.info(
                    f"RMSE: {np.sqrt(results['metrics']['mse']):.4f}, "
                    f"MAE: {results['metrics']['mae']:.4f}, "
                    f"NLPD: {results['metrics']['nlpd']:.4f}, "
                    f"Coverage: {results['metrics']['coverage']:.4f}\n"
                )
                
            except Exception as e:
                logger.error(f"Error in exploration step {step}: {str(e)}")
                raise
                
        return metrics
    
    def _exploration_step(
        self, 
        X_measured: np.ndarray,
        y_measured: np.ndarray,
        X_unmeasured: np.ndarray,
        y_unmeasured: np.ndarray,
        max_num_restarts: int = 1
    ) -> Dict[str, Any]:
        """Perform a single exploration step using pre-trained weights"""
        
        # Initialize PartialBNN with pre-trained weights
        model = PartialBNN(
            self.pretrained_net,
            self.pretrained_params,
            probabilistic_layer_names=self.config.probabilistic_layer_names
        )
        
        # Fit model using pre-trained weights as initialization
        for i in range(max_num_restarts):
            model.fit(
                X_measured,
                y_measured,
                priors_sigma=self.config.priors_sigma,
                num_warmup=1000,
                num_samples=1000,
                extra_fields=('accept_prob',),
                rng_key=PRNGKey(i)
            )
            if model.mcmc.get_extra_fields()['accept_prob'].mean() > 0.55:
                break

        posterior_mean, posterior_var = model.predict(X_unmeasured)
        posterior_mean, posterior_var = posterior_mean.squeeze(), posterior_var.squeeze()
        
        # Convert predictions and variances to unscaled space
        posterior_mean_unscaled = self.y_scaler.inverse_transform(
            posterior_mean.reshape(-1, 1)
        ).ravel()
        
        # Transform variance to unscaled space
        posterior_var_unscaled = posterior_var * (self.y_scaler.scale_[0] ** 2)
        
        # Convert true values to unscaled space
        y_unmeasured_unscaled = self.y_scaler.inverse_transform(
            y_unmeasured.reshape(-1, 1)
        ).ravel()
        
        # Calculate metrics in unscaled space
        metrics = {
            'mse': utils.mse(posterior_mean_unscaled, y_unmeasured_unscaled),
            'mae': utils.mae(posterior_mean_unscaled, y_unmeasured_unscaled),
            'nlpd': utils.nlpd(y_unmeasured_unscaled, posterior_mean_unscaled, posterior_var_unscaled),
            'coverage': utils.coverage(y_unmeasured_unscaled, posterior_mean_unscaled, posterior_var_unscaled)
        }
        
        # Selection of next point still based on scaled variance
        next_point_idx = posterior_var.argmax()
        
        # Update measured data
        new_X_measured = np.append(
            X_measured, 
            X_unmeasured[next_point_idx][None], 
            axis=0
        )
        new_y_measured = np.append(
            y_measured, 
            y_unmeasured[next_point_idx]
        )
        
        # Update unmeasured data
        new_X_unmeasured = np.delete(X_unmeasured, next_point_idx, axis=0)
        new_y_unmeasured = np.delete(y_unmeasured, next_point_idx)
        
        return {
            'new_measured': (new_X_measured, new_y_measured),
            'new_unmeasured': (new_X_unmeasured, new_y_unmeasured),
            'metrics': metrics
        }

def parse_arguments() -> ExplorationConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Pre-trained PartialBNN Active Learning Script")
    parser.add_argument(
        "--seeds", 
        nargs="+", 
        type=int, 
        default=[1, 2, 3, 4, 5],
        help="Random seeds for data splitting"
    )
    parser.add_argument(
        "--exploration_steps",
        type=int,
        default=200,
        help="Number of exploration steps"
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=500,
        help="Number of pre-training epochs"
    )
    parser.add_argument(
        "--pretrain-lr",
        type=float,
        default=5e-3,
        help="Pre-training learning rate"
    )
    parser.add_argument(
        "--pretrain-batch-size",
        type=int,
        default=32,
        help="Pre-training batch size"
    )
    parser.add_argument(
        "--priors-sigma",
        type=float,
        default=0.5,
        help="Sigma for priors in PartialBNN"
    )
    parser.add_argument(
        "--probabilistic-layer-names",
        nargs="+",
        type=str,
        default=['Dense2', 'Dense3', 'Dense4'],
        help="Names of layers to make probabilistic"
    )
    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        default=[32, 16, 8, 8],
        help="Dimensions of hidden layers"
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=['tanh', 'silu'],
        default='silu',
        help="Activation function to use"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/pbnn321688yscaled"),
        help="Directory to save results"
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("bandgaps_non_metals_transfer_learning.npz"),
        help="Input file with theoretical and experimental data"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="bandgaps",
        help="Name of the experiment"
    )
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return ExplorationConfig(**vars(args))

def main():
    """Main execution function."""
    config = parse_arguments()
    data_processor = DataProcessor()
    active_learner = ActiveLearner(config)
    
    try:
        # Load and preprocess data
        scaled_data = data_processor.load_and_preprocess(config.file)
        
        # Pre-train on theoretical data (only once)
        active_learner.pretrain(scaled_data)
        
        results = {
            'config': asdict(config),
            'runs': {},
        }
        
        # Run active learning with multiple seeds
        for seed in config.seeds:
            results['runs'][seed] = active_learner.run_exploration(scaled_data, seed)
            
        # Save results
        output_file = config.get_output_filename()
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        logger.info(f"Results successfully saved to {output_file}")
        
        # Save metadata
        metadata_file = output_file.with_suffix('.json')
        metadata = {
            'parameters': {k: str(v) if isinstance(v, Path) else v 
                         for k, v in asdict(config).items()},
            'timestamp': datetime.datetime.now().isoformat(),
            'results_file': output_file.name
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_file}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()