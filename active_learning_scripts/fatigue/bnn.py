#!/usr/bin/env python3

import sys
sys.path.append("../..")

from pathlib import Path
import logging
import datetime
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any
import numpy as np
from jax.random import PRNGKey
import jax.numpy as jnp
import numpyro.distributions as dist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import pickle

# Local imports
from neurobayes.flax_nets.mlp import FlaxMLP
from neurobayes import BNN
from neurobayes.utils import utils

# import jax
# jax.config.update("jax_enable_x64", True)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ExplorationConfig:
    """Configuration parameters for the exploration process."""
    seeds: List[int]
    exploration_steps: int
    output_dir: Path
    input_file: Path
    hidden_dims: List[int]
    activation: str 
    experiment_name: str = "fatigue"
    
    def get_output_filename(self) -> Path:
        """Generate a descriptive output filename based on parameters."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Convert layer names to compact string for filename
        params = f"steps{self.exploration_steps}"
        return self.output_dir / f"{self.experiment_name}_{params}_{timestamp}.pkl"
    
        
class DataProcessor:
    """Handles data loading and preprocessing."""
    
    def __init__(self, scalerX=None, scalerY=None):
        self.scalerX = scalerX or StandardScaler()
        self.scalerY = scalerY or StandardScaler()
        
    def load_and_preprocess(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess the data."""
        try:
            data = np.load(file_path)
            X = data["features"]
            y = data["targets"]
            
            self._validate_input_data(X, y)
            X_scaled = self.scalerX.fit_transform(X)

            y_reshaped = y.reshape(-1, 1)
            y_scaled = self.scalerY.fit_transform(y_reshaped).ravel()
            
            return X_scaled, y_scaled

        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise

    @staticmethod
    def _validate_input_data(X: np.ndarray, y: np.ndarray) -> None:
        """Validate input data types and shapes."""
        if not isinstance(X, (np.ndarray, jnp.ndarray)):
            raise TypeError(f"X must be a NumPy or JAX NumPy array, not {type(X)}")
        if not isinstance(y, (np.ndarray, jnp.ndarray)):
            raise TypeError(f"y must be a NumPy or JAX NumPy array, not {type(y)}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples. Got {X.shape[0]} and {y.shape[0]}")

    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """Transform scaled y values back to original scale."""
        y_reshaped = y_scaled.reshape(-1, 1)
        return self.scalerY.inverse_transform(y_reshaped).ravel()
        
class ActiveLearner:
    """Handles the active learning process."""
    
    def __init__(self, config: ExplorationConfig, data_processor: DataProcessor):
        self.config = config
        self.data_processor = data_processor 
        self._init_model()
        
    def _init_model(self) -> None:
        """Initialize the neural network model."""
        net = FlaxMLP(
            hidden_dims=self.config.hidden_dims,
            target_dim=1,
            activation=self.config.activation
        )
        self.model = BNN(net)
        
    def fit_and_predict(
        self, 
        X_measured: np.ndarray, 
        y_measured: np.ndarray, 
        X_unmeasured: np.ndarray,
        max_num_restarts: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the model and make predictions."""
        for i in range(max_num_restarts):
            self.model.fit(
                X_measured, 
                y_measured,
                num_warmup=1000,
                num_samples=1000,
                extra_fields=('accept_prob',),
                rng_key=PRNGKey(i)
            )
            if self.model.mcmc.get_extra_fields()['accept_prob'].mean() > 0.55:
                break 
        posterior_mean, posterior_var = self.model.predict(X_unmeasured)
        return posterior_mean.squeeze(), posterior_var.squeeze()

    def run_exploration(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        seed: int
    ) -> Dict[str, List[float]]:
        """Run the active learning exploration process."""
        np.random.seed(seed)
        
        X_measured, X_unmeasured, y_measured, y_unmeasured = train_test_split(
            X, y, test_size=0.95, random_state=seed
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
        y_unmeasured: np.ndarray
    ) -> Dict[str, Any]:
        """Perform a single exploration step"""

        # Reinitialize the model at each step
        self._init_model()

        posterior_mean, posterior_var = self.fit_and_predict(
            X_measured, y_measured, X_unmeasured
        )

        # Transform predictions and true values back to original scale
        posterior_mean_orig = self.data_processor.inverse_transform_y(posterior_mean)
        y_unmeasured_orig = self.data_processor.inverse_transform_y(y_unmeasured)
        
        # Scale the variance accordingly (multiply by square of scale factor)
        scale_factor = self.data_processor.scalerY.scale_[0]
        posterior_var_orig = posterior_var * (scale_factor ** 2)
        
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
        
        # Calculate metrics using original scale values
        metrics = {
            'mse': utils.mse(posterior_mean_orig, y_unmeasured_orig),
            'mae': utils.mae(posterior_mean_orig, y_unmeasured_orig),
            'nlpd': utils.nlpd(y_unmeasured_orig, posterior_mean_orig, posterior_var_orig),
            'coverage': utils.coverage(y_unmeasured_orig, posterior_mean_orig, posterior_var_orig)
        }
        
        return {
            'new_measured': (new_X_measured, new_y_measured),
            'new_unmeasured': (new_X_unmeasured, new_y_unmeasured),
            'metrics': metrics
        }

def parse_arguments() -> ExplorationConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="BNN Active Learning Script")
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
        "--hidden-dims",
        nargs="+",
        type=int,
        default=[8, 8, 8, 8],
        help="Dimensions of hidden layers (e.g., 32 16 8 8)"
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=['tanh', 'silu'],
        default='silu',
        help="Activation function to use in the neural network"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/bnn8888"),
        help="Directory to save results"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="fatigue",
        help="Name of the experiment (used in output filename)"
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("fatigue.npz"),
        help="Input data file path"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    return ExplorationConfig(**vars(args))


def main():
    """Main execution function."""
    config = parse_arguments()
    data_processor = DataProcessor()
    active_learner = ActiveLearner(config, data_processor)
    
    try:
        X, y = data_processor.load_and_preprocess(config.input_file)
        
        results = {
            'config': asdict(config),  # Store configuration in results
            'runs': {}
        }
        
        for seed in config.seeds:
            results['runs'][seed] = active_learner.run_exploration(X, y, seed)
            
        output_file = config.get_output_filename()
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        logger.info(f"Results successfully saved to {output_file}")
        
        # Also save a metadata file in JSON format for easy reading
        metadata_file = output_file.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump({
                'parameters': {k: str(v) if isinstance(v, Path) else v 
                             for k, v in asdict(config).items()},
                'timestamp': datetime.datetime.now().isoformat(),
                'results_file': output_file.name
            }, f, indent=2)
        logger.info(f"Metadata saved to {metadata_file}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()