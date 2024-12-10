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
import jax.numpy as jnp
import numpyro.distributions as dist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import pickle

# Local imports
from neurobayes.flax_nets.mlp import FlaxMLP
from neurobayes import PartialBNN
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
    sgd_epochs: int
    sgd_lr: float
    probabilistic_layer_names: List[str]
    output_dir: Path
    input_file: Path
    hidden_dims: List[int]  # New: configurable hidden dimensions
    activation: str  # New: configurable activation function
    experiment_name: str = "esol"
    
    def get_output_filename(self) -> Path:
        """Generate a descriptive output filename based on parameters."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Convert layer names to compact string for filename
        layer_str = "-".join(l.lower() for l in self.probabilistic_layer_names)
        params = f"prob{layer_str}_steps{self.exploration_steps}_epochs{self.sgd_epochs}_lr{self.sgd_lr}"
        return self.output_dir / f"{self.experiment_name}_{params}_{timestamp}.pkl"
    
        
class DataProcessor:
    """Handles data loading and preprocessing."""
    
    def __init__(self, scaler=None):
        self.scaler = scaler or StandardScaler()
        
    def load_and_preprocess(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess the data."""
        try:
            data = np.load(file_path)
            X = data["features"]
            y = data["targets"]
            
            self._validate_input_data(X, y)
            X_scaled = self.scaler.fit_transform(X)
            
            return X_scaled, y
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

class ActiveLearner:
    """Handles the active learning process."""
    
    def __init__(self, config: ExplorationConfig):
        self.config = config
        self._init_model()
        
    def _init_model(self) -> None:
        """Initialize the neural network model."""
        net = FlaxMLP(
            hidden_dims=self.config.hidden_dims,
            target_dim=1,
            activation=self.config.activation
        )
        self.model = PartialBNN(
            net, 
            probabilistic_layer_names=self.config.probabilistic_layer_names
        )
        
    def fit_and_predict(
        self, 
        X_measured: np.ndarray, 
        y_measured: np.ndarray, 
        X_unmeasured: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the model and make predictions."""
        self.model.fit(
            X_measured, 
            y_measured,
            sgd_epochs=self.config.sgd_epochs,
            sgd_lr=self.config.sgd_lr,
            sgd_batch_size=16,
            map_sigma=utils.calculate_sigma(X_measured),
            num_warmup=1000,
            num_samples=1000
        )
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
        
        # Calculate metrics
        metrics = {
            'mse': utils.mse(posterior_mean, y_unmeasured),
            'mae': utils.mae(posterior_mean, y_unmeasured),
            'nlpd': utils.nlpd(y_unmeasured, posterior_mean, posterior_var),
            'coverage': utils.coverage(y_unmeasured, posterior_mean, posterior_var)
        }
        
        return {
            'new_measured': (new_X_measured, new_y_measured),
            'new_unmeasured': (new_X_unmeasured, new_y_unmeasured),
            'metrics': metrics
        }

def parse_arguments() -> ExplorationConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="PartialBNN Active Learning Script")
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
        default=1,
        help="Number of exploration steps"
    )
    parser.add_argument(
        "--sgd_epochs",
        type=int,
        default=2000,
        help="Number of SGD epochs"
    )
    parser.add_argument(
        "--sgd_lr",
        type=float,
        default=5e-3,
        help="SGD learning rate"
    )
    parser.add_argument(
        "--probabilistic-layer-names",
        nargs="+",
        type=str,
        default=['Dense2', 'Dense4'],
        help="Names of layers to make probabilistic (e.g., Dense0 Dense4)"
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
        default=Path("results/pbnn8888"),
        help="Directory to save results"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="esol",
        help="Name of the experiment (used in output filename)"
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("esol.npz"),
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
    active_learner = ActiveLearner(config)
    
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