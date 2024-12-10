#!/usr/bin/env python3

import sys
sys.path.append("../..")

import numpy as np
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Literal
from pathlib import Path
import pickle
import json
import datetime

from neurobayes.flax_nets.mlp import FlaxMLP
from neurobayes import DeterministicNN
from neurobayes.utils import utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ComparisonConfig:
    """Configuration for the comparison experiment."""
    hidden_dims: List[int]
    learning_rate: float
    epochs: int
    batch_size: int
    seeds: List[int]
    use_map: bool
    output_dir: Path
    pbnn_results_file: Path
    activation: Literal['silu', 'tanh'] = 'silu'

class ModelComparator:
    """Handles comparison between PBNN and deterministic NN."""
    
    def __init__(self, config: ComparisonConfig):
        self.config = config
        
    def load_pbnn_results(self) -> Dict:
        """Load results from PBNN active learning experiment."""
        with open(self.config.pbnn_results_file, 'rb') as f:
            return pickle.load(f)
    
    def run_comparison(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Run comparison experiments for each seed."""
        pbnn_results = self.load_pbnn_results()
        results = {
            'runs': {}
        }
        
        # Create the model architecture
        architecture = FlaxMLP(
            hidden_dims=self.config.hidden_dims,
            target_dim=1,
            activation=self.config.activation
        )
        
        for seed in self.config.seeds:
            logger.info(f"Running comparison for seed {seed}")
            
            # Get the final training set size from PBNN results
            pbnn_run = pbnn_results['runs'][seed]
            initial_points = int(len(X) * 0.05)  # 5% of total data
            final_train_size = initial_points + len(pbnn_run['mse'])
            
            # Create training set of same size
            np.random.seed(seed)
            indices = np.random.permutation(len(X))
            train_indices = indices[:final_train_size]
            test_indices = indices[final_train_size:]
            
            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]
            
            # Initialize deterministic NN for this run
            model = DeterministicNN(
                architecture,
                input_shape=(X_train.shape[-1],),
                learning_rate=self.config.learning_rate,
                map=self.config.use_map,
                sigma=utils.calculate_sigma(X_train)
            )
            
            # Train model
            model.train(
                X_train, y_train,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size
            )
            
            # Make predictions
            y_pred = model.predict(X_test).squeeze()
            
            # Calculate metrics
            metrics = {
                'mse': utils.mse(y_pred, y_test),
                'mae': utils.mae(y_pred, y_test),
                'rmse': np.sqrt(utils.mse(y_pred, y_test)),
            }
                
                
            # Store results
            results['runs'][seed] = {
            'mse': [metrics['mse']],  # Single value as list to match PBNN format
            'nlpd': [np.nan],  # Placeholder to match PBNN format
            'coverage': [np.nan]  # Placeholder to match PBNN format
        }
        
            logger.info(
                f"Seed {seed} - RMSE: {np.sqrt(metrics['mse']):.4f}, "
                f"MAE: {metrics['mae']:.4f}, "
                f"Training points: {final_train_size}"
            )
                    
        return results

def save_results(results: Dict, config: ComparisonConfig, output_dir: Path) -> None:
    """Save comparison results and metadata."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"detnn_{config.activation}_epochs{config.epochs}_lr{config.learning_rate}_{timestamp}.pkl"
    
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Results saved to {output_file}")
    
    # Save metadata in JSON format
    metadata_file = output_file.with_suffix('.json')
    with open(metadata_file, 'w') as f:
        json.dump({
            'parameters': {k: str(v) if isinstance(v, Path) else v 
                         for k, v in asdict(config).items()},
            'timestamp': datetime.datetime.now().isoformat(),
            'results_file': output_file.name
        }, f, indent=2)
    logger.info(f"Metadata saved to {metadata_file}")


def main():
    """Main execution function."""
    config = ComparisonConfig(
        hidden_dims=[8, 8, 8, 8],
        learning_rate=5e-3,
        epochs=2000,
        batch_size=16,
        seeds=[1, 2, 3, 4, 5],
        use_map=True,
        output_dir=Path("results/detnn8888"),
        pbnn_results_file=Path("results/pbnn8888/esol_probdense0-dense4_steps200_epochs2000_lr0.005_20241207_015431.pkl"),
        activation='silu'  # Default value, can be changed to 'tanh'
    )
    
    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load and preprocess data
        data = np.load("esol.npz")
        X, y = data["features"], data["targets"]
        
        # Run comparison
        comparator = ModelComparator(config)
        results = comparator.run_comparison(X, y)
        
        # Save results - pass config
        save_results(results, config, config.output_dir)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()