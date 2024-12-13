#!/usr/bin/env python3

import sys
sys.path.append("../..")

import numpy as np
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Literal, Optional
from pathlib import Path
import pickle
import json
import datetime
from sklearn.ensemble import RandomForestRegressor

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
    model_type: Literal['nn', 'rf']  # Added model type selection
    hidden_dims: List[int]
    learning_rate: float
    epochs: int
    batch_size: int
    seeds: List[int]
    use_map: bool
    output_dir: Path
    pbnn_results_file: Path
    activation: Literal['silu', 'tanh'] = 'silu'
    # Random Forest specific parameters with more optimal defaults for regression
    n_estimators: int = 500  # Increased from 100 for better performance
    max_depth: Optional[int] = 20  # Added limit to prevent overfitting
    min_samples_split: int = 5  # Increased from 2 for more robust splits
    min_samples_leaf: int = 3  # Increased from 1 for better generalization
    max_features: Optional[str] = 'sqrt'  # Added feature selection strategy
    bootstrap: bool = True  # Enable bootstrapping for better generalization

class ModelComparator:
    """Handles comparison between PBNN and deterministic models (NN or RF)."""
    
    def __init__(self, config: ComparisonConfig):
        self.config = config
        
    def load_pbnn_results(self) -> Dict:
        """Load results from PBNN active learning experiment."""
        with open(self.config.pbnn_results_file, 'rb') as f:
            return pickle.load(f)
    
    def create_model(self):
        """Create either a neural network or random forest model based on config."""
        if self.config.model_type == 'nn':
            architecture = FlaxMLP(
                hidden_dims=self.config.hidden_dims,
                target_dim=1,
                activation=self.config.activation
            )
            return lambda X_train, y_train: DeterministicNN(
                architecture,
                input_shape=(X_train.shape[-1],),
                learning_rate=self.config.learning_rate,
                map=self.config.use_map,
                sigma=utils.calculate_sigma(X_train)
            )
        else:  # random forest
            return lambda X_train, y_train: RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                max_features=self.config.max_features,
                bootstrap=self.config.bootstrap,
                random_state=None,  # We'll set this per run
                n_jobs=-1  # Use all available cores
            )
    
    def train_and_predict(self, model, X_train, y_train, X_test):
        """Train model and make predictions based on model type."""
        if self.config.model_type == 'nn':
            model.train(
                X_train, y_train,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size
            )
            return model.predict(X_test).squeeze()
        else:  # random forest
            model.fit(X_train, y_train)
            return model.predict(X_test)
    
    def run_comparison(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Run comparison experiments for each seed."""
        pbnn_results = self.load_pbnn_results()
        results = {
            'runs': {},
            'model_info': {}
        }
        
        model_creator = self.create_model()
        
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
            
            # Initialize model for this run
            if self.config.model_type == 'rf':
                model = model_creator(X_train, y_train)
                model.random_state = seed  # Set random state for reproducibility
            else:
                model = model_creator(X_train, y_train)
            
            # Train model and make predictions
            y_pred = self.train_and_predict(model, X_train, y_train, X_test)
            
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
                'coverage': [np.nan],  # Placeholder to match PBNN format
                'train_size': final_train_size
            }
            
            # For RF, store feature importances
            if self.config.model_type == 'rf':
                results['model_info'][seed] = {
                    'feature_importances': model.feature_importances_.tolist()
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
    model_prefix = "rf" if config.model_type == "rf" else f"detnn_{config.activation}"
    output_file = output_dir / f"{model_prefix}_epochs{config.epochs}_lr{config.learning_rate}_{timestamp}.pkl"
    
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
        model_type='rf',  # 'rf' for Random Forest, 'nn' for deteministic neural net
        hidden_dims=[8, 8, 8, 8],  
        learning_rate=5e-3,      
        epochs=2000,              
        batch_size=None,            
        seeds=[1, 2, 3, 4, 5],
        use_map=True,             
        output_dir=Path("results/rf"),
        pbnn_results_file=Path("results/pbnn8888/esol_probdense0-dense4_steps200_epochs2000_lr0.005_20241207_015431.pkl"),
        
        n_estimators=500,      
        max_depth=20,     
        min_samples_split=5,    
        min_samples_leaf=3,     
        max_features='sqrt',
        bootstrap=True
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
        
        # Save results
        save_results(results, config, config.output_dir)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()