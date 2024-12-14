#!/usr/bin/env python3

import datetime
import logging
import argparse
from pathlib import Path
import numpy as np
import torch
import gpytorch
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DKLConfig:
    """Configuration for DKL comparison experiments."""
    latent_dims: List[int]  # Different latent dimensions to test
    seeds: List[int]
    exploration_steps: int
    num_epochs: int
    learning_rate: float
    output_dir: Path
    input_file: Path
    activation: str = "tanh"  # Default activation function
    experiment_name: str = "dkl_comparison"

    def __post_init__(self):
        """Validate configuration parameters."""
        valid_activations = {"tanh", "silu", "relu"}
        if self.activation.lower() not in valid_activations:
            raise ValueError(
                f"Activation must be one of {valid_activations}, "
                f"got {self.activation}"
            )

    def get_output_filename(self, latent_dim: int) -> Path:
        """Generate output filename for specific latent dimension."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.output_dir / f"{self.experiment_name}_latent{latent_dim}_{self.activation}_{timestamp}.pkl"

class DKLFeatureExtractor(torch.nn.Sequential):
    """Neural network feature extractor for DKL matching the PBNN architecture."""
    
    def __init__(self, input_dim: int, latent_dim: int, activation: str = "tanh"):
        super().__init__()
        
        # Define activation function mapping
        activation_map = {
            "tanh": torch.nn.Tanh,
            "relu": torch.nn.ReLU,
            "silu": torch.nn.SiLU
        }
        
        if activation.lower() not in activation_map:
            raise ValueError(
                f"Activation must be one of {list(activation_map.keys())}, "
                f"got {activation}"
            )
            
        act_fn = activation_map[activation.lower()]
        
        # Build network with specified activation function
        self.add_module('linear1', torch.nn.Linear(input_dim, 8))
        self.add_module('act1', act_fn())
        self.add_module('linear2', torch.nn.Linear(8, 8))
        self.add_module('act2', act_fn())
        self.add_module('linear3', torch.nn.Linear(8, 8))
        self.add_module('act3', act_fn())
        self.add_module('linear4', torch.nn.Linear(8, 8))
        self.add_module('act4', act_fn())
        self.add_module('linear_latent', torch.nn.Linear(8, latent_dim))


class GPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, feature_extractor, latent_dim, likelihood):
            super(GPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim))
                
            self.feature_extractor = feature_extractor

            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        def forward(self, x):
            # We're first putting our data through a deep net (feature extractor)
            projected_x = self.feature_extractor(x)
            projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

class DKLExplorer:
    """Handles DKL-based active learning exploration."""
    
    def __init__(self, config: DKLConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def init_model(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        latent_dim: int
    ) -> Tuple[GPModel, gpytorch.likelihoods.GaussianLikelihood]:
        """Initialize DKL model with specified latent dimension."""
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        feature_extractor = DKLFeatureExtractor(
            X_train.shape[1],
            latent_dim,
            activation=self.config.activation
        )
        model = GPModel(X_train, y_train, feature_extractor, latent_dim, likelihood)
        
        return model.to(self.device), likelihood.to(self.device)
    
    def train_model(
        self,
        model: GPModel,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        train_x: torch.Tensor,
        train_y: torch.Tensor
    ) -> None:
        """Train the DKL model."""
        model.train()
        likelihood.train()
        
        optimizer = torch.optim.Adam([
            {'params': model.feature_extractor.parameters(), 'lr': self.config.learning_rate},
            {'params': model.covar_module.parameters(), 'lr': self.config.learning_rate},
            {'params': model.mean_module.parameters(), 'lr': self.config.learning_rate},
            {'params': likelihood.parameters(), 'lr': self.config.learning_rate}
        ])
        
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        for _ in range(self.config.num_epochs):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
    
    def predict(
        self,
        model: GPModel,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions with uncertainty."""
        model.eval()
        likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(X))
            return observed_pred.mean, observed_pred.variance
    
    def calculate_metrics(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        variance: torch.Tensor,
        y_scaler: StandardScaler
    ) -> Dict[str, float]:
        """Calculate performance metrics with rescaled values."""
        # Rescale predictions and true values back to original scale
        y_true_orig = torch.FloatTensor(y_scaler.inverse_transform(
            y_true.cpu().numpy().reshape(-1, 1)
        )).to(self.device)
        y_pred_orig = torch.FloatTensor(y_scaler.inverse_transform(
            y_pred.cpu().numpy().reshape(-1, 1)
        )).to(self.device)
        
        # Scale variance back to original scale
        variance_orig = variance * (y_scaler.scale_[0] ** 2)
        
        # Calculate metrics in original scale
        mse = torch.mean((y_true_orig - y_pred_orig) ** 2).item()
        mae = torch.mean(torch.abs(y_true_orig - y_pred_orig)).item()
        
        # Calculate NLPD in original scale
        nlpd = 0.5 * torch.mean(
            torch.log(2 * np.pi * variance_orig) + 
            (y_true_orig - y_pred_orig) ** 2 / variance_orig
        ).item()
        
        # Calculate 95% coverage in original scale
        z_score = 1.96
        lower = y_pred_orig - z_score * torch.sqrt(variance_orig)
        upper = y_pred_orig + z_score * torch.sqrt(variance_orig)
        coverage = torch.mean(
            ((y_true_orig >= lower) & (y_true_orig <= upper)).float()
        ).item()
        
        return {
            'mse': mse,
            'mae': mae,
            'nlpd': nlpd,
            'coverage': coverage
        }
    
    def run_exploration(
        self,
        X: np.ndarray,
        y: np.ndarray,
        latent_dim: int,
        seed: int,
        y_scaler: StandardScaler
    ) -> Dict[str, List[float]]:
        """Run active learning exploration process."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Convert to PyTorch tensors
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).to(self.device)
        
        # Initial split
        X_measured, X_unmeasured, y_measured, y_unmeasured = train_test_split(
            X, y, test_size=0.95, random_state=seed
        )
        
        metrics = {metric: [] for metric in ['mse', 'mae', 'nlpd', 'coverage']}
        
        for step in range(self.config.exploration_steps):
            logger.info(f'Latent dim {latent_dim}, Seed {seed}, Step {step}')
            
            try:
                # Initialize and train model
                model, likelihood = self.init_model(X_measured, y_measured, latent_dim)
                self.train_model(model, likelihood, X_measured, y_measured)
                
                # Make predictions
                pred_mean, pred_var = self.predict(model, likelihood, X_unmeasured)
                
                # Calculate metrics with rescaling
                step_metrics = self.calculate_metrics(
                    y_unmeasured, pred_mean, pred_var, y_scaler
                )
                
                for metric, value in step_metrics.items():
                    metrics[metric].append(value)
                
                # Select next point
                next_idx = pred_var.argmax()
                
                # Update datasets
                X_measured = torch.cat([
                    X_measured,
                    X_unmeasured[next_idx].unsqueeze(0)
                ])
                y_measured = torch.cat([
                    y_measured,
                    y_unmeasured[next_idx].unsqueeze(0)
                ])
                
                X_unmeasured = torch.cat([
                    X_unmeasured[:next_idx],
                    X_unmeasured[next_idx + 1:]
                ])
                y_unmeasured = torch.cat([
                    y_unmeasured[:next_idx],
                    y_unmeasured[next_idx + 1:]
                ])
                
                logger.info(
                    f"RMSE: {np.sqrt(step_metrics['mse']):.4f}, "
                    f"MAE: {step_metrics['mae']:.4f}, "
                    f"NLPD: {step_metrics['nlpd']:.4f}, "
                    f"Coverage: {step_metrics['coverage']:.4f}\n"
                )
                
            except Exception as e:
                logger.error(f"Error in exploration step {step}: {str(e)}")
                raise
                
        return metrics


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="DKL Comparison Framework")
    parser.add_argument(
        "--latent-dims",
        nargs="+",
        type=int,
        default=[2,],
        help="List of latent dimensions to test"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Random seeds for experiments"
    )
    parser.add_argument(
        "--exploration-steps",
        type=int,
        default=200,
        help="Number of active learning steps"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs per step"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for optimization"
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=["tanh", "relu", "silu"],
        default="silu",
        help="Activation function to use"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dkl8888"),
        help="Directory to save results"
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("fatigue.npz"),
        help="Input data file path"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="fatigue_dkl",
        help="Name for this experiment"
    )
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    config = DKLConfig(**vars(args))
    
    try:
        # Load data
        data = np.load(config.input_file)
        X, y = data["features"], data["targets"]

        # Scale both X and y
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        
        X = x_scaler.fit_transform(X)
        y = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        
        explorer = DKLExplorer(config)
        
        # Run experiments for each latent dimension
        for latent_dim in config.latent_dims:
            results = {
                'config': asdict(config),
                'latent_dim': latent_dim,
                'runs': {}
            }
            
            for seed in config.seeds:
                results['runs'][seed] = explorer.run_exploration(
                    X, y, latent_dim, seed, y_scaler
                )
            
            # Save results and metadata
            output_file = config.get_output_filename(latent_dim)
            with open(output_file, "wb") as f:
                pickle.dump(results, f)
            
            metadata_file = output_file.with_suffix('.json')
            with open(metadata_file, 'w') as f:
                json.dump({
                    'parameters': {
                        **{k: str(v) if isinstance(v, Path) else v 
                           for k, v in asdict(config).items()},
                        'latent_dim': latent_dim
                    },
                    'timestamp': datetime.datetime.now().isoformat(),
                    'results_file': output_file.name
                }, f, indent=2)
                
            logger.info(f"Results saved to {output_file}")
            logger.info(f"Metadata saved to {metadata_file}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()