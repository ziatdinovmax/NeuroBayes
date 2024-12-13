#!/usr/bin/env python3

import sys
import logging
import argparse
from pathlib import Path
import numpy as np
import torch
import gpytorch
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class GPConfig:
    """Configuration for GP experiments."""
    kernels: List[str]
    seeds: List[int]
    exploration_steps: int
    num_epochs: int
    learning_rate: float
    output_dir: Path
    input_file: Path
    experiment_name: str = "gp_comparison"

    def __post_init__(self):
        """Validate configuration parameters."""
        valid_kernels = {"rbf", "matern52"}
        for kernel in self.kernels:
            if kernel.lower() not in valid_kernels:
                raise ValueError(
                    f"Kernel must be one of {valid_kernels}, "
                    f"got {kernel}"
                )

    def get_output_filename(self, kernel: str) -> Path:
        """Generate output filename for specific kernel."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.output_dir / f"{self.experiment_name}_kernel{kernel}_{timestamp}.pkl"

class GPModel(gpytorch.models.ExactGP):
    """Gaussian Process model with configurable kernel."""
    
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        kernel: str = "rbf"
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Configure kernel
        if kernel == "rbf":
            self.covar_module = gpytorch.kernels.RBFKernel()
        elif kernel == "matern52":
            self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5)
        else:
            raise ValueError(f"Unsupported kernel: {kernel}")
        
        # Add scale kernel for ARD behavior
        self.covar_module = gpytorch.kernels.ScaleKernel(self.covar_module)
        
    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class GPExplorer:
    """Handles GP-based active learning exploration."""
    
    def __init__(self, config: GPConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def init_model(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        kernel: str
    ) -> Tuple[GPModel, gpytorch.likelihoods.GaussianLikelihood]:
        """Initialize GP model with specified kernel."""
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPModel(X_train, y_train, likelihood, kernel=kernel)
        
        return model.to(self.device), likelihood.to(self.device)
    
    def train_model(
        self,
        model: GPModel,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        train_x: torch.Tensor,
        train_y: torch.Tensor
    ) -> None:
        """Train the GP model."""
        model.train()
        likelihood.train()
        
        optimizer = torch.optim.Adam([
            {'params': model.parameters(), 'lr': self.config.learning_rate}
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
        variance: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        mse = torch.mean((y_true - y_pred) ** 2).item()
        mae = torch.mean(torch.abs(y_true - y_pred)).item()
        
        # Calculate NLPD
        nlpd = 0.5 * torch.mean(
            torch.log(2 * np.pi * variance) + 
            (y_true - y_pred) ** 2 / variance
        ).item()
        
        # Calculate 95% coverage
        z_score = 1.96
        lower = y_pred - z_score * torch.sqrt(variance)
        upper = y_pred + z_score * torch.sqrt(variance)
        coverage = torch.mean(
            ((y_true >= lower) & (y_true <= upper)).float()
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
        kernel: str,
        seed: int
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
            logger.info(f'Kernel {kernel}, Seed {seed}, Step {step}')
            
            try:
                # Initialize and train model
                model, likelihood = self.init_model(X_measured, y_measured, kernel)
                self.train_model(model, likelihood, X_measured, y_measured)
                
                # Make predictions
                pred_mean, pred_var = self.predict(model, likelihood, X_unmeasured)
                
                # Calculate metrics
                step_metrics = self.calculate_metrics(
                    y_unmeasured, pred_mean, pred_var
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
    parser = argparse.ArgumentParser(description="GP Active Learning Framework")
    parser.add_argument(
        "--kernels",
        nargs="+",
        type=str,
        default=["matern52"],
        help="List of kernels to test"
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
        default=1000,
        help="Number of training epochs per step"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for optimization"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/gp"),
        help="Directory to save results"
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("htem.npz"),
        help="Input data file path"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="gp_comparison",
        help="Name for this experiment"
    )
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    config = GPConfig(**vars(args))
    
    try:
        # Load data
        data = np.load(config.input_file)
        X, y = data["features"], data["targets"]
        
        x_scaler = StandardScaler()
        X = x_scaler.fit_transform(X)
        y = np.log10(y)
        
        explorer = GPExplorer(config)
        
        # Run experiments for each kernel
        for kernel in config.kernels:
            results = {
                'config': asdict(config),
                'kernel': kernel,
                'runs': {}
            }
            
            for seed in config.seeds:
                results['runs'][seed] = explorer.run_exploration(
                    X, y, kernel, seed
                )
            
            # Save results and metadata
            output_file = config.get_output_filename(kernel)
            with open(output_file, "wb") as f:
                pickle.dump(results, f)
            
            metadata_file = output_file.with_suffix('.json')
            with open(metadata_file, 'w') as f:
                json.dump({
                    'parameters': {
                        **{k: str(v) if isinstance(v, Path) else v 
                           for k, v in asdict(config).items()},
                        'kernel': kernel
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