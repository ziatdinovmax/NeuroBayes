# plot_utils.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from typing import Dict, List, Tuple, Optional

def load_experiment_results(file_path: Path) -> Tuple[Dict, float]:
    """Load results from a pickle file and extract sigma from metadata."""
    with open(file_path, 'rb') as f:
        results = pickle.load(f)
    
    metadata_file = file_path.with_suffix('.json')
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    sigma = float(metadata['parameters'].get('priors_sigma', 
                                           metadata['parameters'].get('map_sigma', 1.0)))
    return results, sigma

def process_results(results: Dict) -> Dict[str, np.ndarray]:
    """Process results to get mean and std for each metric across seeds."""
    metrics = {}
    for metric in ['mse', 'nlpd', 'coverage']:
        values = np.array([run[metric] for run in results['runs'].values()])
        metrics[f'{metric}_mean'] = np.mean(values, axis=0)
        metrics[f'{metric}_std'] = np.std(values, axis=0)
        
        if metric == 'mse':
            metrics['rmse_mean'] = np.sqrt(metrics['mse_mean'])
            metrics['rmse_std'] = 0.5 * metrics['mse_std'] / np.sqrt(metrics['mse_mean'])
    
    return metrics

def plot_sigma_comparison(
    results_dir: Path,
    rmse_ylim: Optional[Tuple[float, float]] = None,
    nlpd_ylim: Optional[Tuple[float, float]] = None,
    coverage_ylim: Optional[Tuple[float, float]] = None,
    rmse_symlog: bool = False,
    nlpd_symlog: bool = False,
    coverage_symlog: bool = False,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """Plot metrics comparison for different sigma values."""

    # Fixed color mapping for sigma values
    sigma_colors = {
        0.1: '#1f77b4',  # blue
        0.5: '#ff7f0e',  # orange
        1.0: '#2ca02c',  # green
        2.0: '#d62728',  # red
        5.0: '#9467bd',  # purple
        10.0: '#8c564b'  # brown
    }

    # Collect results
    all_results = []
    for file_path in results_dir.glob('*.pkl'):
        results, sigma = load_experiment_results(file_path)
        metrics = process_results(results)
        steps = np.arange(len(metrics['rmse_mean']))
        
        for metric in ['rmse', 'nlpd', 'coverage']:
            all_results.append({
                'sigma': sigma,
                'metric': metric,
                'steps': steps,
                'mean': metrics[f'{metric}_mean'],
                'std': metrics[f'{metric}_std']
            })
    
    # Create plots
    metrics = ['rmse', 'nlpd', 'coverage']
    ylims = {'rmse': rmse_ylim, 'nlpd': nlpd_ylim, 'coverage': coverage_ylim}
    symlog = {'rmse': rmse_symlog, 'nlpd': nlpd_symlog, 'coverage': coverage_symlog}
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    for idx, metric in enumerate(metrics):
        metric_results = [r for r in all_results if r['metric'] == metric]
        # Sort by sigma for consistent legend order
        metric_results.sort(key=lambda x: x['sigma'])
        
        for result in metric_results:
            color = sigma_colors.get(result['sigma'], '#17becf')
            axes[idx].plot(result['steps'], result['mean'], 
                         label=f'τ={result["sigma"]:.1f}',
                         color=color)
            axes[idx].fill_between(result['steps'],
                                 result['mean'] - result['std'],
                                 result['mean'] + result['std'],
                                 color=color,
                                 alpha=0.2)
        
        if ylims[metric]:
            axes[idx].set_ylim(ylims[metric])
        if symlog[metric]:
            axes[idx].set_yscale('symlog')
            
        axes[idx].set_xlabel('Active Learning Steps', fontsize=14)
        axes[idx].set_ylabel(metric.upper() if metric != 'coverage' else 'Coverage Probability', fontsize=14)
        legend = axes[idx].legend(fontsize=14)
        legend.set_title("Pre-trained prior width", prop={'size': 'large'})
        legend._legend_box.align = "center"
        legend.get_title().set_position((0.5, 1.1)) 
        title_text = metric.upper() if metric != 'coverage' else 'Coverage'
        axes[idx].set_title(f'{title_text} over Time', fontsize=14)
        axes[idx].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes

def plot_pretrain_comparison(
    pbnn_dir: Path,
    ptpbnn_dir: Path,
    sigma: float = 1.0,
    rmse_ylim: Optional[Tuple[float, float]] = None,
    nlpd_ylim: Optional[Tuple[float, float]] = None,
    coverage_ylim: Optional[Tuple[float, float]] = None,
    rmse_symlog: bool = False,
    nlpd_symlog: bool = False,
    coverage_symlog: bool = False,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """Plot comparison between pretrained and non-pretrained models."""

    sigma_colors = {
        0.1: '#1f77b4',  # blue
        0.5: '#ff7f0e',  # orange
        1.0: '#2ca02c',  # green
        2.0: '#d62728',  # red
        5.0: '#9467bd',  # purple
        10.0: '#8c564b'  # brown
    }
    # Get color for current sigma
    sigma_color = sigma_colors.get(sigma, '#17becf')

    # Collect results
    pretrained_results = []
    non_pretrained_results = []
    
    for file_path in ptpbnn_dir.glob('*.pkl'):
        results, file_sigma = load_experiment_results(file_path)
        if abs(file_sigma - sigma) < 1e-6:
            metrics = process_results(results)
            steps = np.arange(len(metrics['rmse_mean']))
            pretrained_results.append({
                'steps': steps,
                'metrics': metrics
            })
    
    for file_path in pbnn_dir.glob('*.pkl'):
        results, file_sigma = load_experiment_results(file_path)
        if abs(file_sigma - sigma) < 1e-6:
            metrics = process_results(results)
            steps = np.arange(len(metrics['rmse_mean']))
            non_pretrained_results.append({
                'steps': steps,
                'metrics': metrics
            })
    
    # Create plots
    metrics = ['rmse', 'nlpd', 'coverage']
    ylims = {'rmse': rmse_ylim, 'nlpd': nlpd_ylim, 'coverage': coverage_ylim}
    symlog = {'rmse': rmse_symlog, 'nlpd': nlpd_symlog, 'coverage': coverage_symlog}
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    for idx, metric in enumerate(metrics):
        if pretrained_results:
            mean = pretrained_results[0]['metrics'][f'{metric}_mean']
            std = pretrained_results[0]['metrics'][f'{metric}_std']
            steps = pretrained_results[0]['steps']
            
            axes[idx].plot(steps, mean, label=f'Pretrained Priors (τ={sigma})', color=sigma_color)
            axes[idx].fill_between(steps, mean - std, mean + std, 
                                 alpha=0.2, color=sigma_color)
        
        if non_pretrained_results:
            mean = non_pretrained_results[0]['metrics'][f'{metric}_mean']
            std = non_pretrained_results[0]['metrics'][f'{metric}_std']
            steps = non_pretrained_results[0]['steps']
            
            axes[idx].plot(steps, mean, label='Standard Priors', color='k')
            axes[idx].fill_between(steps, mean - std, mean + std, 
                                 alpha=0.2, color='k')
        
        if ylims[metric]:
            axes[idx].set_ylim(ylims[metric])
        if symlog[metric]:
            axes[idx].set_yscale('symlog')
            
        axes[idx].set_xlabel('Active Learning Steps', fontsize=14)
        axes[idx].set_ylabel(metric.upper() if metric != 'coverage' else 'Coverage Probability', fontsize=14)
        axes[idx].legend(fontsize=13)
        axes[idx].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes