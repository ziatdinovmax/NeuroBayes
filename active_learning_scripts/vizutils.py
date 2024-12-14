import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(file_path: str):
    """Load results from pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def get_model_name(file_path):
    """Extract and format model name and configuration from filename."""
    name = Path(file_path).stem
    
    if 'dkl' in name.lower():
        # Parse DKL filename
        parts = name.split('_')
        latent_dim = next(p for p in parts if 'latent' in p)
        activation = next(p for p in parts if p in ['tanh', 'relu', 'silu'])
        return f"DKL (latent={latent_dim.replace('latent', '')}, {activation})"
    elif 'gp' in name.lower():
        # Parse GP filename
        parts = name.split('_')
        kernel = next(p for p in parts if 'kernel' in p)
        return f"GP ({kernel.replace('kernel', '')})"
    else:
        # Parse PBNN filename
        prob_part = name.split('_')[1]
        layers = prob_part.replace('prob', '').split('-')
        layer_names = []
        for l in layers:
            dense_num = l.replace('dense', '')
            #layer_names.append(f"Dense {dense_num}")
            layer_names.append(f"{dense_num}")
        return f"PBNN ({', '.join(layer_names)})"
    
def compute_statistics(results):
    """Compute mean and std for each metric across seeds."""
    metrics = {}
    
    # Handle both PBNN and DKL result formats
    if 'latent_dim' in results:  # DKL results
        runs_data = results['runs']
    else:  # PBNN results
        runs_data = results['runs']
    
    for metric in ['mse', 'nlpd', 'coverage']:
        values = np.array([runs_data[seed][metric] for seed in runs_data.keys()])
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
            
        metrics[metric] = {'mean': mean, 'std': std}
    
    return metrics


def add_detnn_to_plot(detnn_file: str, axes, metric_indices = {'mse': 0, 'nlpd': 1, 'coverage': 2}):
    """
    Add deterministic NN results as horizontal lines to existing plots.
    """
    # Load deterministic NN results
    with open(detnn_file, 'rb') as f:
        detnn_results = pickle.load(f)
    
    # Create legend label
    model_name = "DetNN"
    
    # Calculate mean and std for each metric across seeds
    metrics = {}
    for metric in ['mse']:  # only MSE for detNN
        values = np.array([
            detnn_results['runs'][seed][metric][0]  # take first (only) value from list
            for seed in detnn_results['runs']
        ])
        metrics[metric] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    # Get number of steps from the corresponding axis
    n_steps = len(axes[0].get_lines()[0].get_xdata())
    
    # Plot horizontal lines for each metric
    detnn_color = '#1f77b4'  # blue
    
    # Plot RMSE (transformed from MSE)
    mean_rmse = np.sqrt(metrics['mse']['mean'])
    std_rmse = metrics['mse']['std'] / (2 * np.sqrt(metrics['mse']['mean']))
    
    x = np.arange(n_steps)
    mean_line = np.full(n_steps, mean_rmse)
    upper_line = np.full(n_steps, mean_rmse + std_rmse)
    lower_line = np.full(n_steps, mean_rmse - std_rmse)
    
    # Add to MSE plot
    ax = axes[metric_indices['mse']]
    line = ax.plot(x, mean_line, '--', color=detnn_color, 
                  label=model_name, linewidth=2.5, alpha=0.8)[0]
    ax.fill_between(x, lower_line, upper_line, 
                   color=detnn_color, alpha=0.2)
    
    # Print final values
    print(f"\n{model_name} final values (mean ± std):")
    print(f"RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")



def plot_comparison(pbnn_files, bnn_file=None, dkl_files=None, gp_files=None, detnn_file=None, save_path=None):
    """Plots metrics comparing PBNN, DKL, and GP results."""
    plt.style.use('seaborn-v0_8-paper')
    
    pbnn_base_colors = [
        '#e31a1c',  # red
        '#6a3d9a',  # purple
        '#ff7f00',  # orange
        '#fb9a99'   # pink
    ]

    pbnn_colors = pbnn_base_colors #plt.cm.Reds(np.linspace(0.6, 0.9, len(pbnn_files)))
    if dkl_files:
        dkl_colors = plt.cm.Blues(np.linspace(0.6, 0.9, len(dkl_files)))
    if gp_files:
        gp_colors = plt.cm.Greens(np.linspace(0.6, 0.9, len(gp_files)))
    
    
    titles = {
        'mse': 'RMSE over Time',
        'nlpd': 'NLPD over Time',
        'coverage': 'Coverage over Time'
    }
    
    ylabels = {
        'mse': 'Root Mean Square Error',
        'nlpd': 'Negative Log Predictive Density',
        'coverage': 'Coverage Probability'
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    final_values = {}
    
    # Plot PBNN results
    for file_path, color in zip(pbnn_files, pbnn_colors):
        results = load_results(file_path)
        metrics = compute_statistics(results)
        model_name = get_model_name(file_path)
        final_values[model_name] = {}
        
        for (metric, ax) in zip(metrics, axes):
            mean = metrics[metric]['mean']
            std = metrics[metric]['std']
            steps = np.arange(len(mean))
            
            if metric == 'mse':
                mean_rmse = np.sqrt(mean)
                std_rmse = std / (2 * np.sqrt(mean))
                plot_mean, plot_std = mean_rmse, std_rmse
            else:
                plot_mean, plot_std = mean, std
            
            line = ax.plot(steps, plot_mean, label=model_name, color=color, 
                         linewidth=3.5, alpha=0.8)[0]
            ax.fill_between(steps, plot_mean - plot_std, plot_mean + plot_std, 
                          color=line.get_color(), alpha=0.3)
            
            final_values[model_name][metric] = (plot_mean[-1], plot_std[-1])
    
    if dkl_files: # Plot DKL results
        for file_path, color in zip(dkl_files, dkl_colors):
            results = load_results(file_path)
            metrics = compute_statistics(results)
            model_name = get_model_name(file_path)
            final_values[model_name] = {}
            
            for (metric, ax) in zip(metrics, axes):
                mean = metrics[metric]['mean']
                std = metrics[metric]['std']
                steps = np.arange(len(mean))
                
                if metric == 'mse':
                    mean_rmse = np.sqrt(mean)
                    std_rmse = std / (2 * np.sqrt(mean))
                    plot_mean, plot_std = mean_rmse, std_rmse
                else:
                    plot_mean, plot_std = mean, std
                
                line = ax.plot(steps, plot_mean, label=model_name, color=color, 
                            linewidth=2.5, alpha=0.8, linestyle='--')[0]
                ax.fill_between(steps, plot_mean - plot_std, plot_mean + plot_std, 
                            color=line.get_color(), alpha=0.3)
                
                final_values[model_name][metric] = (plot_mean[-1], plot_std[-1])
    
    if gp_files: # Plot GP results
        for file_path, color in zip(gp_files, gp_colors):
            results = load_results(file_path)
            metrics = compute_statistics(results)
            model_name = get_model_name(file_path)
            final_values[model_name] = {}
            
            for (metric, ax) in zip(metrics, axes):
                mean = metrics[metric]['mean']
                std = metrics[metric]['std']
                steps = np.arange(len(mean))
                
                if metric == 'mse':
                    mean_rmse = np.sqrt(mean)
                    std_rmse = std / (2 * np.sqrt(mean))
                    plot_mean, plot_std = mean_rmse, std_rmse
                else:
                    plot_mean, plot_std = mean, std
                
                line = ax.plot(steps, plot_mean, label=model_name, color=color, 
                            linewidth=4.5, alpha=0.8, linestyle=':')[0]
                ax.fill_between(steps, plot_mean - plot_std, plot_mean + plot_std, 
                            color=line.get_color(), alpha=0.3)
                
                final_values[model_name][metric] = (plot_mean[-1], plot_std[-1])
        

    if detnn_file:
        add_detnn_to_plot(detnn_file, axes)

    if bnn_file:
        results = load_results(bnn_file)
        metrics = compute_statistics(results)
        model_name = 'Full BNN'
        final_values[model_name] = {}
        
        for (metric, ax) in zip(metrics, axes):
            mean = metrics[metric]['mean']
            std = metrics[metric]['std']
            steps = np.arange(len(mean))
            
            if metric == 'mse':
                mean_rmse = np.sqrt(mean)
                std_rmse = std / (2 * np.sqrt(mean))
                plot_mean, plot_std = mean_rmse, std_rmse
            else:
                plot_mean, plot_std = mean, std
            
            line = ax.plot(steps, plot_mean, label=model_name, color='k', 
                            linewidth=2.5, alpha=0.8, linestyle=':')[0]
            ax.fill_between(steps, plot_mean - plot_std, plot_mean + plot_std, 
                            color=line.get_color(), alpha=0.3)
            
            final_values[model_name][metric] = (plot_mean[-1], plot_std[-1])

        
    for metric, ax in zip(titles.keys(), axes):
        ax.set_title(titles[metric], fontsize=16, pad=10)
        ax.set_xlabel('Active Learning Steps', fontsize=16)
        ax.set_ylabel(ylabels[metric], fontsize=16)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        # if metric == 'coverage':
        #     ax.set_ylim(0.7, 1.1)
        #     ax.set_yticks(np.arange(0, 1.1, 0.2))
        
        if metric == 'nlpd':
            ax.set_yscale('symlog')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        legend = ax.legend(loc='best', frameon=True, framealpha=0.9, 
                         fontsize=12, title='Models')
        legend.get_title().set_fontsize(14)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('none')
    
    

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Print final values
    print("\nFinal values (mean ± std):")
    for model_name in final_values:
        print(f"\n{model_name}:")
        for metric in ['mse', 'nlpd', 'coverage']:
            mean, std = final_values[model_name][metric]
            if metric == 'mse':
                print(f"  RMSE: {mean:.4f} ± {std:.4f}") 
            else:
                print(f"  {metric.upper()}: {mean:.4f} ± {std:.4f}")


def plot_final_results(pbnn_files, dkl_files, gp_files, save_path=None):
    """Creates bar plots of final metric values for all models."""
    plt.style.use('seaborn-v0_8-paper')
    
    # Load and process results
    final_values = {}
    
    # Process PBNN results
    for file_path in pbnn_files:
        results = load_results(file_path)
        metrics = compute_statistics(results)
        model_name = get_model_name(file_path)
        final_values[model_name] = {
            metric: (metrics[metric]['mean'][-1], metrics[metric]['std'][-1])
            for metric in ['mse', 'nlpd', 'coverage']
        }
    
    # Process DKL results
    for file_path in dkl_files:
        results = load_results(file_path)
        metrics = compute_statistics(results)
        model_name = get_model_name(file_path)
        final_values[model_name] = {
            metric: (metrics[metric]['mean'][-1], metrics[metric]['std'][-1])
            for metric in ['mse', 'nlpd', 'coverage']
        }
    
    # Process GP results
    for file_path in gp_files:
        results = load_results(file_path)
        metrics = compute_statistics(results)
        model_name = get_model_name(file_path)
        final_values[model_name] = {
            metric: (metrics[metric]['mean'][-1], metrics[metric]['std'][-1])
            for metric in ['mse', 'nlpd', 'coverage']
        }
    
    # Prepare data for plotting
    models = list(final_values.keys())
    metrics = ['mse', 'nlpd', 'coverage']
    titles = {
        'mse': 'Final RMSE',
        'nlpd': 'Final NLPD',
        'coverage': 'Final Coverage'
    }
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Set bar positions
    x = np.arange(len(models))
    bar_width = 0.8
    
    for metric, ax in zip(metrics, axes):
        # Extract values and errors for current metric
        values = [final_values[model][metric][0] for model in models]
        errors = [final_values[model][metric][1] for model in models]
        
        # Take square root for MSE values to show RMSE
        if metric == 'mse':
            values = [np.sqrt(v) for v in values]
            errors = [e / (2 * np.sqrt(v)) for e, v in zip(errors, values)]  # error propagation
        
        # Create bars
        bars = ax.bar(x, values, bar_width, 
                     yerr=errors, 
                     capsize=5,
                     error_kw={'capthick': 2})
        
        # Color bars based on model type
        for i, bar in enumerate(bars):
            if 'PBNN' in models[i]:
                bar.set_color(plt.cm.Reds(0.6))
            elif 'DKL' in models[i]:
                bar.set_color(plt.cm.Blues(0.6))
            else:  # GP
                bar.set_color(plt.cm.Greys(0.6))
        
        # Customize plot
        ax.set_title(titles[metric], fontsize=16, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        
        if metric == 'coverage':
            ax.set_ylim(0, 1)
        # elif metric == 'nlpd':
        #     ax.set_yscale('log')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Adjust tick label size
        ax.tick_params(axis='both', which='major', labelsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes