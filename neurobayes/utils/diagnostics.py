from typing import Dict
import jax.numpy as jnp
from numpyro.diagnostics import split_gelman_rubin as sgr

class MCMCDiagnostics:
    """
    Lightweight diagnostics for BNN MCMC samples with layer-by-layer analysis
    """
    
    @staticmethod
    def analyze_samples(mcmc_samples: Dict[str, jnp.ndarray],
                       threshold_rhat: float = 1.1) -> Dict:
        """Analyze MCMC samples layer-by-layer using NumPyro's split_gelman_rubin"""
        layer_params = {}
        
        # Group by layers and calculate R-hats
        for param_name, samples in mcmc_samples.items():
            if not ('kernel' in param_name or 'bias' in param_name):
                continue
            
            # Handle both full BNN and partial BNN parameter naming
            if param_name.startswith('nn/'):
                # Full BNN case: 'nn/MLPLayerModule_0.Dense0.bias'
                layer_name = param_name.split('/')[1].split('.')[0]
            else:
                # Partial BNN case: 'Dense0/Dense0.bias'
                layer_name = param_name.split('/')[0]
                
            if layer_name not in layer_params:
                layer_params[layer_name] = {}
                
            # Calculate R-hat using split_gelman_rubin
            rhats = sgr(samples).flatten()
            layer_params[layer_name][param_name] = rhats

        results = {
            'overall': {},
            'by_layer': {},
            'worst_params': []
        }
        
        # Rest of the analysis stays the same...
        all_rhats = []
        for layer_name, params in layer_params.items():
            layer_rhats = jnp.concatenate([rhats for rhats in params.values()])
            all_rhats.extend(layer_rhats)
            
            results['by_layer'][layer_name] = {
                'rhat_mean': float(layer_rhats.mean()),
                'rhat_max': float(layer_rhats.max()),
                'percent_bad': float((layer_rhats > threshold_rhat).mean() * 100),
                'num_params': len(layer_rhats)
            }
            
            # Track worst parameters
            for param_name, rhats in params.items():
                if (rhats > threshold_rhat).any():
                    results['worst_params'].append({
                        'param': param_name,
                        'max_rhat': float(rhats.max()),
                        'percent_bad': float((rhats > threshold_rhat).mean() * 100)
                    })

        all_rhats = jnp.array(all_rhats)
        results['overall'] = {
            'rhat_mean': float(all_rhats.mean()),
            'rhat_max': float(all_rhats.max()),
            'percent_bad': float((all_rhats > threshold_rhat).mean() * 100),
            'total_params': len(all_rhats)
        }
        
        return results
    
    @staticmethod
    def print_summary(results: Dict):
        """Print formatted diagnostic summary"""
        print("\n=== Overall MCMC Diagnostics ===")
        print(f"Total parameters: {results['overall']['total_params']}")
        print(f"Mean R-hat: {results['overall']['rhat_mean']:.3f}")
        print(f"Percent bad R-hat: {results['overall']['percent_bad']:.1f}%")
        
        print("\n=== Layer-by-Layer Analysis ===")
        for layer, stats in results['by_layer'].items():
            print(f"\nLayer: {layer}")
            print(f"  Parameters: {stats['num_params']}")
            print(f"  Mean R-hat: {stats['rhat_mean']:.3f}")
            print(f"  Percent bad R-hat: {stats['percent_bad']:.1f}%")
        
        if results['worst_params']:
            print("\n=== Worst Parameters ===")
            for param in sorted(results['worst_params'], 
                            key=lambda x: x['max_rhat'], reverse=True):
                print(f"\n{param['param']}:")
                print(f"  Percent bad: {param['percent_bad']:.1f}%")
                print(f"  Max R-hat: {param['max_rhat']:.3f}")

    
    @staticmethod
    def run_diagnostics(mcmc_samples: Dict[str, jnp.ndarray], threshold_rhat: float = 1.15):
        """Analyze and print MCMC diagnostics summary in one go"""
        results = MCMCDiagnostics.analyze_samples(mcmc_samples, threshold_rhat)
        MCMCDiagnostics.print_summary(results)
        return results
