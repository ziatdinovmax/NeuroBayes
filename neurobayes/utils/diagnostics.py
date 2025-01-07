from typing import Dict
import jax.numpy as jnp
from numpyro.diagnostics import split_gelman_rubin as sgr

class MCMCDiagnostics:
    """Lightweight diagnostics for BNN MCMC samples with layer analysis"""
    
    def __init__(self, threshold_rhat: float = 1.15):
        self.threshold_rhat = threshold_rhat
        self.results = None

    def analyze_samples(self, mcmc_samples: Dict[str, jnp.ndarray]) -> Dict:
        """Analyze MCMC samples layer by layer using NumPyro's split_gelman_rubin"""
        layer_params = {}
        
        # Group by layers and calculate R-hats
        for param_name, samples in mcmc_samples.items():
            if not ('kernel' in param_name or 'bias' in param_name):
                continue
            
            # Handle both full BNN and partial BNN parameter naming
            if param_name.startswith('nn/'):
                layer_name = param_name.split('/')[1].split('.')[0]
            else:
                layer_name = param_name.split('/')[0]
                
            if layer_name not in layer_params:
                layer_params[layer_name] = {}
                
            rhats = sgr(samples).flatten()
            layer_params[layer_name][param_name] = rhats

        self.results = {
            'overall': {},
            'by_layer': {},
            'worst_params': []
        }
        
        all_rhats = []
        for layer_name, params in layer_params.items():
            layer_rhats = jnp.concatenate([rhats for rhats in params.values()])
            all_rhats.extend(layer_rhats)
            
            self.results['by_layer'][layer_name] = {
                'rhat_mean': float(layer_rhats.mean()),
                'rhat_max': float(layer_rhats.max()),
                'percent_bad': float((layer_rhats > self.threshold_rhat).mean() * 100),
                'num_params': len(layer_rhats)
            }
            
            for param_name, rhats in params.items():
                if (rhats > self.threshold_rhat).any():
                    self.results['worst_params'].append({
                        'param': param_name,
                        'max_rhat': float(rhats.max()),
                        'percent_bad': float((rhats > self.threshold_rhat).mean() * 100)
                    })

        all_rhats = jnp.array(all_rhats)
        self.results['overall'] = {
            'rhat_mean': float(all_rhats.mean()),
            'rhat_max': float(all_rhats.max()),
            'percent_bad': float((all_rhats > self.threshold_rhat).mean() * 100),
            'total_params': len(all_rhats)
        }
        
        return self.results

    def print_summary(self):
        """Print formatted diagnostic summary"""
        if self.results is None:
            raise ValueError("No results available. Run analyze_samples first.")
            
        print("\n=== Overall MCMC Diagnostics ===")
        print(f"Total parameters: {self.results['overall']['total_params']}")
        print(f"Mean R-hat: {self.results['overall']['rhat_mean']:.3f}")
        print(f"Percent bad R-hat: {self.results['overall']['percent_bad']:.1f}%")
        
        print("\n=== Layer-by-Layer Analysis ===")
        for layer, stats in self.results['by_layer'].items():
            print(f"\nLayer: {layer}")
            print(f"  Parameters: {stats['num_params']}")
            print(f"  Mean R-hat: {stats['rhat_mean']:.3f}")
            print(f"  Percent bad R-hat: {stats['percent_bad']:.1f}%")
        
        if self.results['worst_params']:
            print("\n=== Worst Parameters ===")
            for param in sorted(self.results['worst_params'], 
                            key=lambda x: x['max_rhat'], reverse=True):
                print(f"\n{param['param']}:")
                print(f"  Percent bad: {param['percent_bad']:.1f}%")
                print(f"  Max R-hat: {param['max_rhat']:.3f}")

    def run_diagnostics(self, mcmc_samples: Dict[str, jnp.ndarray]) -> Dict:
        """Analyze and print MCMC diagnostics summary in one go"""
        self.analyze_samples(mcmc_samples)
        self.print_summary()
        return self.results