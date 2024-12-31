from typing import Dict, Callable, Tuple
import numpy as np


def build_swa_schedule(protocol: Dict) -> Callable:
    """Creates an SWA schedule function.
    
    Default configuration:
    {
        'n_epochs': 10,           # Average last 10 epochs
        'cycle_length': 1,        # Collect every epoch
        'lr': {
            'type': 'constant', 
            'value': 0.01
        }
    }
    
    Alternative cyclical configuration:
    {
        'start_epoch': 0.75,      # When to start collecting (as fraction)
        'cycle_length': 5,        # Length of each cycle
        'lr': {
            'type': 'cyclical',
            'min_lr': 0.001,      # Min learning rate in cycle
            'max_lr': 0.05        # Max learning rate in cycle
        }
    }
    """
    lr_config = protocol.get('lr', {'type': 'constant', 'value': 0.01})
    lr_type = lr_config['type']
    if lr_type not in ['constant', 'cyclical']:
        raise ValueError("lr type must be 'constant' or 'cyclical'")
    
    if lr_type == 'constant':
        lr_value = lr_config.get('value', 0.01)
        n_epochs = protocol.get('n_epochs', 10)  # Default to last 10 epochs
        cycle_length = protocol.get('cycle_length', 1)
    else:  # cyclical
        min_lr = lr_config.get('min_lr', 0.001)
        max_lr = lr_config.get('max_lr', 0.05)
        start_pct = protocol.get('start_epoch', 0.75)
        cycle_length = protocol.get('cycle_length', 5)
    
    def get_cyclical_lr(epoch: int, start_epoch: int) -> float:
        """Implements cyclical learning rate"""
        if epoch < start_epoch:
            return min_lr
            
        # Calculate position within current cycle
        cycle_position = (epoch - start_epoch) % cycle_length
        cycle_progress = cycle_position / cycle_length
        
        # Cosine annealing within cycle
        cos_out = np.cos(np.pi * cycle_progress) + 1
        return min_lr + 0.5 * (max_lr - min_lr) * cos_out
    
    def schedule(epoch: int, total_epochs: int) -> Tuple[float, bool]:
        if lr_type == 'constant':
            # Collect in last n_epochs
            start_epoch = total_epochs - n_epochs
            lr = lr_value
        else:  # cyclical
            # Convert percentage to epoch number if needed
            start_epoch = int(start_pct * total_epochs) if start_pct < 1 else int(start_pct)
            lr = get_cyclical_lr(epoch, start_epoch)
        
        # Collect weights at the end of each cycle after start_epoch
        if epoch < start_epoch:
            should_collect = False
        else:
            is_cycle_end = ((epoch - start_epoch) % cycle_length) == 0
            should_collect = is_cycle_end
        
        return lr, should_collect
    
    return schedule