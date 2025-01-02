from typing import Tuple, Callable


def create_swa_schedule(
        schedule_type: str,
        initial_lr: float,
        swa_lr: float,
        start_epoch: int,
        total_epochs: int,
        cycle_length: int = None,
        decay_fraction: float = 0.05
    ) -> Callable[[int], Tuple[float, bool]]:
    """
    Creates a learning rate schedule function.
    
    Args:
        schedule_type: Type of schedule ('constant', 'cyclic', 'linear')
        initial_lr: Initial learning rate
        swa_lr: SWA learning rate
        start_epoch: Epoch to start SWA
        total_epochs: total number of epochs
        cycle_length: Length of cycle for cyclic schedule (required if schedule_type='cyclic')
        decay_fraction: Fraction of total epochs used for decay period (used in cyclic and linear)
    
    Returns:
        Function that takes epoch number and returns (learning_rate, should_collect)
    """
    # Validate schedule type first
    if schedule_type not in ['constant', 'cyclic', 'linear']:
        raise ValueError(f"schedule_type must be one of: constant, cyclic, linear")
        
    # Then validate cyclic schedule requirements
    if schedule_type == 'cyclic':
        if cycle_length is None:
            raise ValueError("cycle_length must be provided for cyclic schedule")
            
    def constant_schedule(epoch: int) -> Tuple[float, bool]:
        """Keep initial lr, collect after start_epoch"""
        should_collect = epoch > start_epoch
        return initial_lr, should_collect
    
    def cyclic_schedule(epoch: int) -> Tuple[float, bool]:
        """Linear decay followed by cyclic learning rate with collections"""
        if epoch < start_epoch:
            # Before SWA: constant high learning rate
            return initial_lr, False
        
        decay_epochs = int(decay_fraction * total_epochs)
        decay_end = start_epoch + decay_epochs
        
        if epoch < decay_end:
            # Linear decay period
            progress = (epoch - start_epoch) / decay_epochs
            lr = initial_lr + (swa_lr - initial_lr) * progress
            return lr, False
            
        # After decay: start cyclic pattern with collections
        cycle_pos = (epoch - decay_end) % cycle_length
        should_collect = cycle_pos == 0
        
        # Define cycle peak to be midway between swa_lr and the lr at decay end
        cycle_peak = swa_lr + (initial_lr - swa_lr) * 0.5
        
        # Upward then downward triangle wave
        if cycle_pos < cycle_length / 2:
            # First half: go from swa_lr up to cycle_peak
            progress = cycle_pos / (cycle_length / 2)
            lr = swa_lr + (cycle_peak - swa_lr) * progress
        else:
            # Second half: go from cycle_peak back to swa_lr
            progress = (cycle_pos - cycle_length / 2) / (cycle_length / 2)
            lr = cycle_peak - (cycle_peak - swa_lr) * progress
        
        return lr, should_collect
    
    def linear_schedule(epoch: int) -> Tuple[float, bool]:
        """Linear decay followed by constant lr with collection"""
        if epoch < start_epoch:
            # Before SWA: high learning rate, no collection
            return initial_lr, False
            
        decay_epochs = int(decay_fraction * total_epochs)
        decay_end = start_epoch + decay_epochs
        
        if epoch < decay_end:
            # During decay: linear decrease, no collection
            progress = (epoch - start_epoch) / decay_epochs
            lr = initial_lr + (swa_lr - initial_lr) * progress
            return lr, False
        else:
            # After decay: constant low lr, collect weights
            return swa_lr, True
    
    schedules = {
        'constant': constant_schedule,
        'cyclic': cyclic_schedule,
        'linear': linear_schedule
    }
    
    return schedules[schedule_type]