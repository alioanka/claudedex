"""Consecutive Losses Auto-Reset Configuration"""
from datetime import timedelta

CONSECUTIVE_LOSSES = {
    'max_consecutive_losses': 6,
    'enabled': True,
    'auto_reset_enabled': True,
    'block_hours': 2,
    'progressive_block_enabled': True,
    'progressive_multiplier': 2.0,
    'max_block_hours': 6,
    'use_size_reduction': False,
    'size_reduction_schedule': {0: 1.00, 3: 0.75, 4: 0.50, 5: 0.25, 6: 0.10},
    'alert_on_block': True,
    'alert_on_reset': True,
    'alert_on_approach': True,
    'approach_threshold': 4,
}

def get_block_duration(block_count: int = 0) -> timedelta:
    base_hours = CONSECUTIVE_LOSSES['block_hours']
    if not CONSECUTIVE_LOSSES['progressive_block_enabled'] or block_count == 0:
        return timedelta(hours=base_hours)
    multiplier = CONSECUTIVE_LOSSES['progressive_multiplier']
    hours = base_hours * (multiplier ** (block_count - 1))
    return timedelta(hours=min(hours, CONSECUTIVE_LOSSES['max_block_hours']))

def get_position_size_multiplier(consecutive_losses: int) -> float:
    if not CONSECUTIVE_LOSSES['use_size_reduction']:
        return 1.0
    schedule = CONSECUTIVE_LOSSES['size_reduction_schedule']
    multiplier = 1.0
    for threshold, mult in sorted(schedule.items()):
        if consecutive_losses >= threshold:
            multiplier = mult
    return multiplier