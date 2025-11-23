"""
Data generation module.

This module contains functions for generating synthetic voting instances
and loading real-world preference data.
"""

from .generators import (
    generate_instance,
    generate_rd_trap_instance,
    generate_ml_trap_instance
)

__all__ = [
    'generate_instance',
    'generate_rd_trap_instance',
    'generate_ml_trap_instance'
]

