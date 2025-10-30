"""
Trucks and Drones Multi-Agent RL Project
主源代码模块
"""

__version__ = '2.0.0'

# 导出主要组件
from . import algorithms
from . import environment
from . import utils

__all__ = [
    'algorithms',
    'environment',
    'utils',
]

