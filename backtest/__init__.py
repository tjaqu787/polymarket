"""
Polymarket Backtesting Engine

A comprehensive backtesting framework for testing trading strategies
on Polymarket prediction market data.
"""

from .engine import BacktestEngine
from .strategy import Strategy
from .portfolio import Portfolio
from .data_loader import DataLoader
from .performance import PerformanceMetrics

__all__ = [
    'BacktestEngine',
    'Strategy',
    'Portfolio',
    'DataLoader',
    'PerformanceMetrics',
]
