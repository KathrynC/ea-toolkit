"""
ea_toolkit.algorithms -- All evolutionary algorithm implementations.

Algorithms:
- HillClimber: parallel hill climber with restarts
- OnePlusLambdaES: (1+lambda) evolution strategy
- RidgeWalker: multi-objective Pareto search
- CliffMapper: high-sensitivity region search
- NoveltySeeker: novelty-driven search
- EnsembleExplorer: multi-walker ensemble with teleportation
- DifferentialEvolution: DE/rand/1/bin population-based optimizer
- CMAES: Covariance Matrix Adaptation Evolution Strategy
"""

from ea_toolkit.algorithms.hill_climber import HillClimber
from ea_toolkit.algorithms.es import OnePlusLambdaES
from ea_toolkit.algorithms.ridge_walker import RidgeWalker
from ea_toolkit.algorithms.cliff_mapper import CliffMapper
from ea_toolkit.algorithms.novelty_seeker import NoveltySeeker
from ea_toolkit.algorithms.ensemble import EnsembleExplorer
from ea_toolkit.algorithms.de import DifferentialEvolution
from ea_toolkit.algorithms.cma_es import CMAES

__all__ = [
    'HillClimber',
    'OnePlusLambdaES',
    'RidgeWalker',
    'CliffMapper',
    'NoveltySeeker',
    'EnsembleExplorer',
    'DifferentialEvolution',
    'CMAES',
]
