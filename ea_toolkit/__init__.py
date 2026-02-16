"""
ea_toolkit -- Standalone Evolutionary Algorithms Toolkit.

A generalized toolkit extracted from the Evolutionary-Robotics and
how-to-live-much-longer projects. Provides abstract base classes,
mutation operators, selection strategies, population management,
multiple optimization algorithms, telemetry logging, and landscape
analysis tools.

All numerical operations use numpy only (no scipy, no sklearn).
"""

# Base classes
from ea_toolkit.base import (
    FitnessFunction,
    MutationOperator,
    SelectionStrategy,
    Algorithm,
)

# Mutation operators
from ea_toolkit.mutation import (
    GaussianMutation,
    CauchyMutation,
    AdaptiveMutation,
)

# Selection strategies
from ea_toolkit.selection import (
    TournamentSelection,
    TruncationSelection,
    EpsilonGreedy,
)

# Population management
from ea_toolkit.population import (
    PopulationManager,
    random_population,
    elitism,
    diversity_metric,
    behavioral_diversity,
)

# Algorithms
from ea_toolkit.algorithms import (
    HillClimber,
    OnePlusLambdaES,
    RidgeWalker,
    CliffMapper,
    NoveltySeeker,
    EnsembleExplorer,
)

# Telemetry
from ea_toolkit.telemetry import (
    Telemetry,
    load_telemetry,
)

# Landscape analysis
from ea_toolkit.landscape import (
    probe_cliffiness,
    roughness_ratio,
    sign_flip_rate,
    gradient_estimate,
    LandscapeAnalyzer,
)

__all__ = [
    # Base
    'FitnessFunction', 'MutationOperator', 'SelectionStrategy', 'Algorithm',
    # Mutation
    'GaussianMutation', 'CauchyMutation', 'AdaptiveMutation',
    # Selection
    'TournamentSelection', 'TruncationSelection', 'EpsilonGreedy',
    # Population
    'PopulationManager', 'random_population', 'elitism',
    'diversity_metric', 'behavioral_diversity',
    # Algorithms
    'HillClimber', 'OnePlusLambdaES', 'RidgeWalker',
    'CliffMapper', 'NoveltySeeker', 'EnsembleExplorer',
    # Telemetry
    'Telemetry', 'load_telemetry',
    # Landscape
    'probe_cliffiness', 'roughness_ratio', 'sign_flip_rate',
    'gradient_estimate', 'LandscapeAnalyzer',
]
