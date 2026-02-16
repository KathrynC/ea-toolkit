"""
ea_toolkit -- Standalone Evolutionary Algorithms Toolkit.

A generalized toolkit extracted from the Evolutionary-Robotics and
how-to-live-much-longer projects. Provides abstract base classes,
mutation operators, crossover operators, selection strategies, population
management, multiple optimization algorithms, telemetry logging,
landscape analysis tools, benchmark functions, and a callback system.

All numerical operations use numpy only (no scipy, no sklearn).
"""

# Base classes
from ea_toolkit.base import (
    FitnessFunction,
    MutationOperator,
    CrossoverOperator,
    SelectionStrategy,
    Algorithm,
    Callback,
)

# Mutation operators
from ea_toolkit.mutation import (
    GaussianMutation,
    CauchyMutation,
    AdaptiveMutation,
)

# Crossover operators
from ea_toolkit.crossover import (
    SBXCrossover,
    UniformCrossover,
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
    DifferentialEvolution,
    CMAES,
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

# Benchmarks
from ea_toolkit.benchmarks import (
    SphereFitness,
    RosenbrockFitness,
    RastriginFitness,
    AckleyFitness,
    ZDT1Fitness,
)

# Callbacks
from ea_toolkit.callbacks import (
    ConvergenceChecker,
    ProgressPrinter,
    TelemetryCallback,
    HistoryRecorder,
)

# Zimmerman bridge (adapters always available; convenience fns need zimmerman)
from ea_toolkit.zimmerman_bridge import (
    FitnessAsSimulator,
    SimulatorAsFitness,
)

__all__ = [
    # Base
    'FitnessFunction', 'MutationOperator', 'CrossoverOperator',
    'SelectionStrategy', 'Algorithm', 'Callback',
    # Mutation
    'GaussianMutation', 'CauchyMutation', 'AdaptiveMutation',
    # Crossover
    'SBXCrossover', 'UniformCrossover',
    # Selection
    'TournamentSelection', 'TruncationSelection', 'EpsilonGreedy',
    # Population
    'PopulationManager', 'random_population', 'elitism',
    'diversity_metric', 'behavioral_diversity',
    # Algorithms
    'HillClimber', 'OnePlusLambdaES', 'RidgeWalker',
    'CliffMapper', 'NoveltySeeker', 'EnsembleExplorer',
    'DifferentialEvolution', 'CMAES',
    # Telemetry
    'Telemetry', 'load_telemetry',
    # Landscape
    'probe_cliffiness', 'roughness_ratio', 'sign_flip_rate',
    'gradient_estimate', 'LandscapeAnalyzer',
    # Benchmarks
    'SphereFitness', 'RosenbrockFitness', 'RastriginFitness',
    'AckleyFitness', 'ZDT1Fitness',
    # Callbacks
    'ConvergenceChecker', 'ProgressPrinter', 'TelemetryCallback',
    'HistoryRecorder',
    # Zimmerman bridge
    'FitnessAsSimulator', 'SimulatorAsFitness',
]
