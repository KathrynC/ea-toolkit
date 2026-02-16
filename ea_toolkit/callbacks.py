"""
ea_toolkit.callbacks -- Common callback implementations.

Provides ready-to-use callbacks for convergence checking, progress
printing, and telemetry integration. All callbacks extend the base
Callback class from ea_toolkit.base.
"""

from ea_toolkit.base import Callback


class ConvergenceChecker(Callback):
    """Stop early if fitness hasn't improved for `patience` generations.

    Monitors best fitness each generation. If no improvement exceeding
    min_delta is observed for `patience` consecutive generations,
    returns False from on_generation() to request early stopping.

    Args:
        patience: number of generations without improvement before stopping.
            Default 20.
        min_delta: minimum improvement to count as progress. Default 1e-8.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-8):
        self.patience = patience
        self.min_delta = min_delta
        self._best = float('-inf')
        self._stale = 0

    def on_generation(self, algorithm, generation, best_fitness):
        if best_fitness > self._best + self.min_delta:
            self._best = best_fitness
            self._stale = 0
        else:
            self._stale += 1

        if self._stale >= self.patience:
            return False  # Request stop

    def on_start(self, algorithm):
        self._best = float('-inf')
        self._stale = 0


class ProgressPrinter(Callback):
    """Print progress every N generations.

    Args:
        every: print interval in generations. Default 10.
        prefix: optional prefix string for output lines. Default "".
    """

    def __init__(self, every: int = 10, prefix: str = ""):
        self.every = every
        self.prefix = prefix

    def on_generation(self, algorithm, generation, best_fitness):
        if generation % self.every == 0:
            evals = len(algorithm.history)
            print(f"{self.prefix}Gen {generation}: "
                  f"best={best_fitness:.6f}  evals={evals}")

    def on_finish(self, algorithm):
        best = algorithm.best()
        if best:
            print(f"{self.prefix}Finished: best={best['fitness']:.6f}  "
                  f"total_evals={len(algorithm.history)}")


class TelemetryCallback(Callback):
    """Bridge between the callback system and the Telemetry logger.

    Automatically calls Telemetry.start(), log_generation(), and finish()
    at the appropriate callback events.

    Args:
        telemetry: a Telemetry instance from ea_toolkit.telemetry.
    """

    def __init__(self, telemetry):
        self.telemetry = telemetry

    def on_start(self, algorithm):
        self.telemetry.start()

    def on_generation(self, algorithm, generation, best_fitness):
        self.telemetry.log_generation(
            gen=generation,
            best_fitness=best_fitness,
            pop_size=len(algorithm.history),
        )

    def on_finish(self, algorithm):
        self.telemetry.finish()


class HistoryRecorder(Callback):
    """Record per-generation statistics for later analysis.

    After the run, access self.generations for a list of dicts with:
    - generation: int
    - best_fitness: float
    - n_evals: int
    - improved: bool (whether this generation found a new best)
    """

    def __init__(self):
        self.generations: list[dict] = []
        self._prev_best = float('-inf')

    def on_start(self, algorithm):
        self.generations = []
        self._prev_best = float('-inf')

    def on_generation(self, algorithm, generation, best_fitness):
        improved = best_fitness > self._prev_best
        n_evals = len(algorithm.history) if algorithm is not None else 0
        self.generations.append({
            'generation': generation,
            'best_fitness': best_fitness,
            'n_evals': n_evals,
            'improved': improved,
        })
        self._prev_best = best_fitness
