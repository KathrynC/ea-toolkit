"""
ea_toolkit.telemetry -- Generation-by-generation logging in JSON-lines format.

Provides:
- Telemetry: class for logging optimization progress to .jsonl files.
- load_telemetry(): read back a .jsonl file and return a list of dicts.
"""

import json
import time
from pathlib import Path


class Telemetry:
    """Generation-by-generation logger that writes JSON-lines (.jsonl) files.

    Each line in the output file is a self-contained JSON object, making
    it easy to stream, append, and parse incrementally.

    Usage:
        tel = Telemetry("run_001.jsonl")
        tel.start()
        for gen in range(100):
            tel.log_generation(gen, best_fitness=f, pop_size=n)
        tel.finish()
    """

    def __init__(self, path: str | Path):
        """Initialize the telemetry logger.

        Args:
            path: file path for the .jsonl output file.
        """
        self.path = Path(path)
        self._file = None
        self._start_time: float | None = None
        self._gen_count = 0

    def start(self) -> None:
        """Start the telemetry session.

        Opens the output file and writes a header entry with the start
        timestamp. Creates parent directories if they do not exist.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, 'w')
        self._start_time = time.time()
        self._gen_count = 0

        header = {
            'event': 'start',
            'timestamp': self._start_time,
            'timestamp_iso': time.strftime('%Y-%m-%dT%H:%M:%S',
                                           time.localtime(self._start_time)),
        }
        self._write_entry(header)

    def log_generation(self, gen: int, best_fitness: float,
                       pop_size: int, extra: dict | None = None) -> None:
        """Log one generation's summary.

        Args:
            gen: generation number (0-indexed).
            best_fitness: best fitness value in this generation.
            pop_size: number of individuals in the population.
            extra: optional dict of additional data to include.
        """
        elapsed = time.time() - self._start_time if self._start_time else 0.0

        entry = {
            'event': 'generation',
            'gen': gen,
            'best_fitness': best_fitness,
            'pop_size': pop_size,
            'elapsed_s': round(elapsed, 3),
        }
        if extra:
            entry['extra'] = extra

        self._write_entry(entry)
        self._gen_count += 1

    def finish(self) -> None:
        """Finish the telemetry session.

        Writes a footer entry with the total elapsed time and generation
        count, then closes the output file.
        """
        elapsed = time.time() - self._start_time if self._start_time else 0.0

        footer = {
            'event': 'finish',
            'total_elapsed_s': round(elapsed, 3),
            'total_generations': self._gen_count,
            'timestamp': time.time(),
        }
        self._write_entry(footer)

        if self._file:
            self._file.close()
            self._file = None

    def _write_entry(self, entry: dict) -> None:
        """Write a single JSON entry as one line in the .jsonl file.

        Args:
            entry: dict to serialize as JSON.
        """
        if self._file:
            line = json.dumps(entry, default=_json_default)
            self._file.write(line + '\n')
            self._file.flush()


def _json_default(obj):
    """JSON serializer for numpy types and other non-standard objects."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def load_telemetry(path: str | Path) -> list[dict]:
    """Load a .jsonl telemetry file and return a list of dicts.

    Args:
        path: file path to the .jsonl file.

    Returns:
        list of dicts, one per line in the file. Lines that fail to
        parse as JSON are silently skipped.
    """
    path = Path(path)
    entries = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries
