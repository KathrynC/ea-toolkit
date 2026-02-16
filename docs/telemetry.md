# telemetry

Generation-by-generation logging in JSON-lines format.

---

## Overview

Provides structured logging for optimization runs. Each line in the output file is a self-contained JSON object, making it easy to stream, append, and parse incrementally. Handles numpy type serialization automatically.

---

## Class

### `Telemetry`

```python
tel = Telemetry("run_001.jsonl")
tel.start()
for gen in range(100):
    tel.log_generation(gen, best_fitness=best_f, pop_size=n,
                       extra={'sigma': current_sigma})
tel.finish()
```

| Method | Description |
|--------|-------------|
| `start()` | Open file, write header with timestamp. Creates parent directories. |
| `log_generation(gen, best_fitness, pop_size, extra=None)` | Write one generation entry with elapsed time. |
| `finish()` | Write footer with total elapsed time and generation count, close file. |

**Event types in .jsonl output:**

| Event | Fields |
|-------|--------|
| `start` | `timestamp`, `timestamp_iso` |
| `generation` | `gen`, `best_fitness`, `pop_size`, `elapsed_s`, optional `extra` |
| `finish` | `total_elapsed_s`, `total_generations`, `timestamp` |

---

## Function

### `load_telemetry(path) -> list[dict]`

Read a .jsonl file and return a list of dicts. Lines that fail to parse as JSON are silently skipped.

```python
entries = load_telemetry("run_001.jsonl")
gen_entries = [e for e in entries if e['event'] == 'generation']
```

---

## Numpy Serialization

The internal `_json_default` handler converts `np.integer` → `int`, `np.floating` → `float`, and `np.ndarray` → `list` for JSON compatibility.

---

## Source

`ea_toolkit/telemetry.py` — 147 lines.
