from __future__ import annotations

"""Loader for the Natural Plan benchmark datasets.

"""

from dataclasses import dataclass
from typing import Dict, List, Any
import json
import pathlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class NPExample:
    """Container for a single Natural-Plan example."""

    task: str  # "trip" | "meeting" | "calendar"
    example_id: str  
    prompt: str  
    golden: Any  
    meta: Dict[str, Any]  

    # For Natural-Plan we treat the entire prompt as the "question".
    @property
    def question(self) -> str:  # type: ignore
        return self.prompt

    @property
    def choices(self) -> list:  # type: ignore
        """Return an empty list so that generic CSV writer does not break."""
        return []


class NaturalPlanLoader:
    """Load Natural-Plan data from the original json files located in ``eval/data``.

    Parameters
    ----------
    data_root: str | pathlib.Path
        Directory that contains the three json files. Defaults to
        ``../eval/data`` relative to the *bench* package root.
    """

    FILES = {
        "trip": "trip_planning.json",
        "meeting": "meeting_planning.json",
        "calendar": "calendar_scheduling.json",
    }

    def __init__(self, data_root: str | pathlib.Path | None = None):
        if data_root is None:
            self.data_root = (pathlib.Path(__file__).resolve().parents[3] / "eval/data").resolve()
        else:
            self.data_root = pathlib.Path(data_root).expanduser().resolve()

        if not self.data_root.exists():
            raise FileNotFoundError(f"Natural-Plan data dir not found: {self.data_root}")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def load(self, task: str, start_index: int | None = None, end_index: int | None = None, max_count: int | None = None) -> List[NPExample]:
        """Load examples for the given task.

        Parameters
        ----------
        task: str
            One of ``trip``, ``meeting``, or ``calendar``.
        start_index: int | None
            Start loading from this example index (0-based, inclusive).
        end_index: int | None  
            Stop loading at this example index (exclusive).
        max_count: int | None
            Maximum number of examples to load (applied after start_index).
        """
        if task not in self.FILES:
            raise ValueError(f"Unknown Natural-Plan task: {task}. Expected one of {list(self.FILES)}")

        path = self.data_root / self.FILES[task]
        logger.info("Loading Natural-Plan %s from %s", task, path)

        with path.open() as f:
            raw: Dict[str, Dict[str, Any]] = json.load(f)

        # Use keys as-is - they should already be in correct order in the JSON
        all_keys = list(raw.keys())
        
        # Apply start_index filter
        if start_index is not None:
            if start_index < 0 or start_index >= len(all_keys):
                raise ValueError(f"start_index ({start_index}) out of range [0, {len(all_keys)})")
            all_keys = all_keys[start_index:]
        
        # Apply end_index filter
        if end_index is not None:
            if end_index <= 0:
                raise ValueError(f"end_index ({end_index}) must be positive")
            end_idx = end_index - (start_index or 0) 
            if end_idx > 0 and end_idx < len(all_keys):
                all_keys = all_keys[:end_idx]
        
        # Apply max_count filter
        if max_count is not None and max_count < len(all_keys):
            all_keys = all_keys[:max_count]

        examples: List[NPExample] = []
        for example_id in all_keys:
            record = raw[example_id]
            examples.append(
                NPExample(
                    task=task,
                    example_id=example_id,
                    prompt=record["prompt_5shot"],
                    golden=record["golden_plan"],
                    meta=record,
                )
            )

        total_available = len(raw)
        logger.info("Loaded %d of %d examples for Natural-Plan %s", len(examples), total_available, task)
        return examples 