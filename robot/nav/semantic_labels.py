"""
Semantic label store for SLAM maps.

Labels are world-space points with a name, persisted as a JSON sidecar
alongside the .npz voxel map file.  Thread-safe.
"""

from __future__ import annotations

import json
import os
import threading
from typing import List, Optional


class SemanticLabelStore:
    """
    Stores named world-space labels (e.g. "kitchen", "white table").

    Each label is a dict: {"name": str, "x": float, "y": float, "z": float}
    where x/z are the horizontal plane and y is the vertical (up) coordinate.

    Saved as a JSON sidecar: for map path ``my_map.npz`` the labels file is
    ``my_map_labels.json``.  Use :meth:`sidecar_path` to resolve the path.
    """

    # Palette used to colour labels in Viser (cycles for >12 labels).
    _PALETTE = [
        (0.96, 0.26, 0.21),   # red
        (0.13, 0.59, 0.95),   # blue
        (0.30, 0.69, 0.31),   # green
        (1.00, 0.76, 0.03),   # yellow
        (0.61, 0.15, 0.69),   # purple
        (1.00, 0.60, 0.00),   # orange
        (0.00, 0.74, 0.83),   # cyan
        (0.91, 0.12, 0.39),   # pink
        (0.40, 0.23, 0.72),   # indigo
        (0.00, 0.59, 0.53),   # teal
        (0.55, 0.76, 0.29),   # lime
        (0.47, 0.33, 0.28),   # brown
    ]

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._labels: List[dict] = []   # list of {"name", "x", "y", "z"}
        self._dirty: bool = False       # True when labels changed since last Viser render

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_label(self, name: str, x: float, z: float, y: float = 0.0) -> None:
        """Add or overwrite a label by name."""
        name = name.strip()
        if not name:
            return
        entry = {"name": name, "x": float(x), "y": float(y), "z": float(z)}
        with self._lock:
            for i, lbl in enumerate(self._labels):
                if lbl["name"] == name:
                    self._labels[i] = entry
                    self._dirty = True
                    return
            self._labels.append(entry)
            self._dirty = True
        print(f"[SemanticLabels] Added label '{name}' at ({x:.2f}, {y:.2f}, {z:.2f})")

    def remove_label(self, name: str) -> bool:
        """Remove a label by name. Returns True if it existed."""
        with self._lock:
            before = len(self._labels)
            self._labels = [l for l in self._labels if l["name"] != name]
            changed = len(self._labels) < before
            if changed:
                self._dirty = True
        if changed:
            print(f"[SemanticLabels] Removed label '{name}'")
        return changed

    def get_labels(self) -> List[dict]:
        """Return a snapshot copy of all labels."""
        with self._lock:
            return list(self._labels)

    def get_names(self) -> List[str]:
        with self._lock:
            return [l["name"] for l in self._labels]

    def label_color(self, index: int) -> tuple:
        """Return an (r, g, b) float tuple for label at the given index."""
        return self._PALETTE[index % len(self._PALETTE)]

    # ------------------------------------------------------------------
    # Dirty flag (for Viser re-render throttling)
    # ------------------------------------------------------------------

    def consume_dirty(self) -> bool:
        """Return True and reset the dirty flag if labels changed since last call."""
        with self._lock:
            if self._dirty:
                self._dirty = False
                return True
            return False

    def mark_dirty(self) -> None:
        with self._lock:
            self._dirty = True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def sidecar_path(map_path: str) -> str:
        """Return the JSON sidecar path for a given .npz map path."""
        base, _ = os.path.splitext(map_path)
        return base + "_labels.json"

    def save(self, path: str) -> None:
        """Save labels to JSON. Creates parent directories as needed."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with self._lock:
            data = {"version": 1, "labels": list(self._labels)}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[SemanticLabels] Saved {len(data['labels'])} label(s) to {path}")

    def load(self, path: str) -> None:
        """Load labels from JSON. No-op if file does not exist."""
        if not os.path.isfile(path):
            return
        with open(path) as f:
            data = json.load(f)
        labels = data.get("labels", [])
        with self._lock:
            self._labels = [
                {"name": str(l["name"]),
                 "x": float(l["x"]),
                 "y": float(l.get("y", 0.0)),
                 "z": float(l["z"])}
                for l in labels
            ]
            self._dirty = True
        print(f"[SemanticLabels] Loaded {len(self._labels)} label(s) from {path}")

    def __len__(self) -> int:
        with self._lock:
            return len(self._labels)

    def __repr__(self) -> str:
        with self._lock:
            return f"SemanticLabelStore({len(self._labels)} labels)"
