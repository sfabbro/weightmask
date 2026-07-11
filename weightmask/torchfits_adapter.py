"""Optional adapter for the public :mod:`torchfits` image I/O API.

Torchfits is deliberately not a WeightMask dependency.  This adapter imports
only ``torchfits.read`` and ``torchfits.write`` at use time and never touches
private torchfits modules.
"""

from __future__ import annotations

import importlib
from typing import Any, Mapping

import numpy as np


class TorchfitsUnavailableError(ImportError):
    """Raised when the optional torchfits adapter is requested but unavailable."""


def _load_torchfits() -> Any:
    try:
        return importlib.import_module("torchfits")
    except ImportError as exc:
        raise TorchfitsUnavailableError("torchfits is not installed; install it to use TorchfitsArrayHeaderIO") from exc


def torchfits_available() -> bool:
    """Return whether the optional public torchfits module imports successfully."""
    try:
        _load_torchfits()
    except TorchfitsUnavailableError:
        return False
    return True


def _as_numpy(array: Any) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach()
    if hasattr(array, "cpu"):
        array = array.cpu()
    if hasattr(array, "numpy"):
        array = array.numpy()
    return np.asarray(array)


class TorchfitsArrayHeaderIO:
    """``ArrayHeaderIO`` adapter using only torchfits' documented root APIs."""

    def __init__(self, torchfits_module: Any | None = None) -> None:
        self._torchfits = torchfits_module or _load_torchfits()

    def read_array(self, path: str, *, hdu: int = 0) -> tuple[np.ndarray, Mapping[str, Any]]:
        result = self._torchfits.read(path, hdu=hdu, return_header=True)
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("torchfits.read(..., return_header=True) did not return (array, header)")
        array, header = result
        return _as_numpy(array), dict(header or {})

    def write_array(
        self,
        path: str,
        array: np.ndarray,
        header: Mapping[str, Any],
        *,
        overwrite: bool = False,
    ) -> None:
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise TorchfitsUnavailableError("torch is required by the torchfits adapter") from exc
        self._torchfits.write(
            path,
            torch.as_tensor(np.asarray(array)),
            header=dict(header),
            overwrite=overwrite,
        )
