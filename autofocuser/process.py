from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from typing import Sequence

import numpy as np
from astropy.io import fits

from autofocuser.measure import (
    FocusRecommendation,
    MeasureError,
    StarMeasurement,
    measure_star,
    recommend_focus,
)


@dataclass(frozen=True)
class FrameResult:
    path: Path
    focus: int
    measurement: StarMeasurement


@dataclass(frozen=True)
class SweepResult:
    frames: list[FrameResult]
    recommendation: FocusRecommendation | None


def collect_fits(arguments: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for argument in arguments:
        path = Path(argument)
        if path.is_dir():
            found = sorted(
                child
                for child in path.iterdir()
                if child.is_file() and child.suffix.lower() in {".fits", ".fit"}
            )
            if not found:
                raise MeasureError(f"No FITS files in {path}.")
            paths.extend(found)
        elif path.is_file():
            paths.append(path)
        else:
            raise MeasureError(f"Not a FITS file or folder: {path}")
    if not paths:
        raise MeasureError("No FITS files to measure.")
    return paths


def run_fits(
    paths: Sequence[Path],
    box_size: int | tuple[int, int] | None = None,
) -> SweepResult:
    frames: list[FrameResult] = []
    previous: tuple[float, float] | None = None
    for path in paths:
        focus, image = _read_fits(path)
        try:
            measured = measure_star(image, previous, box_size)
        except MeasureError as err:
            raise MeasureError(f"{path.name}: {err}") from err
        previous = (measured.x, measured.y)
        frames.append(FrameResult(path=path, focus=focus, measurement=measured))

    usable = [frame for frame in frames if frame.measurement.usable]
    recommendation = None
    if usable:
        recommendation = recommend_focus(
            [frame.focus for frame in usable],
            [frame.measurement.sigma for frame in usable],
        )
    return SweepResult(frames=frames, recommendation=recommendation)


def _read_fits(path: Path) -> tuple[int, np.ndarray]:
    try:
        with fits.open(path) as hdul:
            header = hdul[0].header
            data = hdul[0].data
    except Exception as err:
        raise MeasureError(f"{path.name}: could not read the FITS file.") from err
    if data is None:
        raise MeasureError(f"{path.name}: the FITS file has no image.")
    image = np.asarray(data)
    if image.ndim != 2:
        raise MeasureError(f"{path.name}: the FITS file has no image.")
    if "FOCUS" not in header:
        raise MeasureError(f"{path.name}: the header has no FOCUS value.")
    try:
        focus = float(header["FOCUS"])
    except (TypeError, ValueError) as err:
        raise MeasureError(f"{path.name}: the header has no FOCUS value.") from err
    if not np.isfinite(focus):
        raise MeasureError(f"{path.name}: the header has no FOCUS value.")
    return int(round(focus)), image
