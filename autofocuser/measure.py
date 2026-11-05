from __future__ import annotations

from dataclasses import dataclass

from typing import Sequence

import numpy as np
import sep
from astropy.modeling import fitting, models
from scipy.ndimage import map_coordinates


MAX_STAR_SHIFT_PX = 32

_SEP_THRESHOLD = 3.0
_SEP_MIN_AREA = 9

# Kron aperture: first moment of the light inside 6 detection radii,
# times 2.5, and at least 3.5 radii. For a Gaussian, r_1 = sqrt(pi/2),
# so 2.5 r_1 is about 3.1 sigma and the floor of 3.5 sigma holds >99% of the light.
_KRON_WINDOW = 6.0
_KRON_FACTOR = 2.5
_KRON_MIN_SCALE = 3.5

_BOX_PADDING_PX = 2
_MIN_DYNAMIC_BOX = 9

# Frames whose Gaussian fit explains less than this share of the box variance
# are left out of the focus parabola.
MIN_R_SQUARED = 0.7

# The donut check fits a quartic between the outermost pixels of a cross-section
# that reach this fraction of its peak.
_DONUT_WINDOW_FRACTION = 0.2
# Cuts at 0, 45, 90 and 135 degrees through the centre of the star's outline.
_DONUT_CUT_ANGLES = tuple(np.deg2rad(a) for a in (0.0, 45.0, 90.0, 135.0))


class MeasureError(Exception):
    pass


@dataclass(frozen=True)
class StarMeasurement:
    x: float
    y: float
    sigma: float
    sigma_x: float
    sigma_y: float
    amplitude: float
    gauss_x: float
    gauss_y: float
    box_origin_x: int
    box_origin_y: int
    box_size: int
    donut: bool = False
    r_squared: float = float("nan")

    @property
    def usable(self) -> bool:
        """True when the frame can go into the focus parabola."""
        return (
            not self.donut
            and np.isfinite(self.r_squared)
            and self.r_squared >= MIN_R_SQUARED
        )


@dataclass(frozen=True)
class FocusRecommendation:
    position: int
    sigma: float
    curve_ok: bool
    message: str
    curve: list[tuple[float, float]]


def measure_star(
    image: np.ndarray,
    previous_center: tuple[float, float] | None = None,
    box_size: int | tuple[int, int] | None = None,
) -> StarMeasurement:
    side = None if box_size is None else _square_side(box_size)
    signal, sources, segmap = _extract(image)
    if len(sources) == 0:
        raise MeasureError("No star in the image.")

    index = int(np.argmax(sources["flux"]))
    star = sources[index]
    x = float(star["x"])
    y = float(star["y"])
    _check_shift(x, y, previous_center)
    if side is None:
        side = _dynamic_box_side(signal, star, segmap)

    origin_x, origin_y, cut = _cut_box(signal, x, y, side)
    if _has_donut(cut):
        nan = float("nan")
        return StarMeasurement(
            x=x,
            y=y,
            sigma=nan,
            sigma_x=nan,
            sigma_y=nan,
            amplitude=nan,
            gauss_x=nan,
            gauss_y=nan,
            box_origin_x=origin_x,
            box_origin_y=origin_y,
            box_size=side,
            donut=True,
        )

    sigma_x, sigma_y, amplitude, gauss_x, gauss_y = _fit_gaussian(cut, side)
    r_squared = _gaussian_r_squared(
        cut, amplitude, gauss_x, gauss_y, sigma_x, sigma_y
    )
    return StarMeasurement(
        x=x,
        y=y,
        sigma=(sigma_x + sigma_y) / 2.0,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        amplitude=amplitude,
        gauss_x=gauss_x,
        gauss_y=gauss_y,
        box_origin_x=origin_x,
        box_origin_y=origin_y,
        box_size=side,
        r_squared=r_squared,
    )


def recommend_focus(
    positions: Sequence[float],
    sigmas: Sequence[float],
) -> FocusRecommendation:
    x = np.asarray(positions, dtype=float)
    y = np.asarray(sigmas, dtype=float)
    if x.size == 0:
        raise MeasureError("No focus measurements to fit.")
    if x.size != y.size:
        raise MeasureError("Focus positions and sigmas do not match.")

    best = int(np.argmin(y))
    lowest_position = int(round(float(x[best])))
    lowest_sigma = float(y[best])

    if x.size < 3:
        return FocusRecommendation(
            position=lowest_position,
            sigma=lowest_sigma,
            curve_ok=False,
            message=(
                "Fewer than 3 points. Lowest measured sigma is "
                f"{lowest_sigma:.3f} at step {lowest_position}."
            ),
            curve=[],
        )

    center = float(np.mean(x))
    # sigma = a u^2 + b u + c, u = focus - mean(focus). Vertex at u = -b / (2a).
    quadratic = np.polyfit(x - center, y, 2)
    slope_coeff, linear_coeff, constant = (float(v) for v in quadratic)
    samples = _curve_samples(
        x,
        center,
        slope_coeff,
        linear_coeff,
        constant,
    )

    if slope_coeff <= 0:
        return FocusRecommendation(
            position=lowest_position,
            sigma=lowest_sigma,
            curve_ok=False,
            message=(
                "The fitted curve does not open upward. Lowest measured sigma is "
                f"{lowest_sigma:.3f} at step {lowest_position}."
            ),
            curve=samples,
        )

    vertex = center - linear_coeff / (2.0 * slope_coeff)
    if vertex < float(np.min(x)) or vertex > float(np.max(x)):
        return FocusRecommendation(
            position=lowest_position,
            sigma=lowest_sigma,
            curve_ok=False,
            message=(
                "The parabola minimum is outside the sweep. Lowest measured sigma is "
                f"{lowest_sigma:.3f} at step {lowest_position}."
            ),
            curve=samples,
        )

    position = int(round(vertex))
    sigma = slope_coeff * (position - center) ** 2
    sigma += linear_coeff * (position - center)
    sigma += constant
    return FocusRecommendation(
        position=position,
        sigma=float(sigma),
        curve_ok=True,
        message=f"Parabola minimum is at step {position}.",
        curve=samples,
    )


def _extract(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.ascontiguousarray(image, dtype=np.float32)
    if data.ndim != 2:
        raise MeasureError("No star in the image.")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        background = sep.Background(data)
        signal = np.ascontiguousarray(data - background, dtype=np.float32)
        sources, segmap = sep.extract(
            signal,
            _SEP_THRESHOLD,
            err=background.globalrms,
            minarea=_SEP_MIN_AREA,
            segmentation_map=True,
        )
    except Exception as err:
        raise MeasureError("No star in the image.") from err
    return signal, sources, segmap


def _dynamic_box_side(
    signal: np.ndarray,
    star: np.ndarray,
    segmap: np.ndarray,
) -> int:
    a = float(star["a"])
    b = float(star["b"])
    theta = float(star["theta"])
    if min(a, b) <= 0 or not np.isfinite(a + b + theta):
        raise MeasureError("No star in the image.")

    scale = _light_scale(signal, star, segmap, a, b, theta)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    # Axis-aligned extents of the rotated ellipse. The square uses the larger one.
    half_x = scale * float(np.hypot(a * cos_t, b * sin_t))
    half_y = scale * float(np.hypot(a * sin_t, b * cos_t))
    half = int(np.ceil(max(half_x, half_y) + _BOX_PADDING_PX))
    return max(2 * half + 1, _MIN_DYNAMIC_BOX)


def _light_scale(
    signal: np.ndarray,
    star: np.ndarray,
    segmap: np.ndarray,
    a: float,
    b: float,
    theta: float,
) -> float:
    peak_y = int(star["ypeak"])
    peak_x = int(star["xpeak"])
    seg_id = int(segmap[peak_y, peak_x])
    mask_args: dict[str, np.ndarray | int] = {}
    if seg_id > 0:
        mask_args["segmap"] = segmap
        mask_args["seg_id"] = seg_id
    try:
        kronrad, _flag = sep.kron_radius(
            signal,
            float(star["x"]),
            float(star["y"]),
            a,
            b,
            theta,
            _KRON_WINDOW,
            **mask_args,
        )
    except Exception as err:
        raise MeasureError("No star in the image.") from err

    kron = float(np.asarray(kronrad).reshape(-1)[0])
    if not np.isfinite(kron) or kron <= 0:
        return _KRON_MIN_SCALE
    return max(_KRON_FACTOR * kron, _KRON_MIN_SCALE)


def _check_shift(
    x: float,
    y: float,
    previous_center: tuple[float, float] | None,
) -> None:
    if previous_center is None:
        return
    shift = float(np.hypot(x - previous_center[0], y - previous_center[1]))
    if shift > MAX_STAR_SHIFT_PX:
        raise MeasureError(
            f"Star moved {shift:.0f} pixels since the previous frame."
        )


def _square_side(box_size: int | tuple[int, int]) -> int:
    if isinstance(box_size, tuple):
        if len(box_size) != 2 or box_size[0] != box_size[1]:
            raise MeasureError("The measurement box must be square.")
        side = box_size[0]
    else:
        side = box_size
    if isinstance(side, bool) or not isinstance(side, (int, np.integer)):
        raise MeasureError(
            "The measurement box side must be a whole number of pixels."
        )
    if int(side) < 8:
        raise MeasureError("The measurement box must be at least 8 pixels on a side.")
    return int(side)


def _cut_box(
    signal: np.ndarray,
    x: float,
    y: float,
    box_size: int,
) -> tuple[int, int, np.ndarray]:
    half = box_size // 2
    origin_x = int(round(x)) - half
    origin_y = int(round(y)) - half
    height, width = signal.shape
    if (
        origin_x < 0
        or origin_y < 0
        or origin_x + box_size > width
        or origin_y + box_size > height
    ):
        raise MeasureError(
            "No star in the measurement box. "
            "The brightest star is too close to the edge of the image."
        )
    cut = np.ascontiguousarray(
        signal[origin_y : origin_y + box_size, origin_x : origin_x + box_size]
    )
    return origin_x, origin_y, cut


def _fit_gaussian(
    cut: np.ndarray,
    box_size: int,
) -> tuple[float, float, float, float, float]:
    try:
        background = sep.Background(cut)
        sources = sep.extract(
            cut,
            _SEP_THRESHOLD,
            err=background.globalrms,
            minarea=_SEP_MIN_AREA,
        )
    except Exception as err:
        raise MeasureError("No star in the measurement box.") from err
    if len(sources) == 0:
        raise MeasureError("No star in the measurement box.")

    star = sources[int(np.argmax(sources["flux"]))]
    model = models.Gaussian2D(
        amplitude=float(max(np.max(cut), 1.0)),
        x_mean=float(star["x"]),
        y_mean=float(star["y"]),
        x_stddev=float(max(star["a"], 1.0)),
        y_stddev=float(max(star["b"], 1.0)),
        theta=0.0,
    )
    model.theta.fixed = True
    model.amplitude.bounds = (0.0, None)
    model.x_mean.bounds = (0.0, box_size - 1.0)
    model.y_mean.bounds = (0.0, box_size - 1.0)
    model.x_stddev.bounds = (0.2, box_size / 2.0)
    model.y_stddev.bounds = (0.2, box_size / 2.0)

    yy, xx = np.mgrid[0:box_size, 0:box_size]
    try:
        fitted = fitting.TRFLSQFitter()(model, xx, yy, cut)
        sigma_x = abs(float(fitted.x_stddev.value))
        sigma_y = abs(float(fitted.y_stddev.value))
        amplitude = float(fitted.amplitude.value)
        gauss_x = float(fitted.x_mean.value)
        gauss_y = float(fitted.y_mean.value)
    except ImportError:
        raise
    except Exception as err:
        raise MeasureError("No star in the measurement box.") from err

    if (
        not np.isfinite(sigma_x)
        or not np.isfinite(sigma_y)
        or not np.isfinite(amplitude)
        or sigma_x <= 0
        or sigma_y <= 0
        or amplitude <= 0
    ):
        raise MeasureError("No star in the measurement box.")
    return sigma_x, sigma_y, amplitude, gauss_x, gauss_y


def _has_donut(cut: np.ndarray) -> bool:
    """Fit a 4th degree polynomial to cuts through the star at several angles.

    A focused star gives an inverted bowl, which a quartic draws as a W: one
    maximum in the middle and a minimum on each flank. Those flank minima are
    normal. A donut has a dip in the middle, which the quartic draws as an M:
    a local minimum with a local maximum on each side. Only that counts, and
    one cut showing it is enough.

    The cuts go through the centre of the star's outline, not its
    brightness-weighted centre. A lopsided donut with a bright knot on one side
    pulls the weighted centre onto the knot, and cuts through the knot miss the
    dark middle.
    """
    center_x, center_y = _outline_center(cut)
    return any(
        _profile_has_donut(_line_profile(cut, center_x, center_y, angle))
        for angle in _DONUT_CUT_ANGLES
    )


def _outline_center(cut: np.ndarray) -> tuple[float, float]:
    """Unweighted mean position of the pixels that reach the window fraction."""
    peak = float(np.max(cut))
    ys, xs = np.nonzero(cut >= _DONUT_WINDOW_FRACTION * peak)
    if xs.size == 0:
        half = (cut.shape[0] - 1) / 2.0
        return half, half
    return float(np.mean(xs)), float(np.mean(ys))


def _line_profile(
    cut: np.ndarray,
    center_x: float,
    center_y: float,
    angle: float,
) -> np.ndarray:
    """Sample the box along a line through the centre, one pixel per step."""
    side = cut.shape[0]
    steps = np.arange(-side, side + 1, dtype=float)
    xs = center_x + steps * np.cos(angle)
    ys = center_y + steps * np.sin(angle)
    inside = (xs >= 0) & (xs <= side - 1) & (ys >= 0) & (ys <= side - 1)
    return map_coordinates(
        np.asarray(cut, dtype=float), [ys[inside], xs[inside]], order=1
    )


def _profile_has_donut(profile: np.ndarray) -> bool:
    values = np.asarray(profile, dtype=float)
    if values.size < 5 or not np.all(np.isfinite(values)):
        return False
    # Fit only across the star, not the flat background the box also holds,
    # or a small ring blurs into one broad hump.
    bright = np.flatnonzero(values >= _DONUT_WINDOW_FRACTION * float(np.max(values)))
    if bright.size == 0:
        return False
    low = max(int(bright[0]) - 1, 0)
    high = min(int(bright[-1]) + 1, values.size - 1)
    values = values[low : high + 1]
    if values.size < 5:
        return False
    # Scale positions to [-1, 1] so the quartic fit is well conditioned.
    u = np.linspace(-1.0, 1.0, values.size)
    coeffs = np.polyfit(u, values, 4)
    slope = np.polyder(coeffs)
    curvature = np.polyder(slope)
    roots = np.roots(slope)
    roots = np.sort(roots[np.abs(roots.imag) < 1e-9].real)
    roots = roots[(roots > -1.0) & (roots < 1.0)]
    kinds = ["min" if np.polyval(curvature, r) > 0 else "max" for r in roots]
    for index, kind in enumerate(kinds):
        if kind == "min" and "max" in kinds[:index] and "max" in kinds[index + 1 :]:
            return True
    return False


def _gaussian_r_squared(
    cut: np.ndarray,
    amplitude: float,
    gauss_x: float,
    gauss_y: float,
    sigma_x: float,
    sigma_y: float,
) -> float:
    side = cut.shape[0]
    yy, xx = np.mgrid[0:side, 0:side]
    model = amplitude * np.exp(
        -0.5 * (((xx - gauss_x) / sigma_x) ** 2 + ((yy - gauss_y) / sigma_y) ** 2)
    )
    data = np.asarray(cut, dtype=float)
    total = float(np.sum((data - float(np.mean(data))) ** 2))
    if total <= 0:
        return float("nan")
    return 1.0 - float(np.sum((data - model) ** 2)) / total


def _curve_samples(
    positions: np.ndarray,
    center: float,
    quadratic: float,
    linear: float,
    constant: float,
) -> list[tuple[float, float]]:
    xs = np.linspace(float(np.min(positions)), float(np.max(positions)), 40)
    ys = quadratic * (xs - center) ** 2 + linear * (xs - center) + constant
    return [(float(x), float(y)) for x, y in zip(xs, ys)]
