from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))

from autofocuser.measure import MIN_R_SQUARED, _extract, measure_star, recommend_focus


def save_detection(folder: Path, frame: dict, measured) -> None:
    image = frame["image"]
    norm = ImageNormalize(image, interval=ZScaleInterval(), stretch=AsinhStretch())
    figure, axis = plt.subplots(figsize=(7.2, 7.2), dpi=130)
    axis.imshow(
        image,
        origin="lower",
        cmap="inferno",
        norm=norm,
        interpolation="nearest",
    )
    axis.add_patch(
        Rectangle(
            (measured.box_origin_x, measured.box_origin_y),
            measured.box_size,
            measured.box_size,
            fill=False,
            edgecolor="#ff1f1f",
            linewidth=1.6,
        )
    )
    axis.set_xlim(0, image.shape[1])
    axis.set_ylim(0, image.shape[0])
    axis.set_xlabel("x (pixels)")
    axis.set_ylabel("y (pixels)")
    axis.set_title(
        f"{frame['path'].name}    FOCUS {frame['focus']}    "
        f"{frame['exposure']:.0f} s    box {measured.box_size}"
    )
    figure.tight_layout()
    figure.savefig(folder / f"{frame['path'].stem}_focus_{frame['focus']}.png")
    plt.close(figure)


def _amplitude_text(value: float) -> str:
    mantissa, exponent = f"{value:.3e}".split("e")
    return rf"{float(mantissa):.3g}\times 10^{{{int(exponent)}}}"


def save_gaussian(folder: Path, frame: dict, measured) -> None:
    span = max(32.0, 8.0 * max(measured.sigma_x, measured.sigma_y))
    x0 = max(0.0, measured.gauss_x - span / 2.0)
    x1 = min(measured.box_size - 1.0, measured.gauss_x + span / 2.0)
    y0 = max(0.0, measured.gauss_y - span / 2.0)
    y1 = min(measured.box_size - 1.0, measured.gauss_y + span / 2.0)
    xx, yy = np.meshgrid(np.linspace(x0, x1, 80), np.linspace(y0, y1, 80))
    model = measured.amplitude * np.exp(
        -0.5
        * (
            ((xx - measured.gauss_x) / measured.sigma_x) ** 2
            + ((yy - measured.gauss_y) / measured.sigma_y) ** 2
        )
    )
    figure = plt.figure(figsize=(8.4, 6.6), dpi=130)
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_surface(
        xx,
        yy,
        model,
        cmap="inferno",
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
    )
    axis.set_xlabel("x (box pixels)")
    axis.set_ylabel("y (box pixels)")
    axis.set_zlabel("fitted signal")
    axis.set_title(
        f"{frame['path'].name}    FOCUS {frame['focus']}    {frame['exposure']:.0f} s",
        loc="left",
    )
    axis.view_init(elev=28, azim=-58)
    figure.text(
        0.98,
        0.97,
        "\n".join(
            [
                rf"$\sigma=(\sigma_x+\sigma_y)/2={measured.sigma:.3f}$ px",
                (
                    rf"$G(x,y)={_amplitude_text(measured.amplitude)}\,"
                    rf"\exp\left(-\left["
                    rf"\frac{{(x-{measured.gauss_x:.2f})^2}}"
                    rf"{{(2\times {measured.sigma_x:.3f}^2)}}"
                    rf"+\frac{{(y-{measured.gauss_y:.2f})^2}}"
                    rf"{{(2\times {measured.sigma_y:.3f}^2)}}"
                    rf"\right]\right)$"
                ),
            ]
        ),
        ha="right",
        va="top",
        fontsize=11,
    )
    figure.subplots_adjust(left=0.02, right=0.96, top=0.78, bottom=0.06)
    figure.savefig(folder / f"{frame['path'].stem}_focus_{frame['focus']}.png")
    plt.close(figure)


def _r_squared(data: np.ndarray, model: np.ndarray) -> float:
    residual = np.asarray(data, dtype=float) - np.asarray(model, dtype=float)
    total = np.asarray(data, dtype=float) - float(np.mean(data))
    denominator = float(np.sum(total**2))
    if denominator <= 0:
        return float("nan")
    return 1.0 - float(np.sum(residual**2)) / denominator


def _gaussian_at(xs: np.ndarray, ys: np.ndarray, measured) -> np.ndarray:
    return measured.amplitude * np.exp(
        -0.5
        * (
            ((xs - measured.gauss_x) / measured.sigma_x) ** 2
            + ((ys - measured.gauss_y) / measured.sigma_y) ** 2
        )
    )


def save_cross_section(folder: Path, frame: dict, measured, signal: np.ndarray) -> None:
    side = measured.box_size
    x0 = measured.box_origin_x
    y0 = measured.box_origin_y
    raw = np.asarray(frame["image"], dtype=np.float64)[y0 : y0 + side, x0 : x0 + side]
    cut = np.asarray(signal, dtype=np.float64)[y0 : y0 + side, x0 : x0 + side]
    row = int(np.clip(round(measured.gauss_y), 0, side - 1))
    col = int(np.clip(round(measured.gauss_x), 0, side - 1))
    pixels = np.arange(side, dtype=float)
    model_row = _gaussian_at(pixels, np.full(side, row), measured)
    model_col = _gaussian_at(np.full(side, col), pixels, measured)

    yy, xx = np.mgrid[0:side, 0:side]
    model_box = _gaussian_at(xx, yy, measured)
    span = max(12.0, 6.0 * max(measured.sigma_x, measured.sigma_y))
    row_window = np.abs(pixels - measured.gauss_x) <= span
    col_window = np.abs(pixels - measured.gauss_y) <= span
    core = (xx - measured.gauss_x) ** 2 / measured.sigma_x**2 + (
        yy - measured.gauss_y
    ) ** 2 / measured.sigma_y**2 <= 1.0
    peak = float(raw.max())
    near_peak = int(np.count_nonzero(raw >= 0.98 * peak))
    core_residual = float(np.median(np.abs(cut[core] - model_box[core])))
    core_fraction = core_residual / measured.amplitude
    r2_row = _r_squared(cut[row, row_window], model_row[row_window])
    r2_col = _r_squared(cut[col_window, col], model_col[col_window])
    r2_box = _r_squared(cut, model_box)

    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), dpi=130)
    panels = (
        (axes[0, 0], axes[1, 0], cut[row], model_row, row_window, float(row), "horizontal"),
        (axes[0, 1], axes[1, 1], cut[:, col], model_col, col_window, float(col), "vertical"),
    )
    for top, bottom, samples, model, window, fixed, name in panels:
        fine = np.linspace(float(pixels[window][0]), float(pixels[window][-1]), 400)
        if name == "horizontal":
            curve = _gaussian_at(fine, np.full(fine.shape, fixed), measured)
        else:
            curve = _gaussian_at(np.full(fine.shape, fixed), fine, measured)
        top.plot(pixels[window], samples[window], "o", ms=3.5, color="#222", label="pixels")
        top.plot(fine, curve, color="#d17d00", lw=2, label="Gaussian")
        top.set_ylabel("ADU above background")
        top.legend(frameon=False, fontsize=8)
        bottom.axhline(0, color="#888", lw=0.8)
        bottom.plot(
            pixels[window],
            samples[window] - model[window],
            "o",
            ms=3.5,
            color="#9b1d1d",
        )
        bottom.set_xlabel(f"{name} pixel in the box")
        bottom.set_ylabel("residual ADU")

    figure.suptitle(
        f"{frame['path'].name}    FOCUS {frame['focus']}    "
        f"{frame['exposure']:.0f} s    box {side}\n"
        f"cut R²  x={r2_row:.3f}   y={r2_col:.3f}    full-box R²={r2_box:.3f}"
        f"    core residual {core_fraction * 100:.1f}% of amplitude"
        f"    peak {peak:.0f} ADU, {near_peak} pixels within 2%"
    )
    figure.tight_layout()
    figure.savefig(folder / f"{frame['path'].stem}_focus_{frame['focus']}_cut.png")
    plt.close(figure)


def save_parabola(folder: Path, frames: list[dict]) -> None:
    positions = [frame["focus"] for frame in frames]
    sigmas = [frame["sigma"] for frame in frames]
    recommendation = recommend_focus(positions, sigmas)
    x = np.asarray(positions, dtype=float)
    y = np.asarray(sigmas, dtype=float)
    center = float(np.mean(x))
    quadratic, linear, _constant = (float(value) for value in np.polyfit(x - center, y, 2))
    vertex = None
    if quadratic > 0:
        vertex = center - linear / (2.0 * quadratic)

    figure, axis = plt.subplots(figsize=(8.2, 5.6), dpi=140)
    point_colors = ["#00c3d1", "#7aa2ff", "#c06bff"]
    exposures = sorted({frame["exposure"] for frame in frames})
    for index, exposure in enumerate(exposures):
        chosen = np.asarray([frame["exposure"] == exposure for frame in frames])
        axis.scatter(
            x[chosen],
            y[chosen],
            s=46,
            color=point_colors[index % len(point_colors)],
            label=f"{exposure:.0f} s exposure",
            zorder=3,
        )
    curve_x = np.asarray([point[0] for point in recommendation.curve])
    curve_y = np.asarray([point[1] for point in recommendation.curve])
    axis.plot(curve_x, curve_y, color="#d17d00", linewidth=2, label="parabola")
    best = recommendation.position
    best_sigma = float(y[int(np.argmin(np.abs(x - best)))])
    axis.axvline(best, color="#5bcc09", linewidth=1.4, label=f"best focus {best}")
    axis.scatter([best], [best_sigma], s=80, color="#5bcc09", zorder=4)
    axis.set_xlabel("Focus position (steps)")
    axis.set_ylabel("Sigma (pixels)")
    outside = vertex is not None and (
        vertex < float(np.min(x)) or vertex > float(np.max(x))
    )
    if vertex is None:
        title = f"Best focus {recommendation.position} steps, the lowest measured sigma"
    elif outside:
        title = (
            f"Best focus in the sweep: {best} steps\n"
            f"Fitted vertex {int(round(vertex))} is outside these images"
        )
    else:
        title = f"Best focus by the fit: {best} steps"
    axis.set_title(title)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(folder / "focus_parabola.png")
    plt.close(figure)
    print(
        f"quadratic={quadratic:.6g} vertex={vertex} "
        f"recommended={recommendation.position} curve_ok={recommendation.curve_ok}"
    )
    print(recommendation.message)


def load_frames(source: Path) -> list[dict]:
    frames = []
    paths = sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in {".fits", ".fit"}
    )
    for path in paths:
        with fits.open(path) as hdul:
            header = hdul[0].header
            image = np.asarray(hdul[0].data)
        frames.append(
            {
                "path": path,
                "image": image,
                "focus": int(round(float(header["FOCUS"]))),
                "exposure": float(header["EXPTIME"]),
            }
        )
    if not frames:
        raise SystemExit(f"No FITS files to plot in {source}.")
    return frames


def run_folder(source: Path) -> None:
    run = source / "debug run"
    detection = run / "detection"
    gaussian = run / "gaussian"
    cross_section = run / "cross section"
    parabola = run / "parabola"
    for folder in (detection, gaussian, cross_section, parabola):
        folder.mkdir(parents=True, exist_ok=True)

    frames = load_frames(source)
    previous = None
    for frame in frames:
        measured = measure_star(frame["image"], previous)
        previous = (measured.x, measured.y)
        frame["sigma"] = measured.sigma
        frame["usable"] = measured.usable
        save_detection(detection, frame, measured)
        if measured.donut:
            print(
                f"{frame['path'].name} focus={frame['focus']} "
                f"DONUTS: no Gaussian fit; left out of the parabola."
            )
            continue
        signal, _sources, _segmap = _extract(frame["image"])
        save_gaussian(gaussian, frame, measured)
        save_cross_section(cross_section, frame, measured, signal)
        note = "" if measured.usable else (
            f" R^2 below {MIN_R_SQUARED}; left out of the parabola."
        )
        print(
            f"{frame['path'].name} focus={frame['focus']} "
            f"sigma={measured.sigma:.3f} r2={measured.r_squared:.3f} "
            f"box={measured.box_size}{note}"
        )
    usable = [frame for frame in frames if frame["usable"]]
    if not usable:
        print("No frames left for the parabola fit.")
        return
    save_parabola(parabola, usable)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a debug focus run.")
    parser.add_argument("source", type=Path, help="Folder of FITS files.")
    args = parser.parse_args()
    run_folder(args.source)


if __name__ == "__main__":
    main()
