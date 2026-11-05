
from __future__ import annotations

import argparse
import sys

from autofocuser.measure import MIN_R_SQUARED, MeasureError
from autofocuser.process import collect_fits, run_fits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure focus from a set of FITS frames.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="FITS files, or a folder of them. Folder contents are sorted by name.",
    )
    parser.add_argument(
        "--box-size",
        type=int,
        default=None,
        help=(
            "Side of the square measurement box, in pixels. "
            "Omit it to size the box to the star."
        ),
    )
    args = parser.parse_args(argv)

    try:
        paths = collect_fits(args.paths)
        result = run_fits(paths, box_size=args.box_size)
    except MeasureError as err:
        print(err, file=sys.stderr)
        return 1

    for frame in result.frames:
        measured = frame.measurement
        if measured.donut:
            print(
                f"{frame.path.name} focus={frame.focus} "
                f"DONUTS: the star has a dip in the middle. "
                f"No Gaussian fit; left out of the parabola. "
                f"star=({measured.x:.1f},{measured.y:.1f})"
            )
            continue
        note = "" if measured.usable else (
            f" R^2 below {MIN_R_SQUARED}; left out of the parabola."
        )
        print(
            f"{frame.path.name} focus={frame.focus} "
            f"sigma={measured.sigma:.3f} "
            f"sx={measured.sigma_x:.3f} sy={measured.sigma_y:.3f} "
            f"r2={measured.r_squared:.3f} "
            f"box={measured.box_size} "
            f"star=({measured.x:.1f},{measured.y:.1f}){note}"
        )
    if result.recommendation is None:
        print("No frames left for the parabola fit.")
        return 1
    print(result.recommendation.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
