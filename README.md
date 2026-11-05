# WIP — Autofocuser

Measure focus from a set of FITS frames. Yet to be implemented into evora server. This folder only runs the math. It does not talk to the camera or move the focuser.

Each frame needs a `FOCUS` value in the header. That value is the focus position, in steps.

## Methodology

Background is subtracted, then the brightest star in the frame is found. The same star is found again on every frame. If it moves more than 32 pixels from the previous frame, the run stops.

The measurement box is a square sized to that star. Its radius is a Kron aperture: the first moment of the light inside 6 detection radii, multiplied by 2.5, and never smaller than 3.5 detection radii. For a Gaussian that holds more than 99% of the light. Two pixels of padding are added past the aperture. Other detected objects are left out of the size, so a neighbor does not grow the box. Pass `--box-size` to force one square side for every frame.

Inside the box, four cuts through the star (at 0°, 45°, 90° and 135°) are each fit with a 4th degree polynomial, using only the stretch between the outermost pixels that reach 20% of that cut's peak. The cuts cross at the centre of the star's outline: the unweighted mean position of the pixels above 20% of the box peak. A brightness-weighted centre would be pulled onto the bright knot of a lopsided donut, and the cuts would miss its dark middle. If any fit has a local minimum with a local maximum on each side (a dip in the middle of the star), the frame is reported as donuts and left out of the parabola. The minima a quartic draws on the flanks of an ordinary star do not count.

Otherwise a 2D Gaussian is fit with the rotation fixed at zero. Its R² over the whole box is reported. Frames with R² below 0.7 are left out of the parabola.

\[
G(x,y) = A \exp\left(-\frac{1}{2}\left[\left(\frac{x-x_0}{\sigma_x}\right)^2 + \left(\frac{y-y_0}{\sigma_y}\right)^2\right]\right)
\]

The width used for focusing is the mean of the two axes, \((\sigma_x + \sigma_y)/2\).

Those widths are plotted against the `FOCUS` positions and fit with a parabola, in coordinates centered on the mean position. If the parabola opens upward and its vertex lies inside the sweep, the recommendation is the vertex rounded to the nearest step. Otherwise the recommendation is the sampled position with the lowest width. The result is then reported. Need to add evora-client side display of focus sigma parabola.

## Run

Python 3.12 or newer. From this folder:

```powershell
pip install -e .
python -m autofocuser C:\path\to\frames
```

A folder is read in filename order. Individual files are used in the order given. That order is the order of the shift check.

```powershell
python -m autofocuser C:\path\to\one.fits C:\path\to\two.fits --box-size 32
```

To write the detection, Gaussian, cross-section, and parabola images into a `debug run` folder next to the FITS files, install `matplotlib` and run:

```powershell
python debug_run.py C:\path\to\frames
```
