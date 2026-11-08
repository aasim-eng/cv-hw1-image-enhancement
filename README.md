# Homework 1 – Image Enhancement (classical methods)

Computer Vision, Fall 2026, University of Verona (Prof. Francesco Setti).
Author: Mohammad Aasim (student number VR558207).

A Python pipeline that damages 3 real night photos (lower contrast + noise),
enhances them with 4 classical methods, and compares the results with
figures and with numbers (PSNR, SSIM, MAE).
No deep learning, no neural networks, no pre-trained models.

The report is `report/HW1-Report.pdf`.

## Requirements

- Python 3.11 (the author uses Python 3.11.9 on a Mac)
- Libraries (exact versions in `requirements.txt`):
  OpenCV, NumPy, scikit-image (only for SSIM), matplotlib (only for figures)

Install (only the first time):

```bash
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to run

```bash
python run_all.py
```

This one command builds everything from zero: reference images, degraded
images, result images, all figures, and `results/metrics.csv`.
It takes about 20 seconds for 3 images.

## File structure

```
HW1-Image-Enhancement/
  run_all.py            ONE command runs everything; all settings are at the top
  requirements.txt      library versions
  IMAGE_SOURCES.md      source and license of each photo
  README.md             this file
  .gitignore            files that do not go to GitHub
  src/
    degrade.py          load, resize, lower contrast, add noise
    methods.py          the 4 enhancement methods
    metrics.py          PSNR, SSIM, MAE
    figures.py          side-by-side and summary figures
  data/
    original/           the real photos 1.jpg, 2.jpg, 4.jpg (input, full size)
    reference/          originals resized to 600x800 (made by the code)
    degraded/           damaged copies (made by the code)
  results/
    images/             <image>_<method>.png
    figures/            <image>_<method>.png  (original | degraded | result)
                        <image>_summary.png   (all methods together; bottom row = zoom)
    metrics.csv         columns: image, method, psnr, ssim, mae
  report/
    HW1-Report.pdf      the report
    report.html         the text of the report (the PDF is made from it)
    figures/            JPEG copies of the 3 summary figures (smaller, for the PDF)
```

## Settings (top of `run_all.py`)

| Setting | Value | Meaning |
|---|---|---|
| MAX_IMAGE_SIDE | 800 | longest side after resize (photos become 600 x 800) |
| CONTRAST_FACTOR | 0.5 | `degraded = 128 + 0.5 * (image - 128)` (half the contrast) |
| NOISE_SIGMA | 10 | std of Gaussian noise, on the 0–255 scale |
| RANDOM_SEED | 42 | fixed seed; the same seed is used for every image |
| STRETCH_LOW / HIGH_PERCENTILE | 1 / 99 | range used by contrast stretching |
| NLM_STRENGTH (h) | 5 | non-local means strength for lightness |
| NLM_COLOR_STRENGTH (hColor) | 5 | non-local means strength for color |
| NLM_TEMPLATE_SIZE | 7 | patch size |
| NLM_SEARCH_SIZE | 21 | search window size |
| ZOOM_LEFT / TOP / SIZE | 0.20 / 0.55 / 0.40 | the zoomed square in the summary figures (parts of the image size) |

The same images and the same degradation settings are used for all methods.
Degradation order: lower contrast → add noise → clip to 0–255 → uint8.

## Methods

1. **baseline** – the degraded image, no processing.
2. **contrast_stretch** – linear contrast stretching **on luminance**
   (the L channel of CIE-Lab), not per channel. Reason: stretching B, G, R
   separately can change colors. The 1st and 99th percentiles of L are
   mapped to 0 and 255 (percentiles instead of min/max because noise
   creates a few extreme pixels).
3. **hist_eq_L** – histogram equalization (`cv2.equalizeHist`) on the
   L channel of CIE-Lab; the a and b color channels are not changed.
4. **nlm_then_hist_eq_L** – non-local means denoising
   (`cv2.fastNlMeansDenoisingColored`, h = hColor = 5) followed by histogram
   equalization on L. Reason: equalization makes noise stronger, so we remove
   noise first. Note: denoising can also smooth away small details.
   We tried h = 3, 5, 7 and 10: with 10 the trees and the gravel became flat,
   like a painting; with 3 a lot of grain was left. To see this yourself,
   change `NLM_STRENGTH` and `NLM_COLOR_STRENGTH` in `run_all.py` and run again.

## Metrics

All metrics compare each result with the reference (resized original).

- **PSNR** (dB) – higher is better. Written by hand; checked against `cv2.PSNR`.
- **SSIM** – higher is better (1 = same structure). From scikit-image,
  computed per color channel and averaged.
- **MAE** – lower is better (0 = identical). Written by hand.

## Important note about the reference

We can compare with the original only because **we made the damage
ourselves** (synthetic degradation). In real life, when a photo is dark
or noisy, a clean "correct" version usually does not exist, so these
full-reference metrics cannot be used directly.
Also, the originals here are night photos. A method that makes the image
much brighter than the original can look clearer but get worse numbers,
because the numbers measure "how close to the original", not "how nice".

## Reproducibility

Running `python run_all.py` twice gives an identical `metrics.csv` and
identical result images (checked with file hashes).

## Images

3 own photos, license CC BY 4.0. Details: `IMAGE_SOURCES.md`.
