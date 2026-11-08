"""
run_all.py
ONE command runs the whole homework:

    python run_all.py

Steps for every photo in data/original/:
  1. load it and make it smaller (reference image)
  2. make a degraded copy (lower contrast + Gaussian noise)
  3. apply the 4 methods to the degraded copy
  4. compute PSNR, SSIM, MAE against the reference
  5. save result images, figures, and results/metrics.csv
"""

import csv
import sys
from pathlib import Path

import cv2

# Let Python find our files inside the src/ folder.
PROJECT_FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_FOLDER / "src"))

from degrade import load_image, resize_longest_side, degrade_image
from methods import baseline, contrast_stretch, equalize_lightness, denoise_then_equalize
from metrics import psnr, ssim, mean_absolute_error
from figures import save_side_by_side, save_summary


# =====================================================================
# SETTINGS (all in one place, so they are easy to write in the report)
# =====================================================================

# Image size: the longest side is resized to this many pixels.
MAX_IMAGE_SIDE = 800

# Degradation (the same for every image and every method)
CONTRAST_FACTOR = 0.5   # 1.0 = no change, 0.5 = half the contrast
NOISE_SIGMA = 10.0      # std of Gaussian noise, on the 0-255 scale
RANDOM_SEED = 42        # fixed seed -> the same noise every run

# Method 2: linear contrast stretching on L (luminance)
STRETCH_LOW_PERCENTILE = 1    # values below this percentile -> 0
STRETCH_HIGH_PERCENTILE = 99  # values above this percentile -> 255

# Method 4: non-local means denoising (then equalization on L)
# We tried h = 3, 5, 7 and 10. With 10 the trees and the gravel became flat,
# like a painting; with 3 a lot of grain was left. 5 was the best balance.
NLM_STRENGTH = 5         # h: bigger = more smoothing (and more lost detail)
NLM_COLOR_STRENGTH = 5   # hColor: the same idea for the color channels
NLM_TEMPLATE_SIZE = 7    # patch size in pixels (odd number)
NLM_SEARCH_SIZE = 21     # search window size in pixels (odd number)

# Summary figures: one small square of each photo is also shown bigger
# (zoom), so the noise and the lost details are easy to see.
# The numbers are parts of the image size: 0.20 = 20% from the left edge.
ZOOM_LEFT = 0.20   # left side of the square (part of the image width)
ZOOM_TOP = 0.55    # top side of the square (part of the image height)
ZOOM_SIZE = 0.40   # side of the square (part of the image width)

# Folders
ORIGINAL_FOLDER = PROJECT_FOLDER / "data" / "original"
REFERENCE_FOLDER = PROJECT_FOLDER / "data" / "reference"
DEGRADED_FOLDER = PROJECT_FOLDER / "data" / "degraded"
RESULT_IMAGE_FOLDER = PROJECT_FOLDER / "results" / "images"
FIGURE_FOLDER = PROJECT_FOLDER / "results" / "figures"
METRICS_FILE = PROJECT_FOLDER / "results" / "metrics.csv"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


# =====================================================================
# FUNCTIONS
# =====================================================================

def find_original_images():
    """Return a sorted list of all photos in data/original/.
    Sorting makes the order the same on every computer."""
    paths = []
    for path in sorted(ORIGINAL_FOLDER.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)
    return paths


def make_folders():
    """Create the output folders if they do not exist yet."""
    for folder in [REFERENCE_FOLDER, DEGRADED_FOLDER, RESULT_IMAGE_FOLDER, FIGURE_FOLDER]:
        folder.mkdir(parents=True, exist_ok=True)


def run_all_methods(degraded):
    """Apply the 4 methods to one degraded image.
    Returns a dictionary {method name: result image}."""
    results = {}
    results["baseline"] = baseline(degraded)
    results["contrast_stretch"] = contrast_stretch(
        degraded, STRETCH_LOW_PERCENTILE, STRETCH_HIGH_PERCENTILE
    )
    results["hist_eq_L"] = equalize_lightness(degraded)
    results["nlm_then_hist_eq_L"] = denoise_then_equalize(
        degraded, NLM_STRENGTH, NLM_COLOR_STRENGTH, NLM_TEMPLATE_SIZE, NLM_SEARCH_SIZE
    )
    return results


def compute_metrics(reference, result):
    """All three metrics for one result, in a small dictionary."""
    return {
        "psnr": psnr(reference, result),
        "ssim": ssim(reference, result),
        "mae": mean_absolute_error(reference, result),
    }


def process_one_image(image_path):
    """Do all steps for one photo. Returns a list of rows for the CSV."""
    image_name = image_path.stem  # file name without ".jpg"
    print("Processing", image_name, "...")

    # 1. Reference image: the original, made smaller.
    #    We save it as PNG because PNG has no compression loss.
    reference = resize_longest_side(load_image(image_path), MAX_IMAGE_SIDE)
    cv2.imwrite(str(REFERENCE_FOLDER / (image_name + ".png")), reference)

    # 2. Degraded image (same settings and same seed for every image).
    degraded = degrade_image(reference, CONTRAST_FACTOR, NOISE_SIGMA, RANDOM_SEED)
    cv2.imwrite(str(DEGRADED_FOLDER / (image_name + ".png")), degraded)

    # 3. Run the methods.
    results = run_all_methods(degraded)

    # 4 + 5. Metrics, result images, side-by-side figures.
    metrics_by_method = {}
    csv_rows = []
    for method_name, result in results.items():
        values = compute_metrics(reference, result)
        metrics_by_method[method_name] = values

        result_file = RESULT_IMAGE_FOLDER / (image_name + "_" + method_name + ".png")
        cv2.imwrite(str(result_file), result)

        figure_file = FIGURE_FOLDER / (image_name + "_" + method_name + ".png")
        save_side_by_side(reference, degraded, result, method_name, values, figure_file)

        csv_rows.append({
            "image": image_name,
            "method": method_name,
            "psnr": round(values["psnr"], 4),
            "ssim": round(values["ssim"], 4),
            "mae": round(values["mae"], 4),
        })

    # One summary figure with all methods (and a zoomed square).
    summary_file = FIGURE_FOLDER / (image_name + "_summary.png")
    zoom = (ZOOM_LEFT, ZOOM_TOP, ZOOM_SIZE)
    save_summary(reference, results, metrics_by_method, image_name, summary_file, zoom)

    return csv_rows


def save_metrics_csv(all_rows):
    """Write all metrics into results/metrics.csv."""
    with open(METRICS_FILE, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image", "method", "psnr", "ssim", "mae"])
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


def main():
    make_folders()
    image_paths = find_original_images()
    if len(image_paths) == 0:
        print("No photos found. Put 3-5 photos into", ORIGINAL_FOLDER)
        return

    all_rows = []
    for image_path in image_paths:
        rows = process_one_image(image_path)
        all_rows.extend(rows)

    save_metrics_csv(all_rows)
    print("Done. Metrics saved to", METRICS_FILE)


if __name__ == "__main__":
    main()
