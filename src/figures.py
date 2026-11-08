"""
figures.py
Make comparison pictures with matplotlib.

Important: OpenCV images are BGR, but matplotlib expects RGB.
If we forget to convert, red and blue are swapped in the figure.
"""

import cv2
import matplotlib

# "Agg" draws figures straight into files without opening a window.
# This must be set before importing pyplot.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def bgr_to_rgb(image_bgr):
    """Change channel order from BGR (OpenCV) to RGB (matplotlib)."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def show_in_axis(axis, image_bgr, title, title_size=10):
    """Draw one image in one panel, with a title and no axis ticks.

    vmin=0 and vmax=255 make sure every panel uses the same scale.
    interpolation="nearest" means matplotlib does not smooth pixels.
    title_size is the font size of the title (in points).
    """
    axis.imshow(bgr_to_rgb(image_bgr), vmin=0, vmax=255, interpolation="nearest")
    axis.set_title(title, fontsize=title_size)
    axis.axis("off")


def save_side_by_side(original, degraded, result, method_name,
                      result_metrics, output_path):
    """Save one figure: original | degraded | result.

    All three images have the same size, and each panel has the same
    size, so they are easy to compare.
    """
    # constrained_layout=True makes room for all titles, so nothing is cut off.
    figure, axes = plt.subplots(1, 3, figsize=(12, 5.8), constrained_layout=True)

    show_in_axis(axes[0], original, "Original (reference)")
    show_in_axis(axes[1], degraded, "Degraded (input)")

    result_title = (
        "Result: " + method_name + "\n"
        + "PSNR {:.2f} dB | SSIM {:.3f} | MAE {:.2f}".format(
            result_metrics["psnr"], result_metrics["ssim"], result_metrics["mae"]
        )
    )
    show_in_axis(axes[2], result, result_title)

    # bbox_inches="tight" cuts away empty white borders, and no title is cut off.
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)  # free memory, otherwise many open figures pile up


def zoom_square(image, zoom):
    """Change the zoom settings (parts of the image size) into pixels.

    zoom = (left, top, size), for example (0.20, 0.55, 0.40).
    Returns x, y (the top-left corner) and the side of the square.
    """
    height, width = image.shape[:2]
    left, top, size = zoom
    x = int(left * width)
    y = int(top * height)
    side = int(size * width)
    return x, y, side


def save_summary(original, results_by_method, metrics_by_method,
                 image_name, output_path, zoom):
    """Save one figure with the original and ALL methods.

    Top row: the whole images. The white square shows the zoom area.
    Bottom row: that square, bigger, so noise and small details are easy to see.

    results_by_method: dictionary {method name: result image}
    metrics_by_method: dictionary {method name: {"psnr":..,"ssim":..,"mae":..}}
    zoom: (left, top, size), see zoom_square()
    """
    # A list of (title, image): first the original, then every method.
    panels = [("Original (reference)", original)]
    for method_name, result in results_by_method.items():
        values = metrics_by_method[method_name]
        title = method_name + "\nPSNR {:.2f} | SSIM {:.3f}\nMAE {:.2f}".format(
            values["psnr"], values["ssim"], values["mae"]
        )
        panels.append((title, result))

    # Big fonts: this figure is very wide (5 panels), so in the PDF report
    # it is shrunk to about 1/3 of its size. Size 10 would become too small to read.
    panel_title_size = 17
    figure_title_size = 21

    x, y, side = zoom_square(original, zoom)
    figure, axes = plt.subplots(2, len(panels), figsize=(3.6 * len(panels), 9.6),
                                constrained_layout=True)

    for column, (title, image) in enumerate(panels):
        # top row: the whole image, with a white square around the zoom area
        show_in_axis(axes[0, column], image, title, panel_title_size)
        square = plt.Rectangle((x, y), side, side, fill=False, edgecolor="white", linewidth=1.5)
        axes[0, column].add_patch(square)
        # bottom row: only the square (rows y to y+side, columns x to x+side)
        show_in_axis(axes[1, column], image[y:y + side, x:x + side], "Zoom", panel_title_size)

    figure.suptitle("All methods - image " + image_name, fontsize=figure_title_size)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
