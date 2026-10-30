import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_gmm_results(image, results):

    fig, ax = plt.subplots(figsize=(12, 10))

    log_image = np.log1p(np.maximum(image, 0))

    ax.imshow(
        log_image,
        cmap="grey",
        origin="upper"
    )

    # Extract integrated intensities
    intensities = np.array([
        spot["integrated_intensity"]
        for spot in results["spots"]
    ], dtype=float)

    # Normalize intensities robustly
    vmin, vmax = np.percentile(intensities, [5, 95])

    if vmax > vmin:
        normalized = np.clip(
            (intensities - vmin) / (vmax - vmin),
            0, 1
        )
    else:
        normalized = np.ones_like(intensities)

    # Opacity range
    alpha_min = 0.15
    alpha_max = 0.9

    alphas = alpha_min + normalized * (alpha_max - alpha_min)

    for i, (spot, alpha) in enumerate(zip(results["spots"], alphas)):

        cx = spot["center_x"]
        cy = spot["center_y"]

        ellipse = Ellipse(
            xy=(cx, cy),
            width=spot["sigma_major"],
            height=spot["sigma_minor"],
            angle=spot["orientation_deg"],
            facecolor="red",
            edgecolor="white",
            alpha=alpha,
            linewidth=1.8,
            zorder=3
        )

        ax.add_patch(ellipse)

        ax.scatter(
            cx, cy,
            color="white",
            marker="+",
            s=80,
            linewidths=2,
            zorder=4
        )

        ax.annotate(
            f"{spot['integrated_intensity']:.0f}",
            (cx, cy),
            xytext=(8, 8),
            textcoords="offset points",
            color="yellow",
            fontsize=9,
            fontweight="bold",
            zorder=5
        )

    ax.set_title("Diffraction spots - GMM covariance ellipses")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt


def plot_intensity_distribution(
    image,
    percentiles=(95, 97, 98, 99, 99.5, 99.9),
    bins=200,
    log_scale=False
):
    image = np.asarray(image, dtype=float)

    if image.ndim != 2:
        raise ValueError("Image must be 2D.")

    values = image[np.isfinite(image)].ravel()

    if len(values) == 0:
        raise ValueError("Image contains no finite pixels.")

    # Calculate percentile thresholds
    thresholds = {
        p: np.percentile(values, p)
        for p in percentiles
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Intensity histogram
    axes[0].hist(
        values,
        bins=bins,
        color="steelblue",
        alpha=0.75,
        density=True
    )

    for p, threshold in thresholds.items():
        axes[0].axvline(
            threshold,
            linestyle="--",
            linewidth=1.5,
            label=f"{p}% = {threshold:.2f}"
        )

    axes[0].set_title("Image intensity distribution")
    axes[0].set_xlabel("Pixel intensity")
    axes[0].set_ylabel("Density")

    if log_scale:
        axes[0].set_yscale("log")

    axes[0].legend(fontsize=8)

    # 2. Cumulative intensity distribution
    sorted_values = np.sort(values)
    cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)

    axes[1].plot(
        sorted_values,
        cumulative * 100,
        color="darkorange",
        linewidth=2
    )

    for p, threshold in thresholds.items():
        axes[1].axvline(
            threshold,
            linestyle="--",
            alpha=0.7,
            label=f"{p}%"
        )

    axes[1].set_title("Cumulative intensity distribution")
    axes[1].set_xlabel("Pixel intensity")
    axes[1].set_ylabel("Cumulative pixels (%)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()

    return thresholds