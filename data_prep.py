
import zipfile
import fabio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import re


def find_gmm_clusters(image, N=32, threshold_percentile=99,
                      intensity_power=1.0, random_state=42):

    image = np.asarray(image, dtype=float)

    if image.ndim != 2:
        raise ValueError("Image must be 2D.")

    valid = np.isfinite(image)
    if not valid.any():
        raise ValueError("Image contains no finite pixels.")

    height, width = image.shape
    threshold = np.percentile(image[valid], threshold_percentile)

    y, x = np.where(valid & (image > threshold))
    X = np.column_stack((x, y))
    intensities = image[y, x]

    if len(X) < N:
        raise ValueError("Not enough pixels for N clusters.")

    gmm = GaussianMixture(
        n_components=N, covariance_type="full",
        n_init=5, max_iter=300, reg_covar=1e-3,
        random_state=random_state
    ).fit(X)

    probabilities = gmm.predict_proba(X)
    labels = probabilities.argmax(axis=1)
    excess = np.maximum(intensities - threshold, 0)

    spots = []

    for k in range(N):
        weights = probabilities[:, k] * excess**intensity_power
        if weights.sum() <= 1e-12:
            weights = probabilities[:, k]

        center = np.average(X, axis=0, weights=weights)
        cov = gmm.covariances_[k]

        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        major, minor = np.sqrt(eigvals)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

        eccentricity = np.sqrt(max(0, 1 - eigvals[1] / (eigvals[0] + 1e-10)))
        intensity = np.sum(probabilities[:, k] * excess)

        spots.append({
            "center_x": center[0],
            "center_y": center[1],
            "gmm_center_x": gmm.means_[k, 0],
            "gmm_center_y": gmm.means_[k, 1],
            "covariance": cov,
            "sigma_major": major,
            "sigma_minor": minor,
            "fwhm_major": 2.35482 * major,
            "fwhm_minor": 2.35482 * minor,
            "orientation_deg": angle,
            "integrated_intensity": intensity,
            "mixture_weight": gmm.weights_[k],
            "eccentricity": eccentricity
        })

    cx, cy = width / 2, height / 2

    spots.sort(key=lambda s: np.arctan2(
        s["center_y"] - cy, s["center_x"] - cx
    ))

    features = []

    for s in spots:
        angle = np.radians(s["orientation_deg"])

        features.extend([
            (s["center_x"] - cx) / cx,
            (s["center_y"] - cy) / cy,
            s["sigma_major"] / width,
            s["sigma_minor"] / height,
            np.log1p(max(s["integrated_intensity"], 0)),
            np.sin(2 * angle),
            np.cos(2 * angle),
            s["eccentricity"]
        ])

    return {
        "spots": spots,
        "features": np.array(features, dtype=np.float32),
        "centers": gmm.means_,
        "covariances": gmm.covariances_,
        "labels": labels,
        "probabilities": probabilities,
        "coordinates": X,
        "threshold": threshold,
        "gmm": gmm
    }


def plot_gmm_results(image, results):

    fig, ax = plt.subplots(figsize=(12, 10))

    # Log-normalized original image
    log_image = np.log1p(np.maximum(image, 0))

    ax.imshow(log_image,cmap="viridis",origin="upper")

    for i, spot in enumerate(results["spots"]):

        cx = spot["center_x"]
        cy = spot["center_y"]

        # Fitted Gaussian ellipse (2 sigma)
        ellipse = Ellipse(xy=(cx, cy),width=4 * spot["sigma_major"],height=4 * spot["sigma_minor"],angle=spot["orientation_deg"],facecolor="red",edgecolor="white",alpha=0.30,linewidth=1.8,zorder=3)

        ax.add_patch(ellipse)

        # Center
        ax.scatter(cx, cy,color="white",marker="+",s=80,linewidths=2,zorder=4)

        # Intensity annotation
        ax.annotate(f"{spot['integrated_intensity']:.0f}",(cx, cy),xytext=(8, 8),textcoords="offset points",color="yellow",fontsize=9,fontweight="bold",zorder=5)

    ax.set_title("Diffraction spots - GMM covariance ellipses")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    plt.tight_layout()
    plt.show()


def main():

    zip_file_path = "Mainz_2022-10_AXO-Optics-Scan-forML.zip"
    accumulated_file_path = "combined_output.txt"

    # Open the ZIP archive
    with zipfile.ZipFile(zip_file_path, "r") as archive:
        # Loop through every file in the zip

        name_list = archive.namelist()

        i = 0

        for file_name in name_list:

            # Skip directories inside the zip
            if file_name.endswith('/'):
                continue

            if not any([x in file_name for x in ["130cm", "160cm"]]):
                continue

            if not "image" in file_name.split("/")[-1]:
                continue

            logfile = file_name.replace("image", "log").replace("cbf", "txt")

            if not logfile in name_list:
                raise FileNotFoundError("No logfile for image found")

            print(f"Processing: {file_name}")

            with archive.open(file_name, "r") as file_in_zip:

                # Load CBF file
                img = fabio.open(file_in_zip)

                # Extract image data as a NumPy array
                data = img.data
                data = np.log1p(np.maximum(data, 0))


                results = find_gmm_clusters(
                    data,
                    N=6,
                    threshold_percentile=97,
                    intensity_power=1.0
                )

                #plot_gmm_results(data, results)

                with archive.open(logfile, "r") as logfile_in_zip:
                    for line in logfile_in_zip.readlines():
                        line = line.decode("utf-8").strip()
                        if not any([x in line for x in ["hexapod x", "hexapod z"]]):
                            continue
                        print(line)


            i += 1
            if i > 1:
                break



    print(f"All done! Data accumulated into {accumulated_file_path}")

if __name__ == "__main__":
    main()
