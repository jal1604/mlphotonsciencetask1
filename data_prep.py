
import zipfile
import fabio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def find_gmm_clusters(
    image,
    N=32,
    threshold_percentile=99,
    intensity_power=1.0,
    random_state=42
):

    image = np.asarray(image, dtype=float)

    # Select high-intensity pixels
    valid = np.isfinite(image)

    threshold = np.percentile(
        image[valid], threshold_percentile
    )

    mask = valid & (image > threshold)

    y, x = np.where(mask)
    intensities = image[y, x]

    if len(x) < N:
        raise ValueError("Not enough pixels.")

    # Spatial coordinates only
    X = np.column_stack([x, y])

    # Fit standard GMM
    gmm = GaussianMixture(
        n_components=N,
        covariance_type="full",
        n_init=5,
        max_iter=300,
        reg_covar=1e-3,
        init_params="random_from_data",
        random_state=random_state
    )

    gmm.fit(X)

    # Membership probabilities
    probabilities = gmm.predict_proba(X)

    labels = np.argmax(probabilities, axis=1)

    # Extract fitted parameters
    spots = []

    for k in range(N):

        mean = gmm.means_[k]
        cov = gmm.covariances_[k]

        # Apply intensity weighting to reported measurements
        weights = probabilities[:, k] * (
            np.maximum(intensities - threshold, 0)
            ** intensity_power
        )

        if weights.sum() == 0:
            weights = probabilities[:, k]

        # Intensity-weighted center
        center = np.average(X, axis=0, weights=weights)

        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(cov)

        order = np.argsort(eigvals)[::-1]

        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        sigma_major = np.sqrt(eigvals[0])
        sigma_minor = np.sqrt(eigvals[1])

        angle = np.degrees(
            np.arctan2(
                eigvecs[1, 0],
                eigvecs[0, 0]
            )
        )

        spots.append({
            "center_x": center[0],
            "center_y": center[1],
            "gmm_center_x": mean[0],
            "gmm_center_y": mean[1],
            "covariance": cov,
            "sigma_major": sigma_major,
            "sigma_minor": sigma_minor,
            "fwhm_major": 2.35482 * sigma_major,
            "fwhm_minor": 2.35482 * sigma_minor,
            "orientation_deg": angle,
            "integrated_intensity": np.sum(weights),
            "mixture_weight": gmm.weights_[k]
        })

    return {
        "spots": spots,
        "centers": gmm.means_,
        "covariances": gmm.covariances_,
        "labels": labels,
        "probabilities": probabilities,
        "coordinates": X,
        "gmm": gmm,
        "threshold": threshold
    }


def find_intensity_clusters(
    image,
    N=10,
    threshold_percentile=99,
    spatial_weight=1.0,
    intensity_weight=2.0,
    random_state=42
):
    """
    Find N spatially distinct, high-intensity clusters in a 2D image.

    Parameters
    ----------
    image : np.ndarray
        2D intensity image.
    N : int
        Number of clusters.
    threshold_percentile : float
        Ignore pixels below this intensity percentile.
    spatial_weight : float
        Weight of spatial coordinates in clustering.
    intensity_weight : float
        Weight of intensity in clustering.
    random_state : int
        Random seed.

    Returns
    -------
    centers : np.ndarray
        Cluster centers (y, x).
    labels : np.ndarray
        Image-sized array with cluster labels (-1 = background).
    """

    image = np.asarray(image, dtype=float)

    if image.ndim != 2:
        raise ValueError("Image must be 2D.")

    # Ignore invalid pixels
    valid = np.isfinite(image)

    # Threshold high intensity pixels
    threshold = np.percentile(image[valid], threshold_percentile)
    mask = valid & (image >= threshold)

    y, x = np.where(mask)
    intensities = image[y, x]

    if len(x) < N:
        raise ValueError("Not enough pixels above threshold for N clusters.")

    # Normalize coordinates
    x_norm = (x - x.mean()) / (x.std() + 1e-10)
    y_norm = (y - y.mean()) / (y.std() + 1e-10)

    # Normalize intensity
    i_norm = (intensities - intensities.min()) / (
        intensities.max() - intensities.min() + 1e-10
    )

    # Feature space: spatial position + intensity
    features = np.column_stack([
        spatial_weight * x_norm,
        spatial_weight * y_norm,
        intensity_weight * i_norm
    ])

    # K-means
    kmeans = KMeans(
        n_clusters=N,
        n_init=20,
        random_state=random_state
    )

    cluster_labels = kmeans.fit_predict(features)

    # Create full-size label image
    label_image = np.full(image.shape, -1, dtype=int)
    label_image[y, x] = cluster_labels

    # Calculate actual spatial centers
    centers = np.array([
        [y[cluster_labels == i].mean(),
         x[cluster_labels == i].mean()]
        for i in range(N)
    ])

    return centers, label_image



def plot_clusters(image, centers, labels, sw, iv):

    fig, ax = plt.subplots(figsize=(10, 8))

    # Original image
    ax.imshow(image,
              cmap="viridis",
              #norm="log"
              )

    N = len(centers)

    # Calculate mean cluster intensities
    intensities = np.array([
        np.mean(image[labels == i])
        for i in range(N)
    ])

    # Scale circle sizes
    min_size = 50
    max_size = 1200

    normalized = (
        (intensities - intensities.min()) /
        (intensities.max() - intensities.min() + 1e-10)
    )

    sizes = min_size + normalized * (max_size - min_size)

    # Plot filled circles
    for i, (cy, cx) in enumerate(centers):

        ax.scatter(
            cx,
            cy,
            s=sizes[i],
            color="red",
            alpha=0.35,
            edgecolors="red",
            linewidths=1.5
        )

    ax.set_title(f"Diffraction clusters {sw} , {iv} (circle size = mean intensity)")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    plt.tight_layout()
    plt.show()


def plot_gmm_results(image, results):

    fig, ax = plt.subplots(figsize=(12, 10))

    # Log-normalized original image
    log_image = np.log1p(np.maximum(image, 0))

    ax.imshow(
        log_image,
        cmap="viridis",
        origin="upper"
    )

    for i, spot in enumerate(results["spots"]):

        cx = spot["center_x"]
        cy = spot["center_y"]

        # Fitted Gaussian ellipse (2 sigma)
        ellipse = Ellipse(
            xy=(cx, cy),
            width=4 * spot["sigma_major"],
            height=4 * spot["sigma_minor"],
            angle=spot["orientation_deg"],
            facecolor="red",
            edgecolor="white",
            alpha=0.30,
            linewidth=1.8,
            zorder=3
        )

        ax.add_patch(ellipse)

        # Center
        ax.scatter(
            cx, cy,
            color="white",
            marker="+",
            s=80,
            linewidths=2,
            zorder=4
        )

        # Intensity annotation
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
                    N=8,
                    threshold_percentile=97,
                    intensity_power=1.0
                )


                plot_gmm_results(data, results)

                # print(type(data))  # <class 'numpy.ndarray'>
                # print(data.shape)  # Image dimensions
                # print(data.ndim)
                # print(data.dtype)  # Data type
                #
                # plt.imshow(data, cmap="viridis", norm="log")
                # plt.colorbar(label="Intensity")
                # plt.show()
                #
                # print(img.header)

                break

                with archive.open(logfile, "r") as logfile_in_zip:
                    for line in logfile_in_zip.readlines():
                        print(line)
                        if not any([x in line for x in ["hexapod x", "hexapod z"]]):
                            continue
                        print(line)

                if i > 1:
                    break

                i += 1

                # Open the master accumulated file in write/append mode
             #   with open(accumulated_file_path, "w", encoding="utf-8") as output_file:



    print(f"All done! Data accumulated into {accumulated_file_path}")

if __name__ == "__main__":
    main()
