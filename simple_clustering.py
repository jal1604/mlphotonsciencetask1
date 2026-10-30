import numpy as np
from sklearn.mixture import GaussianMixture


def find_gmm_clusters(image, N=32, threshold_percentile=99,
                      intensity_power=1.0, random_state=42):

    image = np.asarray(image, dtype=float)
    valid = np.isfinite(image)

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
    excess = np.maximum(intensities - threshold, 0)

    spots = []

    for k in range(N):
        weights = probabilities[:, k] * excess**intensity_power

        if weights.sum() <= 1e-12:
            weights = probabilities[:, k]

        center = np.average(X, axis=0, weights=weights)

        spots.append({
            "center_x": center[0],
            "center_y": center[1],
            "intensity": np.sum(probabilities[:, k] * excess)
        })

    return {"spots": spots, "gmm": gmm}
