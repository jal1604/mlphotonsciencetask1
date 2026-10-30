
import zipfile
import fabio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import re
import os
import csv

from clustering import find_gmm_clusters, save_gmm_results
from plotting import plot_gmm_results, plot_intensity_distribution


def main():

    zip_file_path = "Mainz_2022-10_AXO-Optics-Scan-forML.zip"
    accumulated_file_path = "data3.csv"

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
                    N=4,
                    threshold_percentile=99,
                    intensity_power=1.0
                )

                #plot_gmm_results(data, results)
                #plot_intensity_distribution(data)

                hex_x = 0
                hex_z = 0


                with archive.open(logfile, "r") as logfile_in_zip:
                    for line in logfile_in_zip.readlines():
                        line = line.decode("utf-8").strip()
                        if "hexapod x" in line:
                            hex_x = float(line.split()[-1].replace(" ", ""))
                            continue
                        if "hexapod z" in line:
                            hex_z = float(line.split()[-1].replace(" ", ""))
                            continue

                # Accumulate results["features"] and the hex_x and hex_z into a single file
                save_gmm_results(
                    accumulated_file_path,
                    file_name.split("/")[-1],
                    hex_x,
                    hex_z,
                    results["features"]
                )

            # i += 1
            # if i > 5000:
            #     break



    print(f"All done! Data accumulated into {accumulated_file_path}")

if __name__ == "__main__":
    main()
