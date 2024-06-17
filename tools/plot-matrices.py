# Imports
import os
import matplotlib.pyplot as plt
import numpy as np


# ----------------------- author: Minsi Hu ---------------------------

# ----------------------- program description: -----------------------
#         Performs classification of varying levels of mental
#    workload on EEG features extracted from feature_extraction.py
# --------------------------------------------------------------------

RANGES = ["Theta (4-7 Hz)", "Low-Alpha (8-10 Hz)", "High-Alpha (11-13 Hz)", 
          "Low-Beta (14-22 Hz)", "High-Beta (23-35 Hz)", "Gamma (36-44 Hz)"]
LABELS = ["Fp1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3", "T7", "CP5",
          "CP1", "Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8", "TP10",
          "CP6", "CP2", "FCz", "C4", "T8", "FT10", "FC6", "FC2", "F4", "F8",
          "Fp2", "AF7", "AF3", "AFz", "F1", "F5", "FT7", "FC3", "C1", "C5",
          "TP7", "CP3", "P1", "P5", "PO7", "PO3", "POz", "PO4", "PO8", "P6",
          "P2", "CPz", "CP4", "TP8", "C6", "C2", "FC4", "FT8", "F6", "AF8",
          "AF4", "F2"]

"""
The plot_m() function
  Plots the matrix given in argument and saves it to file

Parameters
  matrix : numpy.ndarray
    a 62x62 adjacency matrix where each row/column corresponds to a sensor
  title : string
    title of the plot
  filename : string
    directory and file to be saved into
"""
def plot_m(matrix, filename):
    plt.figure(figsize=(12, 12))
    plt.matshow(matrix, vmin=0, vmax=1)
    ticks = [i for i in range(62)]
    plt.grid(True, color="white", linewidth=0.25, alpha=0.5)
    plt.xticks(ticks, [], rotation=90, fontsize=4)
    plt.yticks(ticks, [], fontsize=4)
    plt.tick_params(top=False, left=False, bottom=False, right=False, axis="both")
    plt.savefig(filename, dpi=1000, bbox_inches="tight", format="png")
    plt.close()

"""
The main() function

Reads features produced by feature_extraction.py, and performs 
classification on the EEG data. Saves the results into ./results
"""
def main():

    # Update console
    print(f"Reading in features...")

    combined_easy = None
    combined_med = None
    combined_diff = None

    # Read in features for all subjects from 1 to 29
    for n in range(1, 30):

        # Update console
        print(f"Reading sub-{n:02}...")

        # Read in easy level data
        easy = np.load(f"./features-coh-only/matrices/sub-{n:02}/easy.npy")
        combined_easy = easy if (combined_easy is None) else np.vstack(
            (easy, combined_easy))

        med = np.load(f"./features-coh-only/matrices/sub-{n:02}/med.npy")
        combined_med = med if (combined_med is None) else np.vstack(
            (med, combined_med))

        diff = np.load(f"./features-coh-only/matrices/sub-{n:02}/diff.npy")
        combined_diff = diff if (combined_diff is None) else np.vstack(
            (diff, combined_diff))

    # Update console
    print(f"Finished reading in features!")

    # Get number of images and channels
    NUM_IMAGES = easy.shape[0]
    NUM_CHANNELS = easy.shape[3]
    
    # Update console
    print("Plotting matrices...")
    
    for channel in range(NUM_CHANNELS): # for every frequency range
        avg_easy_image = None
        avg_med_image = None
        avg_diff_image = None

        for i in range(NUM_IMAGES): # for every image
            easy_image = easy[i, :, :, channel]
            med_image = med[i, :, :, channel]
            diff_image = diff[i, :, :, channel]

            avg_easy_image = easy_image if (avg_easy_image is None) else avg_easy_image + easy_image
            avg_med_image = med_image if (avg_med_image is None) else avg_med_image + med_image
            avg_diff_image = diff_image if (avg_diff_image is None) else avg_diff_image + diff_image
        
        # Compute averages
        avg_easy_image = avg_easy_image/NUM_IMAGES
        avg_med_image = avg_med_image/NUM_IMAGES
        avg_diff_image = avg_diff_image/NUM_IMAGES

        # Create directory if it doesn't exist already
        dir = f"./matrices/{RANGES[channel]}"
        if not os.path.exists(dir): os.makedirs(dir)
        
        # Save to file
        plot_m(matrix=avg_easy_image, filename=f"{dir}/easy_image.png")
        plot_m(matrix=avg_med_image, filename=f"{dir}/med_image.png")
        plot_m(matrix=avg_diff_image, filename=f"{dir}/diff_image.png")

if __name__ == "__main__":
  main()