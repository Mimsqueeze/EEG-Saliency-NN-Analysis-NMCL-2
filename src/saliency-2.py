# Imports
import csv
import gc
import os
import time
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import models


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
The get_model() function

Returns a model from models
"""
def get_model():
    return models.get_mlp_2()
    # return models.get_cnn_2()

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
def plot_m(matrix, title, filename):
    plt.figure(figsize=(12, 12))
    cax = plt.matshow(matrix, vmin=0, vmax=1)
    ticks = [i for i in range(62)]
    plt.title(title, fontsize=8, pad=4)
    cbar = plt.colorbar(cax, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1], shrink=0.84, pad=0.01, aspect=30)
    cbar.ax.tick_params(labelsize=6, length=1, pad=1)
    plt.grid(True, color="white", linewidth=0.25, alpha=0.5)
    plt.xticks(ticks, LABELS, rotation=90, fontsize=4)
    plt.yticks(ticks, LABELS, fontsize=4)
    plt.tick_params(bottom=False, right=False, axis="both", length=1, pad=1)
    plt.savefig(filename, dpi=1000, bbox_inches="tight", format="svg")
    plt.close()

"""
The main() function

Reads features produced by feature_extraction.py, and performs 
classification on the EEG data. Saves the results into ./results
"""
def main():

    # Ensure tensorflow is utilizing the GPU
    if (len(tf.config.list_physical_devices("GPU")) == 0):
        quit()

    # Update console
    print(f"Reading in features...")

    combined_easy = None
    combined_diff = None

    # Read in features for all subjects from 1 to 29
    for n in range(1, 30):

        # Update console
        print(f"Reading sub-{n:02}...")

        # Read in easy level data
        easy = np.load(f"./features-coh-only/matrices/sub-{n:02}/easy.npy")
        combined_easy = easy if (combined_easy is None) else np.vstack(
            (easy, combined_easy))

        diff = np.load(f"./features-coh-only/matrices/sub-{n:02}/diff.npy")
        combined_diff = diff if (combined_diff is None) else np.vstack(
            (diff, combined_diff))

    # Update console
    print(f"Finished reading in features!")

    # For testing
    # image = easy[1, :, :, 1]
    # plot_m(matrix=image, title= "Coherence for Theta (4-7 Hz) MATB-Easy", filename= "test.svg")
    # quit()

    easy_labels = np.full((2088, 1), 0)
    diff_labels = np.full((2088, 1), 1)

    images = np.vstack((combined_easy, combined_diff))
    labels = np.vstack((easy_labels, diff_labels))

    # Number of folds for cross validation
    K = 10

    # Compute average test accuracy and loss
    avg_train_acc = 0
    avg_train_loss = 0
    avg_test_acc = 0
    avg_test_loss = 0
    avg_fit_time = 0

    # Perform K-Fold cross validation
    k_fold = KFold(n_splits=K, shuffle=True)

    # Build CNN
    model, MODEL_NAME = get_model()

    # Store the summary
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    SUMMARY = "\n".join(summary_list)
    print(SUMMARY)

    # Initialize tictionaries to store average smaps over k folds
    avg_easy_smap_dict = {}
    avg_diff_smap_dict = {}

    # Get number of images and channels
    NUM_IMAGES = easy.shape[0]
    NUM_CHANNELS = easy.shape[3]

    for channel in range(NUM_CHANNELS): # for every frequency range
        avg_easy_smap_dict[channel] = None
        avg_diff_smap_dict[channel] = None

    for train_indices, test_indices in k_fold.split(images):

        # Free memory
        gc.collect()

        # Split into training and testing sets
        train_images = images[train_indices]
        train_labels = labels[train_indices]
        test_images = images[test_indices]
        test_labels = labels[test_indices]

        valid_model = False

        while (not valid_model):
            # Get new model
            model, _ = get_model()

            callback = callbacks.EarlyStopping(
                monitor='val_accuracy', baseline=0.8, verbose=1, 
                patience=30, restore_best_weights=True)

            model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

            # Number of epochs
            NUM_EPOCHS = 200

            # Get training start time
            start_time = time.time()

            # Train the model
            model.fit(train_images, train_labels, epochs=NUM_EPOCHS, 
                    validation_data=(test_images, test_labels), 
                    callbacks=[callback])
            
            # Get training end time
            end_time = time.time()

            # Test the model and get metrics
            train_loss, train_acc = model.evaluate(train_images, train_labels, 
                                    verbose=2)
            test_loss, test_acc = model.evaluate(test_images, test_labels, 
                                                verbose=2)

            if (test_acc >= 0.9):
                # Update flag
                valid_model = True

                # Update metrics
                avg_train_acc = avg_train_acc + train_acc
                avg_train_loss = avg_train_loss + train_loss
                avg_test_acc = avg_test_acc + test_acc
                avg_test_loss = avg_test_loss + test_loss
                avg_fit_time = avg_fit_time + end_time - start_time

                # Update console
                print(f"Train accuracy: {train_acc}")
                print(f"Train loss: {train_loss}")
                print(f"Test accuracy: {test_acc}")
                print(f"Test loss: {test_loss}")
                print(f"Fit time: {end_time - start_time}")

                print(f"Adding saliency maps...")
                for channel in range(NUM_CHANNELS): # for every frequency range
                    for i in range(NUM_IMAGES): # for every image
                        easy_smap = get_saliency_map(model, tf.convert_to_tensor(easy[i:i+1, :, :, :]), channel)
                        diff_smap = get_saliency_map(model, tf.convert_to_tensor(diff[i:i+1, :, :, :]), channel)
                        avg_easy_smap_dict[channel] = easy_smap if (avg_easy_smap_dict[channel] is None) else avg_easy_smap_dict[channel] + easy_smap
                        avg_diff_smap_dict[channel] = diff_smap if (avg_diff_smap_dict[channel] is None) else avg_diff_smap_dict[channel] + diff_smap
            else:
                print(f"Model invalid, rerunning...")

    # Compute averages
    avg_train_acc = avg_train_acc/K
    avg_train_loss = avg_train_loss/K
    avg_test_acc = avg_test_acc/K
    avg_test_loss = avg_test_loss/K
    avg_fit_time = avg_fit_time/K

    # Update console
    print(f"Average train accuracy: {avg_train_acc}")
    print(f"Average train loss: {avg_train_loss}")
    print(f"Average test accuracy: {avg_test_acc}")
    print(f"Average test loss: {avg_test_loss}")
    print(f"Average Fit time: {avg_fit_time}")

    # Create a dataframe to store results to be saved into file
    results = pd.DataFrame()
    results["Classifier"] = [MODEL_NAME]
    results["Summary"] = [SUMMARY]
    results["Number of Epochs"] = [NUM_EPOCHS]
    results["Mean Fit Time"] = [avg_fit_time]
    results["Mean Train Accuracy"] = [avg_train_acc]
    results["Mean Train Loss"] = [avg_train_loss]
    results["Mean Test Accuracy"] = [avg_test_acc]
    results["Mean Test Loss"] = [avg_test_loss]
    
    # Transpose the dataframe to get metrics as rows
    results = results.T

    # Update console
    print("Results: ")
    print(results)

    # Create directory if it doesn't exist already
    base_dir = f"./plots/saliency maps/binary/{MODEL_NAME}/"
    if not os.path.exists(base_dir): os.makedirs(base_dir)
    
    # Save results
    print(f"Storing results in {base_dir}results.csv")
    if not os.path.exists(base_dir): os.makedirs(base_dir)
    results.to_csv(f"{base_dir}results.csv", header=False)
    
    # Update console
    print("Plotting saliency maps...")
    
    for channel in range(NUM_CHANNELS): # for every frequency range
        avg_easy_image = None
        avg_diff_image = None

        for i in range(NUM_IMAGES): # for every image
            easy_image = easy[i, :, :, channel]
            diff_image = diff[i, :, :, channel]

            avg_easy_image = easy_image if (avg_easy_image is None) else avg_easy_image + easy_image
            avg_diff_image = diff_image if (avg_diff_image is None) else avg_diff_image + diff_image

        # Compute averages
        avg_easy_image = avg_easy_image/NUM_IMAGES
        avg_diff_image = avg_diff_image/NUM_IMAGES

        avg_easy_smap_dict[channel] = avg_easy_smap_dict[channel]/(NUM_IMAGES * K)
        avg_diff_smap_dict[channel] = avg_diff_smap_dict[channel]/(NUM_IMAGES * K)

        # Create directory if it doesn't exist already
        dir = f"./plots/saliency maps/binary/{MODEL_NAME}/{RANGES[channel]}"
        if not os.path.exists(dir): os.makedirs(dir)
        
        # Save to file
        plot_m(matrix=avg_easy_image, title= f"Coherence for {RANGES[channel]} MATB-Easy", filename=f"{dir}/easy_image.svg")
        plot_m(matrix=avg_diff_image, title= f"Coherence for {RANGES[channel]} MATB-Difficult", filename=f"{dir}/diff_image.svg")

        plot_m(matrix=avg_easy_smap_dict[channel], title= f"Saliency Map for {RANGES[channel]} MATB-Easy", filename=f"{dir}/easy_smap.svg")
        plot_m(matrix=avg_diff_smap_dict[channel], title= f"Saliency Map for {RANGES[channel]} MATB-Difficult", filename=f"{dir}/diff_smap.svg")

        # Threshold
        threshold = 0.8

        # Get indices where values are greater than threshold
        easy_indices = np.where(np.triu(avg_easy_smap_dict[channel] > threshold))
        diff_indices = np.where(np.triu(avg_diff_smap_dict[channel] > threshold))

        # Convert indices to list of index tuples
        easy_list = list(zip(easy_indices[0], easy_indices[1]))
        diff_list = list(zip(diff_indices[0], diff_indices[1]))
        
        # Convert indices to list of string tuples
        easy_cells = list(map(lambda x: (f"{LABELS[x[0]]}-{LABELS[x[1]]}, S: {avg_easy_smap_dict[channel][x[0]][x[1]]}, E: {avg_easy_image[x[0]][x[1]]}, D: {avg_diff_image[x[0]][x[1]]}"), easy_list))
        diff_cells = list(map(lambda x: (f"{LABELS[x[0]]}-{LABELS[x[1]]}, S: {avg_diff_smap_dict[channel][x[0]][x[1]]}, E: {avg_easy_image[x[0]][x[1]]}, D: {avg_diff_image[x[0]][x[1]]}"), diff_list))

        # Save the important cells to file
        with open(f"{dir}/easy_cells.txt", "w") as f:
            for item in easy_cells:
                f.write(str(item) + '\n')
        
        with open(f"{dir}/diff_cells.txt", "w") as f:
            for item in diff_cells:
                f.write(str(item) + '\n')

def get_saliency_map(model, input_image, channel):
    # Compute the gradient of the model's output with respect to the input image
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        preds = model(input_image)
        predicted_class = tf.argmax(preds[0])
        loss = preds[:, predicted_class]

    grads = tape.gradient(loss, input_image)
    
    # Compute the saliency map
    saliency_map = tf.abs(grads)[0, :, :, channel].numpy()

    # Normalize the saliency map
    saliency_map /= np.max(saliency_map)

    return saliency_map

if __name__ == "__main__":
  main()