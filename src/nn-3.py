# Imports
import gc
import os
import time
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import models


# ----------------------- author: Minsi Hu ---------------------------

# ----------------------- program description: -----------------------
#         Performs classification of varying levels of mental
#    workload on EEG features extracted from feature_extraction.py
# --------------------------------------------------------------------

"""
The get_model() function

Returns a model from models
"""
def get_model():
    # return models.get_mlp_3()
    return models.get_cnn_3()

"""
The main() function

Reads features produced by feature_extraction.py, and performs 
classification on the EEG data. Saves the results into ./results
"""
def main():

    # Ensure tensorflow is utilizing the GPU
    if (len(tf.config.list_physical_devices('GPU')) == 0):
        quit()

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
        combined_diff= diff if (combined_diff is None) else np.vstack(
            (diff, combined_diff))

    # Update console
    print(f"Finished reading in features!")

    easy_labels = np.full((2088, 1), 0)
    med_labels = np.full((2088, 1), 1)
    diff_labels = np.full((2088, 1), 2)

    images = np.vstack((combined_easy, combined_med, combined_diff))
    labels = np.vstack((easy_labels, med_labels, diff_labels))

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

    # Save df to csv for classification
    dir = f"./results/ternary"
    print(f"Storing results in {dir}/{MODEL_NAME}-coh-only.csv")
    if not os.path.exists(dir): os.makedirs(dir)
    results.to_csv(f"{dir}/{MODEL_NAME}-coh-only.csv", header=False)

if __name__ == "__main__":
  main()
