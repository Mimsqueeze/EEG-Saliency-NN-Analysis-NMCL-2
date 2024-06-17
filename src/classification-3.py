# Imports
import os
import numpy as np
import scipy.signal as ss
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from scipy.stats import shapiro
from scipy.stats import f_oneway
from scipy.stats import kruskal
from sklearn.decomposition import PCA

# ----------------------- author: Minsi Hu ---------------------------

# ----------------------- program description: -----------------------
#         Performs classification of varying levels of mental
#    workload on EEG features extracted from feature_extraction.py
# --------------------------------------------------------------------

# Hyperparameters
SIGNIFICANCE = 1e-15 # Significance level for feature selection
MODEL = "SVM"
HYPERPARAMETERS = "kernel='linear', C=1.0, probability=True"
SVM = SVC(kernel='linear', C=1.0, probability=True)
CLF = CalibratedClassifierCV(SVM)

"""
The get_pvalues(easy, med, diff) function

Computes p-values for differences between data for each feature

Parameters
    easy : numpy.ndarray
        features for easy, where each column is a feature and each row
        is a sample
    med : numpy.ndarray
        features for med, where each column is a feature and each row
        is a sample
    diff : numpy.ndarray
        features for diff, where each column is a feature and each row
        is a sample

Credit to: Arya Teymourlouei
"""
def get_pvalues(easy, med, diff):
    # we want to iterate through each feature, so transpose
    t_easy, t_med, t_diff = np.transpose(easy), np.transpose(med), np.transpose(diff)

    # Array to store p-values
    pvals = []

    # Iterate through every column
    for i in range(len(t_easy)):
        # Get ith column
        easy_d, med_d, diff_d = t_easy.iloc[i], t_med.iloc[i], t_diff.iloc[i]

        # complete test for normality (Shapiro-Wilk test)
        _, easy_p = shapiro(easy_d)
        _, med_p = shapiro(med_d)
        _, diff_p = shapiro(diff_d)

        # Normal distribution: Use ANOVA Test
        # Non-normal distribution: Use Kruskal-Wallis H-test
        if (easy_p < 0.05 and med_p < 0.05 and diff_p < 0.05):
            _, p_value = f_oneway(easy_d, med_d, diff_d)
        else:
            _, p_value = kruskal(easy_d, med_d, diff_d) 
            
        # append p-value and return scores
        pvals.append(p_value)
    
    # Return p-values
    return pvals

"""
The main() function

Reads features produced by feature_extraction.py, and performs 
classification on the EEG data. Saves the results into ./results
"""
def main():
    
    # Initialize dataframe to store features, where each row is a sample
    combined_easy = pd.DataFrame()
    combined_med = pd.DataFrame()
    combined_diff = pd.DataFrame()
  
    # Update console
    print(f"Reading in features...")

    # Read in features for all subjects from 1 to 29
    for n in range(1, 30):

        # Update console
        print(f"Reading sub-{n:02}...")

        # Read in easy level data
        easy = pd.read_csv(f"./features-coh-only/vectors/sub-{n:02}/easy.csv")
        easy.rename(columns={"Unnamed: 0": "Time"}, inplace=True)
        easy.set_index("Time", inplace=True)
        combined_easy = pd.concat([combined_easy, easy])

        # Read in med level data
        med = pd.read_csv(f"./features-coh-only/vectors/sub-{n:02}/med.csv")
        med.rename(columns={"Unnamed: 0": "Time"}, inplace=True)
        med.set_index("Time", inplace=True)
        combined_med = pd.concat([combined_med, med])

        # Read in diff level data
        diff = pd.read_csv(f"./features-coh-only/vectors/sub-{n:02}/diff.csv")
        diff.rename(columns={"Unnamed: 0": "Time"}, inplace=True)
        diff.set_index("Time", inplace=True)
        combined_diff = pd.concat([combined_diff, diff])
    
    # Update console
    print(f"Finished reading in features!")

    # Assign labels
    combined_easy["label"] = 0
    combined_med["label"] = 1
    combined_diff["label"] = 2

    # Combine the easy, med, and diff features into a single dataframe
    df = pd.concat([combined_easy, combined_med, combined_diff])

    # Shuffle the rows of the dataframe
    df = df.sample(frac = 1)

    # Compute p values between columns
    pvals = get_pvalues(combined_easy, combined_med, combined_diff)

    # Declare array of columns to drop
    drop_cols = []
    for i in range(len(pvals)):
        if (pvals[i] >= SIGNIFICANCE): drop_cols.append(str(i))

    # Drop the columns
    df = df.drop(drop_cols, axis=1)

    # Update console
    print(f"{len(drop_cols)} feature indices dropped from {len(pvals)} total.")

    # Get number of columns
    num_columns = df.shape[1]

    # Update console
    print(f"Final features to train with: {df}")

    metrics = {'f1' : make_scorer(f1_score, average='weighted'),
            'accuracy' : make_scorer(accuracy_score),
            'neg_log_loss' : 'neg_log_loss'}
  
    # Split data into features and labels
    X = df.drop(columns=["label"])
    y = df["label"]

    print(f"Training and testing...")

    # Compute cross validation scores
    scores = cross_validate(CLF, X, y, cv=10, scoring=metrics, 
                            return_estimator=True, return_train_score=True)

    print(f"Finished training and testing!")

    # Create a dataframe to store results to be saved into file
    results = pd.DataFrame()
    results["Classifier"] = [MODEL]
    results["Hyperparameters"] = [HYPERPARAMETERS]
    results["Significance Level"] = [SIGNIFICANCE]
    results["Number of Columns"] = [num_columns]
    results["Mean Fit Time"] = [np.mean(scores['fit_time'])]
    results["Mean Score Time"] = [np.mean(scores['score_time'])]
    results["Mean Train Accuracy"] = [np.mean(scores['train_accuracy'])]
    results["Mean Test Accuracy"] = [np.mean(scores['test_accuracy'])]
    results["Mean Train F1"] = [np.mean(scores['train_f1'])]
    results["Mean Test F1"] = [np.mean(scores['test_f1'])]
    results["Mean Train Neg Log Loss"] = [np.mean(scores['train_neg_log_loss'])]
    results["Mean Test Neg Log Loss"] = [np.mean(scores['test_neg_log_loss'])]
    
    # Transpose the dataframe to get metrics as rows
    results = results.T

    print("Results: ")
    print(results)

    print(f"Storing results in ./results/{MODEL}-coh-only.csv")

    # Save df to csv for classification
    dir = f"./results/ternary/"
    if not os.path.exists(dir): os.makedirs(dir)
    results.to_csv(f"{dir}/{MODEL}-{HYPERPARAMETERS}-{SIGNIFICANCE}-coh-only-3.csv", header=False)

if __name__ == "__main__":
  main()
