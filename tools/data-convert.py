import mne.io
import pandas as pd
import sys
import os
# ----------------------- author: Minsi Hu ---------------------------

# ----------------------- program description: -----------------------
#               Preprocesses the data in ./MATB-data and
#          saves the data into ./data for feature extraction
# --------------------------------------------------------------------

# Reads in the data, applies a filter, re-references the data, converts the
# data into a pandas DataFrame, and saves the data as csv into ./data
def main():

    # Subject number of data to be processed
    n = 1

    # Get the subject number from command line arguments
    if (len(sys.argv) > 1):
        n = int(sys.argv[1])

    # Extract data from sessions 1, 2, 3 and levels easy, med, diff
    for s in [1, 2, 3]:
        for l in ["easy", "med", "diff"]:
            # Read in the data
            raw = mne.io.read_raw_eeglab(
                f"./MATB-data/sub-{n:02}/ses-S{s}/MATB{l}.set", 
                preload=True, eog=["ECG1"])

            # Convert data into numpy array
            data, _ = raw[:]

            # Convert numpy array to pandas dataframe
            df = pd.DataFrame(data).T

            # Rename the columns of the dataframe to sensor names
            df.columns = raw.ch_names

            # Drop the ECG1 column
            df = df.drop(["ECG1"], axis=1)
            
            # Convert values from V to mV
            df = df.apply(lambda x: x*1000000)

            # Save df to csv for feature extraction
            dir = f"./raw-data/sub-{n:02}/ses-S{s}/"
            if not os.path.exists(dir): os.makedirs(dir)
            df.to_csv(f"{dir}/{l}.csv")

if __name__ == '__main__':
    main()

