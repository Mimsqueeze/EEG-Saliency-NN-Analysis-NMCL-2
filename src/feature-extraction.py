import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as ss
import warnings

# ----------------------------- Minsi Hu -----------------------------

# ----------------------- program description: -----------------------
#               Reads in EEG data and extracts features to 
#                      be used in classification
# --------------------------------------------------------------------

NUM_SENSORS = 62
SAMPLING_FREQUENCY = 500
RANGES = {"theta":(4,7), "low-alpha":(8,10), "high-alpha":(11,13), 
          "low-beta":(14,22), "high-beta":(23,35), "gamma":(36,44)}
PLOT = False
PRINT_DF = False
PRINT_MAT_DIM = True

"""
The plot_m() function
    Plots the matrix given in argument and saves it to file

Parameters
    matrix : numpy.ndarray
        matrix
    title : string
        title of the plot
    filename : string
        filename of the plot
"""
def plot_m(matrix, title, filename):
  cax = plt.matshow(matrix)
  plt.title(title)
  plt.colorbar(cax)
  plt.tick_params(axis='both', which='both', labelsize=6)
  plt.savefig(f"./matrices/{filename}.png")
  plt.clf()


"""
The main() function
    Reads in the cleaned data from preprocessing.py, extracts pli, wpli, dpi,
    coh, and icoh features, and saves them into ./features directory to be 
    used in classification.py

Command Line Arguments
    argv[1] - n : Subject number of data to be processed
    argv[2] - l : Level of data to be processed
"""
def main():

    # Ignore warnings
    warnings.filterwarnings('ignore')

    # Subject number of data to be processed
    n = 1

    # Get the subject number from command line arguments
    if (len(sys.argv) > 1): 
      n = int(sys.argv[1])

    # Level of the data to be processed
    l = "easy"

    # Get the level from command line arguments
    if (len(sys.argv) > 2): 
      l = sys.argv[2]

    # Print to console
    print(f"Processing sub-{n:02}/{l}")

    # Declare dataframe and matrix to store results of feature creation
    df_features = pd.DataFrame()
    mat = None
    
    # Loop through samples
    for s in [1, 2, 3]:

        # Read in the data
        df = pd.read_csv(f"./data/sub-{n:02}/ses-S{s}/{l}.csv")

        # Drop the extra index column and convert to numpy array
        df = df.drop("Unnamed: 0", axis=1).to_numpy()

        # Compute features for every 10 second interval and for every
        # frequency range 
        for t in range(0, 120000, 5000):

            # Vector storing combined result of all feature extraction 
            # methods and all frequency ranges for a 10 second interval
            v, m = None, None

            # Take a 10 second slice of the data
            data = df[t:t+5000,:]
            
            # Dictionary of precompute csd's for every pair of sensors
            csd = {}

            # Precompute the csd for every pair of sensors
            for i in range(NUM_SENSORS):
                for k in range(i+1, NUM_SENSORS):

                    # Use scipy csd function to estimate csd
                    csd[(i, k, "Pxy")] = ss.csd(data[:,i], data[:,k], 
                                                fs = SAMPLING_FREQUENCY)
                    csd[(i, k, "Pxx")] = ss.csd(data[:,i], data[:,i], 
                                                fs = SAMPLING_FREQUENCY)
                    csd[(i, k, "Pyy")] = ss.csd(data[:,k], data[:,k], 
                                                fs = SAMPLING_FREQUENCY)

            # Compute features for every frequency range
            for r in RANGES:

                # Get the min and max frequencies for a range
                min, max = RANGES[r]

                # Declare feature matrices
                # pli_m = np.zeros([NUM_SENSORS, NUM_SENSORS], dtype=float)
                # wpli_m = np.zeros([NUM_SENSORS, NUM_SENSORS], dtype=float)
                # dpli_m = np.zeros([NUM_SENSORS, NUM_SENSORS], dtype=float)
                coh_m = np.zeros([NUM_SENSORS, NUM_SENSORS], dtype=float)
                # icoh_m = np.zeros([NUM_SENSORS, NUM_SENSORS], dtype=float)

                # Fill in the matrices for every pair of sensors
                for i in range(NUM_SENSORS):
                    for k in range(i+1, NUM_SENSORS):

                        # Get precomputed csd
                        fxy, Pxy = csd[i, k, "Pxy"]
                        fxx, Pxx = csd[i, k, "Pxx"]
                        fyy, Pyy = csd[i, k, "Pyy"]

                        # Restrict csd based on frequency range
                        fPxy = Pxy[np.where((fxy >= min) & (fxy <= max))]
                        fPxx = Pxx[np.where((fxx >= min) & (fxx <= max))]
                        fPyy = Pyy[np.where((fyy >= min) & (fyy <= max))]
                        
                        # Get imaginary component of csd
                        iPxy = np.imag(fPxy)

                        # Compute coh and icoh numerators and denominators
                        num = fPxy*fPxy
                        den = fPxx*fPyy

                        # Fill feature matrices
                        # pli_m[i,k] = np.abs(np.mean(np.sign(iPxy)))
                        # wpli_m[i,k] = np.abs(np.mean(iPxy))/np.mean(
                        #     np.abs(iPxy))
                        # dpli_m[i,k] = np.mean(np.heaviside(iPxy, 0.5))
                        coh_m[i,k] =  np.mean(num/den)
                        # icoh_m[i,k] = np.mean(np.imag(num)/den)

                        # Fill in the other half of the matrices
                        # pli_m[k,i] = pli_m[i,k]
                        # wpli_m[k,i] = wpli_m[i,k]
                        # dpli_m[k,i] = dpli_m[i,k]
                        coh_m[k,i] = coh_m[i,k]
                        # icoh_m[k,i] = icoh_m[i,k]

                if PLOT:
                    # plot_m(pli_m, f"pli_{r}", f"pli_{r}")
                    # plot_m(wpli_m, f"wpli_{r}", f"wpli_{r}")
                    # plot_m(dpli_m, f"dpli_{r}", f"dpli_{r}")
                    plot_m(coh_m, f"coh_{r}", f"coh_{r}")
                    # plot_m(icoh_m, f"icoh_{r}", f"icoh_{r}")

                # Compute vectors from matrices to reduce dimensionality
                # pli_v = pli_m[np.triu_indices(pli_m.shape[0], k=1)]
                # wpli_v = wpli_m[np.triu_indices(wpli_m.shape[0], k=1)]
                # dpli_v = dpli_m[np.triu_indices(dpli_m.shape[0], k=1)]
                coh_v = coh_m[np.triu_indices(coh_m.shape[0], k=1)]
                # icoh_v = icoh_m[np.triu_indices(icoh_m.shape[0], k=1)]

                # Combine vectors
                # c = np.concatenate((pli_v, wpli_v, dpli_v, coh_v, icoh_v))
                c = coh_v
                v = c if (v is None) else np.concatenate((v, c))

                # Combine matrices
                m_temp = np.expand_dims(coh_m, axis=0)
                m = m_temp if (m is None) else np.vstack((m, m_temp))
    
            # Assign combined vector to dataframe
            df_features[f"N:{n}, S:{s}: {10*t/5000}-{10 + 10*t/5000}"] = v

            # Transpose the matrix to change dimensions to 64x64x6
            m_t = np.transpose(m, (1, 2, 0))

            # Combine matrices
            mat_temp = np.expand_dims(m_t, axis=0)
            mat = mat_temp if (mat is None) else np.vstack((mat, mat_temp))

    # Print dataframe
    if PRINT_DF:
        print(df_features)
    
    # Print matrix dimensions
    if PRINT_MAT_DIM:
        print(mat.shape)

    # Take the transpose to make classification easier
    df_features = df_features.T

    # Update console
    print(f"Saving features for sub-{n:02}/{l}")

    # Save df to csv for classification
    dir = f"./features-coh-only/vectors/sub-{n:02}/"
    if not os.path.exists(dir): os.makedirs(dir)
    df_features.to_csv(f"{dir}/{l}.csv")

    # Save matrix to npy for classification
    dir = f"./features-coh-only/matrices/sub-{n:02}/"
    if not os.path.exists(dir): os.makedirs(dir)
    np.save(f"{dir}/{l}.npy", mat)

if __name__ == '__main__':
    main()