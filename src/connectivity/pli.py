import numpy as np
import scipy.signal as ss

"""
The pli(data, num_sensors, f_min, f_max, sfreq) function
  Computes pli matrix

Parameters
  data : numpy.ndarray
    the EEG data in a numpy.ndarray, where each column is a sensor, each row
    is a sample, and each entry is a float of the mV
  num_sensors : int
    number of sensors used for EEG
  f_min : float
    min frequency in hz
  f_max : float
    max frequency in hz
  sfreq : float
    sampling frequency of the data in hz

Returns
  pli_m : numpy.ndarray
    The pli matrix for data, where each row and column is a sensor and each 
    entry is a float of the pli between two sensors
  pli_v : numpy.ndarray
    The pli vector for data
"""
def pli(data, num_sensors, f_min, f_max, sfreq):
  # Declare empty pli matrix
  pli_m = np.zeros([num_sensors, num_sensors], dtype=float)
  
  # Fill pli matrix by computing pli for every pair of sensors
  for i in range(num_sensors):
    for k in range(num_sensors):
      # Use scipy csd function to estimate cross spectral density
      f, Pxy = ss.csd(data[:,i], data[:,k], fs = sfreq)

      # Restrict frequency ranges
      fPxy = Pxy[np.where((f >= f_min) & (f <= f_max))]

      # Get imaginary component of csd
      iPxy = np.imag(fPxy)

      # Compute pli
      pli_m[i,k] = np.abs(np.mean(np.sign(iPxy)))

  # Computing pli vector to reduce dimensionality
  pli_v = pli_m[np.triu_indices(pli_m.shape[0], k=1)]

  # Return the pli matrix and vector
  return pli_m, pli_v