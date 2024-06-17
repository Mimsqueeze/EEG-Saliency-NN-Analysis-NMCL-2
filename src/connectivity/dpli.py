import numpy as np
import scipy.signal as ss

"""
The dpli(data, num_sensors, f_min, f_max, sfreq) function
  Computes dpli matrix

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
  dpli_m : numpy.ndarray
    The dpli matrix for data, where each row and column is a sensor and each 
    entry is a float of the dpli between two sensors
  dpli_v : numpy.ndarray
    The dpli vector for data
"""
def dpli(data, num_sensors, f_min, f_max, sfreq):
  # Declare empty dpli matrix
  dpli_m = np.zeros([num_sensors, num_sensors], dtype=float)
  
  # Fill dpli matrix by computing dpli for every pair of sensors
  for i in range(num_sensors):
    for k in range(num_sensors):
      # Use scipy csd function to estimate cross spectral density
      f, Pxy = ss.csd(data[:,i], data[:,k], fs = sfreq)

      # Restrict frequency ranges
      fPxy = Pxy[np.where((f >= f_min) & (f <= f_max))]

      # Get imaginary component of csd
      iPxy = np.imag(fPxy)

      # Compute dpli using heaviside 0.5
      dpli_m[i,k] = np.mean(np.heaviside(iPxy, 0.5))

  # Computing dpli vector to reduce dimensionality
  dpli_v = dpli_m[np.triu_indices(dpli_m.shape[0], k=1)]

  # Return the dpli matrix and vector
  return dpli_m, dpli_v