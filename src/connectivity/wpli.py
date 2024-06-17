import numpy as np
import scipy.signal as ss

"""
The wpli(data, num_sensors, f_min, f_max, sfreq) function
  Computes wpli matrix

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
  wpli_m : numpy.ndarray
    The wpli matrix for data, where each row and column is a sensor and each 
    entry is a float of the wpli between two sensors
  wpli_v : numpy.ndarray
    The wpli vector for data
"""
def wpli(data, num_sensors, f_min, f_max, sfreq):
  # Declare empty wpli matrix
  wpli_m = np.zeros([num_sensors, num_sensors], dtype=float)
  
  # Fill wpli matrix by computing wpli for every pair of sensors
  for i in range(num_sensors):
    for k in range(num_sensors):
      # Use scipy csd function to estimate cross spectral density
      f, Pxy = ss.csd(data[:,i], data[:,k], fs = sfreq)

      # Restrict frequency ranges
      fPxy = Pxy[np.where((f >= f_min) & (f <= f_max))]

      # Get imaginary component of csd
      iPxy = np.imag(fPxy)

      # Compute numerator of wpli
      wpli_num = np.abs(np.mean(iPxy))

      # Compute denominator of wpli
      wpli_den = np.mean(np.abs(iPxy))

      # Compute wpli
      wpli_m[i,k] = wpli_num/wpli_den

  # Computing wpli vector to reduce dimensionality
  wpli_v = wpli_m[np.triu_indices(wpli_m.shape[0], k=1)]

  # Return the wpli matrix and vector
  return wpli_m, wpli_v