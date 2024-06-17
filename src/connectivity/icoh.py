import numpy as np
import scipy.signal as ss

"""
The icoh(num_sensors, data, f_min, f_max, sfreq) function
  Computes imaginary coherence matrix

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
  icoh_m : numpy.ndarray
    The imaginary coherence matrix for data, where each row and column is a 
    sensor and each entry is a float of the imaginary coherence between 
    two sensors
  icoh_v : numpy.ndarray
    The imaginary coherence vector for data
"""
def icoh(data, num_sensors, f_min, f_max, sfreq):
  # Declare empty imaginary coherence matrix
  icoh_m = np.zeros([num_sensors, num_sensors], dtype=float)
  
  # Fill imaginary coherence matrix by computing imaginary coherence for every 
  # pair of sensors
  for i in range(num_sensors):
    for k in range(num_sensors):
      # Use scipy csd function to estimate cross/auto spectral densities
      fxy, Pxy = ss.csd(data[:,i], data[:,k], fs = sfreq)
      fxx, Pxx = ss.csd(data[:,i], data[:,i], fs = sfreq)
      fyy, Pyy = ss.csd(data[:,k], data[:,k], fs = sfreq)
      
      # Restrict frequency ranges
      fPxy = Pxy[np.where((fxy >= f_min) & (fxy <= f_max))]
      fPxx = Pxx[np.where((fxx >= f_min) & (fxx <= f_max))]
      fPyy = Pyy[np.where((fyy >= f_min) & (fyy <= f_max))]

      # Compute numerator of coh
      icoh_num = np.imag(fPxy*fPxy)

      # Compute denominator of coh
      icoh_den = fPxx*fPyy

      # Compute coherence
      icoh_m[i,k] = np.mean(icoh_num/icoh_den)
  
  # Computing imaginary coherence vector to reduce dimensionality
  icoh_v = icoh_m[np.triu_indices(icoh_m.shape[0], k=1)]

  # Return the imaginary coherence matrix and vector
  return icoh_m, icoh_v
