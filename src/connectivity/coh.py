import numpy as np
import scipy.signal as ss

"""
The coh(num_sensors, data, f_min, f_max, sfreq) function
  Computes coherence matrix

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
  coh_m : numpy.ndarray
    The coherence matrix for data, where each row and column is a sensor and 
    each entry is a float of the coherence between two sensors
  coh_v : numpy.ndarray
    The coherence vector for data
"""
def coh(data, num_sensors, f_min, f_max, sfreq):
  # Declare empty coherence matrix
  coh_m = np.zeros([num_sensors, num_sensors], dtype=float)
  
  # Fill coherence matrix by computing coherence for every pair of sensors
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
      coh_num = fPxy*fPxy

      # Compute denominator of coh
      coh_den = fPxx*fPyy

      # Compute coherence
      coh_m[i,k] = np.mean(coh_num/coh_den)
  
  # Computing coherence vector to reduce dimensionality
  coh_v = coh_m[np.triu_indices(coh_m.shape[0], k=1)]

  # Return the coherence matrix and vector
  return coh_m, coh_v
