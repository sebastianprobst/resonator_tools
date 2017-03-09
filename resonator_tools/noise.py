
######
## Functions to evaluate noise data
######

import warnings
import numpy as np
from scipy.signal import periodogram




def PSD(X,fs,Z=50.):
    '''
    Calculates the power spectral density of a quadrature.
    If X is in units of V and Z=50 Ohm and fs in Hz,
    then the output is in W/Hz
    '''
    Xs = np.mean(X) #signal
    dX = X-Xs #noise
    f, P_xx = periodogram(dX,fs)
    P_xx /= float(Z)
    return f, P_xx
    
def PSD_dBc(X,fs):
    '''
    Calculates the power spectral density of a quadrature.
    the output is in dBc/Hz
    '''
    Xs = np.mean(X) #signal
    dX = X-Xs #noise
    f, P_xx = periodogram(dX,fs)
    return f, 10.*np.log10(P_xx/Xs**2)
    
def PSD_rel(X,fs):
    '''
    Calculates the power spectral density of a quadrature.
    the output is in 1/Hz
    '''
    Xs = np.mean(X) #signal
    dX = X-Xs #noise
    f, P_xx = periodogram(dX,fs)
    return f, P_xx/Xs**2
    
def freq_noise(X,fs,Q):
    '''
    returns the frequency noise in units
    of 1/Hz
    '''
    f, P = PSD_rel(X,fs)
    return f, P/(4.*Q)
