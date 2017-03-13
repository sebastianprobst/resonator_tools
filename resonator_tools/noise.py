
######
## Functions to evaluate noise data
######

import warnings
import numpy as np
from scipy.signal import periodogram


class noisedata(object):
    
    def __init__(self,IQ,IQref,fr,Ql,fs,gain_corr=[1.,1.],Z=50):
        '''
        units are assumed to be in volts
        -> IQ = I+1j*Q ; with amplitude signal on Q and phase on I
        this signal is measured on resonance
        -> IQref = Iref+1j*Qref ; with amplitude signal on Qref and phase on Iref
        this signal is measured far off resonance
        IMPORTANT: IQ and IQref describe signals on opposite sides of the resonance circle
        Therefore, take care that Q and Qref have the correct signs in order that 
        the program can determine the diameter of the resonance circle.
        -> fr: resonance frequency
        -> Ql: loaded Q of the resonator
        -> fs: sampling rate
        -> gain_corr = [1.,1.] ; enter here if the gain of IQ and IQref signals
        are different
        -> Z: impedance
        The signals will be normalized to the reference such that IQref = 1.        
        '''
        self.Z = Z
        self.fr = fr
        self.Ql = Ql
        self.offrespoint = np.mean(np.imag(IQref))
        self.respoint = np.mean(np.imag(IQref))
        self.radius = (self.offrespoint - self.respoint)/self.offrespoint
        self.P_I = periodogram(self._demean(np.real(IQ)),fs=fs)
        self.P_Q = periodogram(self._demean(np.imag(IQ)),fs=fs)
        self.P_Iref = periodogram(self._demean(np.real(IQref)),fs=fs)
        self.P_Qref = periodogram(self._demean(np.imag(IQref)),fs=fs)

################################# 
        #functions to evalate multiple things
    def P_I_eval_all(self):
        '''
        returns a 2D numpy array with all the results
        and a 1D list with the description
        '''
        comment = ['P_I','P_Inorm','P_Ipower','P_dtheta','P_dphi','P_df','P_']
        return np.vstack((self.P_I,self.P_Inorm(),self.P_Ipower(),self.P_dtheta(),self.P_dphi(),self.P_df(),self.P_())), comment
        
    def P_Iref_eval_all(self):
        '''
        returns a 2D numpy array with all the results
        and a 1D list with the description
        '''
        comment = ['P_Iref','P_Irefnorm','P_Irefpower','P_refdtheta','P_refdphi','P_refdf','P_ref']
        return np.vstack((self.P_Iref,self.P_Irefnorm(),self.P_Irefpower(),self.P_refdtheta(),self.P_refdphi(),self.P_refdf(),self.P_ref())), comment

    def P_Q_eval_all(self):
        '''
        returns a 2D numpy array with all the results
        and a 1D list with the description
        '''
        comment = ['P_Q','P_Qnorm','P_Qpower']
        return np.vstack((self.P_Q,self.P_Qnorm(),self.P_Qpower())), comment
        
    def P_Qref_eval_all(self):
        '''
        returns a 2D numpy array with all the results
        and a 1D list with the description
        '''
        comment = ['P_Qref','P_Qrefnorm','P_Qrefpower']
        return np.vstack((self.P_Qref,self.P_Qrefnorm(),self.P_Qrefpower())), comment

################################# 
        #helpers
    def _demean(self,x):
        '''
        removes the mean value from x
        '''
        return x - x.mean()
    
        
################################# 
        #noise on I
        
    def P_Inorm(self):
        '''
        V^2/Hz
        '''
        return self.P_I/(self.offrespoint**2)
        
    def P_Ipower(self):
        '''
        W/Hz
        '''
        return self.P_I/self.Z
        
    def P_dtheta(self):
        '''
        rad^2/Hz
        phase noise on the resonator circle phase
        (this is not the real measured phase)
        '''
        return self.P_Inorm()/self.r**2
        
    def P_dphi(self):
        '''
        rad^2/Hz
        phase noise on the phase measured with the VNA
        '''
        return self.P_Inorm()/np.absolute(self.respoint**2)
        
    def P_df(self):
        '''
        Hz^2/Hz
        frequency noise
        '''
        return self.P_theta() * self.fr**2 / (16.*self.Ql**2)
        
    def P_(self):
        '''
        1/Hz
        fractional frequency noise
        '''
        return self.P_theta() / (16.*self.Ql**2)
        
#################################        
        #noise on Iref
        
    def P_Irefnorm(self):
        '''
        V^2/Hz
        '''
        return self.P_Iref/(self.offrespoint**2)
        
    def P_Irefpower(self):
        '''
        W/Hz
        '''
        return self.P_Iref/self.Z
        
    def P_refdtheta(self):
        '''
        rad^2/Hz
        phase noise on the resonator circle phase
        (this is not the real measured phase)
        '''
        return self.P_Irefnorm()/self.r**2
        
    def P_refdphi(self):
        '''
        rad^2/Hz
        phase noise on the phase measured with the VNA
        '''
        return self.P_Irefnorm()/np.absolute(self.respoint**2)
        
    def P_refdf(self):
        '''
        Hz^2/Hz
        frequency noise
        '''
        return self.P_reftheta() * self.fr**2 / (16.*self.Ql**2)
        
    def P_ref(self):
        '''
        1/Hz
        fractional frequency noise
        '''
        return self.P_reftheta() / (16.*self.Ql**2)
        
#################################        
        #noise on Q
        
    def P_Qnorm(self):
        '''
        V^2/Hz
        '''
        return self.P_Q/(self.offrespoint**2)
        
    def P_Qpower(self):
        '''
        W/Hz
        '''
        return self.P_Q/self.Z
        
#################################        
        #noise on Qref
        
    def P_Qrefnorm(self):
        '''
        V^2/Hz
        '''
        return self.P_Qref/(self.offrespoint**2)
        
    def P_Qrefpower(self):
        '''
        W/Hz
        '''
        return self.P_Qref/self.Z
        
