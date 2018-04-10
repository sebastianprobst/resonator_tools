import numpy as np

from resonator_tools import circuit

#---------------------------------------
#generate test data
fr = 7e9  #resonance frequency in Hz
Qi = 200e3
Qc = 300e3
freq = np.linspace(fr-5e6, fr+5e6, 5000)
port1 = circuit.reflection_port()  #define a reflection port
#add noise
noise = np.random.normal(loc=1.0,scale=0.01,size=(len(freq),))
S11 = noise * port1._S11_directrefl(freq,fr=fr,Ql=Qi*Qc/(Qc+Qi),Qc=Qc,a=1.,alpha=0.,delay=.0)
# add wiggly baseline
baseline = 1.+0.1*np.cos(2.*np.pi*0.001e-4*freq)+0.05*np.sin(2.*np.pi*0.0069e-4*freq)+0.001*np.sin(2.*np.pi*0.0001e-4*freq)
S11b = S11*baseline
#-----------------------------------------

# create fitting object
port1 = circuit.reflection_port() 
port1.add_data(freq,S11b)

# fit and remove base line
port1.GUIbaselinefit()

# fit the corrected data
port1.GUIfit()
print("Fit results:", port1.fitresults)
port1.plotall()
print("single photon limit:", port1.get_single_photon_limit(), "dBm")
print("done")

