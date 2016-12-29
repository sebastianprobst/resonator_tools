
from resonator_tools import circuit


port1 = circuit.notch_port()
port1.add_froms2p('S21testdata.s2p',3,4,'realimag',fdata_unit=1e9,delimiter=None)
port1.autofit()
print "Fit results:", port1.fitresults
port1.plotall()
print "single photon limit:", port1.get_single_photon_limit(diacorr=True), "dBm"
print "photons in reso for input -140dBm:", port1.get_photons_in_resonator(-140,unit='dBm',diacorr=True), "photons"
print "done"
