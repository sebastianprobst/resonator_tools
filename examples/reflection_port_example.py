
from resonator_tools import circuit


port1 = circuit.reflection_port()
port1.add_fromtxt('S11.txt','dBmagphasedeg',1)
port1.autofit()
print("Fit results:", port1.fitresults)
port1.plotall()
print("single photon limit:", port1.get_single_photon_limit(), "dBm")
print("done")
