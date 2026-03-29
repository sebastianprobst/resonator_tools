from _example_setup import data_path, ensure_repo_root_on_path

ensure_repo_root_on_path()

from resonator_tools import circuit  # noqa: E402


port1 = circuit.reflection_port()
port1.add_fromtxt(str(data_path("S11.txt")), "dBmagphasedeg", 1)
port1.autofit()
print("Fit results:", port1.fitresults)
port1.plotall()
print("single photon limit:", port1.get_single_photon_limit(), "dBm")
print("done")
