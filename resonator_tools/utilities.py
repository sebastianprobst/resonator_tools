import warnings
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]


def Watt2dBm(x: float | FloatArray) -> float | FloatArray:
    """
    converts from units of watts to dBm
    """
    return 10.0 * np.log10(x * 1000.0)


def dBm2Watt(x: float | FloatArray) -> float | FloatArray:
    """
    converts from units of watts to dBm
    """
    return 10 ** (x / 10.0) / 1000.0


class plotting(object):
    """
    some helper functions for plotting
    """

    # TODO: refactor architecture using composition instead of inheritance, so that plotting is a separate class that can be used by any port type without needing to inherit from it
    def plotall(self) -> None:
        # FIXME: variable assignments depend on the presence of raw and sim data via inheritance, which may not always be the case. This should be refactored to be more robust and flexible.
        real = self.z_data_raw.real  # type: ignore
        imag = self.z_data_raw.imag  # type: ignore
        real2 = self.z_data_sim.real  # type: ignore
        imag2 = self.z_data_sim.imag  # type: ignore
        plt.subplot(221)
        plt.plot(real, imag, label="rawdata")
        plt.plot(real2, imag2, label="fit")
        plt.xlabel("Re(S21)")
        plt.ylabel("Im(S21)")
        plt.legend()
        plt.subplot(222)
        plt.plot(self.f_data * 1e-9, np.absolute(self.z_data_raw), label="rawdata")  # type: ignore
        plt.plot(self.f_data * 1e-9, np.absolute(self.z_data_sim), label="fit")  # type: ignore
        plt.xlabel("f (GHz)")
        plt.ylabel("|S21|")
        plt.legend()
        plt.subplot(223)
        plt.plot(self.f_data * 1e-9, np.angle(self.z_data_raw), label="rawdata")  # type: ignore
        plt.plot(self.f_data * 1e-9, np.angle(self.z_data_sim), label="fit")  # type: ignore
        plt.xlabel("f (GHz)")
        plt.ylabel("arg(|S21|)")
        plt.legend()
        plt.show()

    def plotcalibrateddata(self) -> None:
        real = self.z_data.real  # type: ignore
        imag = self.z_data.imag  # type: ignore
        plt.subplot(221)
        plt.plot(real, imag, label="rawdata")
        plt.xlabel("Re(S21)")
        plt.ylabel("Im(S21)")
        plt.legend()
        plt.subplot(222)
        plt.plot(self.f_data * 1e-9, np.absolute(self.z_data), label="rawdata")  # type: ignore
        plt.xlabel("f (GHz)")
        plt.ylabel("|S21|")
        plt.legend()
        plt.subplot(223)
        plt.plot(self.f_data * 1e-9, np.angle(self.z_data), label="rawdata")  # type: ignore
        plt.xlabel("f (GHz)")
        plt.ylabel("arg(|S21|)")
        plt.legend()
        plt.show()

    def plotrawdata(self) -> None:
        real = self.z_data_raw.real  # type: ignore
        imag = self.z_data_raw.imag  # type: ignore
        plt.subplot(221)
        plt.plot(real, imag, label="rawdata")
        plt.xlabel("Re(S21)")
        plt.ylabel("Im(S21)")
        plt.legend()
        plt.subplot(222)
        plt.plot(self.f_data * 1e-9, np.absolute(self.z_data_raw), label="rawdata")  # type: ignore
        plt.xlabel("f (GHz)")
        plt.ylabel("|S21|")
        plt.legend()
        plt.subplot(223)
        plt.plot(self.f_data * 1e-9, np.angle(self.z_data_raw), label="rawdata")  # type: ignore
        plt.xlabel("f (GHz)")
        plt.ylabel("arg(|S21|)")
        plt.legend()
        plt.show()


class save_load(object):
    """
    procedures for loading and saving data used by other classes
    """

    def _ConvToCompl(
        self,
        x: FloatArray,
        y: FloatArray,
        dtype: str,
    ) -> ComplexArray | None:
        """
        dtype = 'realimag', 'dBmagphaserad', 'linmagphaserad', 'dBmagphasedeg', 'linmagphasedeg'
        """
        if dtype == "realimag":
            return x + 1j * y
        elif dtype == "linmagphaserad":
            return (x * np.exp(1j * y)).astype(np.complex128)
        elif dtype == "dBmagphaserad":
            return (10 ** (x / 20.0) * np.exp(1j * y)).astype(np.complex128)
        elif dtype == "linmagphasedeg":
            return (x * np.exp(1j * y / 180.0 * np.pi)).astype(np.complex128)
        elif dtype == "dBmagphasedeg":
            return (10 ** (x / 20.0) * np.exp(1j * y / 180.0 * np.pi)).astype(
                np.complex128
            )
        else:
            warnings.warn(
                "Undefined input type! Use 'realimag', 'dBmagphaserad', 'linmagphaserad', 'dBmagphasedeg' or 'linmagphasedeg'.",
                SyntaxWarning,
            )

    def add_data(self, f_data: FloatArray, z_data: ComplexArray) -> None:
        self.f_data = np.array(f_data)
        self.z_data_raw = np.array(z_data)

    def cut_data(self, f1: float, f2: float) -> None:
        def findpos(f_data: FloatArray, val: float) -> int:
            pos = 0
            for i in range(len(f_data)):
                if f_data[i] < val:
                    pos = i
            return pos

        pos1 = findpos(self.f_data, f1)
        pos2 = findpos(self.f_data, f2)
        self.f_data = self.f_data[pos1:pos2]
        self.z_data_raw = self.z_data_raw[pos1:pos2]  # type: ignore

    def add_fromtxt(
        self,
        fname: str,
        dtype: str,
        header_rows: int,
        usecols: tuple[int, int, int] = (0, 1, 2),
        fdata_unit: float = 1.0,
        delimiter: str | None = None,
    ) -> None:
        """
        dtype = 'realimag', 'dBmagphaserad', 'linmagphaserad', 'dBmagphasedeg', 'linmagphasedeg'
        """
        data = np.loadtxt(
            fname, usecols=usecols, skiprows=header_rows, delimiter=delimiter
        )
        self.f_data = data[:, 0] * fdata_unit
        self.z_data_raw = self._ConvToCompl(data[:, 1], data[:, 2], dtype=dtype)

    def add_fromhdf(self) -> None:
        pass

    def add_froms2p(
        self,
        fname: str,
        y1_col: int,
        y2_col: int,
        dtype: str,
        fdata_unit: float = 1.0,
        delimiter: str | None = None,
    ) -> None:
        """
        dtype = 'realimag', 'dBmagphaserad', 'linmagphaserad', 'dBmagphasedeg', 'linmagphasedeg'
        """
        if dtype == "dBmagphasedeg" or dtype == "linmagphasedeg":
            phase_conversion = 1.0 / 180.0 * np.pi
        else:
            phase_conversion = 1.0
        with open(fname) as f:
            lines = f.readlines()
        z_data_raw = []
        f_data = []
        if dtype == "realimag":
            for line in lines:
                if (line != "\n") and (line[0] != "#") and (line[0] != "!"):
                    lineinfo = line.split(delimiter)
                    f_data.append(float(lineinfo[0]) * fdata_unit)
                    z_data_raw.append(
                        complex(float(lineinfo[y1_col]), float(lineinfo[y2_col]))
                    )
        elif dtype == "linmagphaserad" or dtype == "linmagphasedeg":
            for line in lines:
                if (
                    (line != "\n")
                    and (line[0] != "#")
                    and (line[0] != "!")
                    and (line[0] != "M")
                    and (line[0] != "P")
                ):
                    lineinfo = line.split(delimiter)
                    f_data.append(float(lineinfo[0]) * fdata_unit)
                    z_data_raw.append(
                        float(lineinfo[y1_col])
                        * np.exp(
                            complex(0.0, phase_conversion * float(lineinfo[y2_col]))
                        )
                    )
        elif dtype == "dBmagphaserad" or dtype == "dBmagphasedeg":
            for line in lines:
                if (
                    (line != "\n")
                    and (line[0] != "#")
                    and (line[0] != "!")
                    and (line[0] != "M")
                    and (line[0] != "P")
                ):
                    lineinfo = line.split(delimiter)
                    f_data.append(float(lineinfo[0]) * fdata_unit)
                    linamp = 10 ** (float(lineinfo[y1_col]) / 20.0)
                    z_data_raw.append(
                        linamp
                        * np.exp(
                            complex(0.0, phase_conversion * float(lineinfo[y2_col]))
                        )
                    )
        else:
            warnings.warn(
                "Undefined input type! Use 'realimag', 'dBmagphaserad', 'linmagphaserad', 'dBmagphasedeg' or 'linmagphasedeg'.",
                SyntaxWarning,
            )
        self.f_data = np.array(f_data)
        self.z_data_raw = np.array(z_data_raw)

    def save_fitresults(self, fname: str) -> None:
        pass
