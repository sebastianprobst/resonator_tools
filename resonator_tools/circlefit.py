import numpy as np
import numpy.typing as npt
import scipy.optimize as spopt
from scipy import stats
from typing import Any


FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]


class circlefit(object):
    """
    contains all the circlefit procedures
    see http://scitation.aip.org/content/aip/journal/rsi/86/2/10.1063/1.4907935
    arxiv version: http://arxiv.org/abs/1410.3365
    """

    def _remove_cable_delay(
        self, f_data: FloatArray, z_data: ComplexArray, delay: float
    ) -> ComplexArray:
        return z_data / np.exp(2j * np.pi * f_data * delay)

    def _center(self, z_data: ComplexArray, zc: complex) -> ComplexArray:
        return z_data - zc

    def _dist(self, x: FloatArray) -> FloatArray:
        np.absolute(x, x)
        c = (x > np.pi).astype(int)
        return x + c * (-2.0 * x + 2.0 * np.pi)

    def _periodic_boundary(self, x: float, bound: float) -> float:
        return np.fmod(x, bound) - np.trunc(x / bound) * bound

    def _phase_fit_wslope(
        self,
        f_data: FloatArray,
        z_data: ComplexArray,
        theta0: float,
        Ql: float,
        fr: float,
        slope: float,
    ) -> npt.NDArray[np.float64]:
        phase = np.angle(z_data)
        # Normalize so all parameters are ~O(1)
        fr_scale = np.mean(f_data)
        Ql_scale = max(abs(Ql), 1.0)
        slope_scale = max(abs(slope), 1.0 / fr_scale)
        x_norm = np.array(f_data) / fr_scale

        def residuals(p, x_n, y):
            theta0_n, Ql_n, fr_n, slope_n = p
            _theta0 = theta0_n
            _Ql = Ql_n * Ql_scale
            _fr = fr_n * fr_scale
            _slope = slope_n * slope_scale
            err = self._dist(
                y - (_theta0 + 2.0 * np.arctan(2.0 * _Ql * (1.0 - x_n * fr_scale / _fr)) - _slope * x_n * fr_scale)
            )
            return err

        p0 = [theta0, Ql / Ql_scale, fr / fr_scale, slope / slope_scale]
        p_final = spopt.leastsq(
            residuals,
            p0,
            args=(x_norm, np.array(phase)),
            ftol=1e-12,
            xtol=1e-12,
        )
        theta0_f, Ql_f, fr_f, slope_f = p_final[0]
        return np.array([theta0_f, Ql_f * Ql_scale, fr_f * fr_scale, slope_f * slope_scale])

    def _phase_fit(
        self,
        f_data: FloatArray,
        z_data: ComplexArray,
        theta0: float,
        Ql: float,
        fr: float,
    ) -> npt.NDArray[np.float64]:
        phase = np.angle(z_data)
        # Normalize: fit with fr_n = fr/fr_scale (~1), Ql_n = Ql/Ql_scale (~1)
        fr_scale = np.mean(f_data)
        Ql_scale = max(abs(Ql), 1.0)
        x_norm = np.array(f_data) / fr_scale

        def _phase_model(x_n, theta0, Ql_n, fr_n):
            return theta0 + 2.0 * np.arctan(2.0 * Ql_n * Ql_scale * (1.0 - x_n / fr_n))

        def residuals_1(p, x_n, y, Ql_n):
            theta0, fr_n = p
            return self._dist(y - _phase_model(x_n, theta0, Ql_n, fr_n))

        def residuals_2(p, x_n, y, theta0):
            Ql_n, fr_n = p
            return self._dist(y - _phase_model(x_n, theta0, Ql_n, fr_n))

        def residuals_3(p, x_n, y, theta0, Ql_n):
            fr_n = p
            return self._dist(y - _phase_model(x_n, theta0, Ql_n, fr_n))

        def residuals_4(p, x_n, y, theta0, fr_n):
            Ql_n = p
            return self._dist(y - _phase_model(x_n, theta0, Ql_n, fr_n))

        def residuals_5(p, x_n, y):
            theta0, Ql_n, fr_n = p
            return self._dist(y - _phase_model(x_n, theta0, Ql_n, fr_n))

        fr_n = fr / fr_scale
        Ql_n = Ql / Ql_scale

        p0 = [theta0, fr_n]
        p_final = spopt.leastsq(
            lambda a, b, c: residuals_1(a, b, c, Ql_n),
            p0,
            args=(x_norm, phase),
            ftol=1e-12,
            xtol=1e-12,
        )
        theta0, fr_n = p_final[0]
        p0 = [Ql_n, fr_n]
        p_final = spopt.leastsq(
            lambda a, b, c: residuals_2(a, b, c, theta0),
            p0,
            args=(x_norm, phase),
            ftol=1e-12,
            xtol=1e-12,
        )
        Ql_n, fr_n = p_final[0]
        p0 = fr_n
        p_final = spopt.leastsq(
            lambda a, b, c: residuals_3(a, b, c, theta0, Ql_n),
            p0,
            args=(x_norm, phase),
            ftol=1e-12,
            xtol=1e-12,
        )
        fr_n = p_final[0][0]
        p0 = Ql_n
        p_final = spopt.leastsq(
            lambda a, b, c: residuals_4(a, b, c, theta0, fr_n),
            p0,
            args=(x_norm, phase),
            ftol=1e-12,
            xtol=1e-12,
        )
        Ql_n = p_final[0][0]
        p0 = np.array([theta0, Ql_n, fr_n], dtype="float64")
        p_final = spopt.leastsq(
            residuals_5,
            p0,
            args=(x_norm, phase),
            ftol=1e-12,
            xtol=1e-12,
        )
        theta0_f, Ql_n_f, fr_n_f = p_final[0]
        return np.array([theta0_f, Ql_n_f * Ql_scale, fr_n_f * fr_scale])

    def _fit_skewed_lorentzian(
        self, f_data: FloatArray, z_data: ComplexArray
    ) -> npt.NDArray[np.float64]:
        amplitude = np.absolute(z_data)
        amplitude_sqr = amplitude**2
        A1a = np.minimum(amplitude_sqr[0], amplitude_sqr[-1])
        A3a = -np.max(amplitude_sqr)
        fra = f_data[np.argmin(amplitude_sqr)]
        # Normalize frequency so fr parameter is ~O(1)
        fr_scale = np.mean(f_data)
        f_norm = np.array(f_data) / fr_scale
        fra_n = fra / fr_scale
        # Scale slope coefficients to match normalized frequency
        amp_sqr = np.array(amplitude_sqr)

        def residuals(p, x_n, y):
            A2_n, A4_n, Ql = p
            err = y - (
                A1a
                + A2_n * fr_scale * (x_n - fra_n)
                + (A3a + A4_n * fr_scale * (x_n - fra_n)) / (1.0 + 4.0 * Ql**2 * ((x_n - fra_n) / fra_n) ** 2)
            )
            return err

        p0 = [0.0, 0.0, 1e3]
        p_final = spopt.leastsq(
            residuals,
            p0,
            args=(f_norm, amp_sqr),
            ftol=1e-12,
            xtol=1e-12,
        )
        A2a_n, A4a_n, Qla = p_final[0]
        A2a = A2a_n
        A4a = A4a_n

        def fitfunc(x_n, A1, A2_n, A3, A4_n, fr_n, Ql):
            return (
                A1
                + A2_n * fr_scale * (x_n - fr_n)
                + (A3 + A4_n * fr_scale * (x_n - fr_n)) / (1.0 + 4.0 * Ql**2 * ((x_n - fr_n) / fr_n) ** 2)
            )

        p0 = [A1a, A2a, A3a, A4a, fra_n, Qla]
        try:
            popt, pcov = spopt.curve_fit(
                fitfunc, f_norm, amp_sqr, p0=p0,
                bounds=(
                    [-np.inf, -np.inf, -np.inf, -np.inf, f_norm[0], 0],
                    [np.inf, np.inf, np.inf, np.inf, f_norm[-1], np.inf],
                ),
            )
            if pcov is not None:
                self.df_error = np.sqrt(pcov[4][4]) * fr_scale
                self.dQl_error = np.sqrt(pcov[5][5])
            else:
                self.df_error = np.inf
                self.dQl_error = np.inf
            # Convert back to original frequency units
            popt[4] *= fr_scale  # fr
        except Exception:
            popt = np.array([A1a, A2a, A3a, A4a, fra, Qla])
            self.df_error = np.inf
            self.dQl_error = np.inf
        return popt  # type: ignore

    def _fit_circle(
        self, z_data: ComplexArray, refine_results: bool = False
    ) -> tuple[float, float, float]:
        def calc_moments(z_data):
            xi = z_data.real
            xi_sqr = xi * xi
            yi = z_data.imag
            yi_sqr = yi * yi
            zi = xi_sqr + yi_sqr
            Nd = float(len(xi))
            xi_sum = xi.sum()
            yi_sum = yi.sum()
            zi_sum = zi.sum()
            xiyi_sum = (xi * yi).sum()
            xizi_sum = (xi * zi).sum()
            yizi_sum = (yi * zi).sum()
            return np.array(
                [
                    [(zi * zi).sum(), xizi_sum, yizi_sum, zi_sum],
                    [xizi_sum, xi_sqr.sum(), xiyi_sum, xi_sum],
                    [yizi_sum, xiyi_sum, yi_sqr.sum(), yi_sum],
                    [zi_sum, xi_sum, yi_sum, Nd],
                ]
            )

        M = calc_moments(z_data)

        a0 = (
            (
                (M[2][0] * M[3][2] - M[2][2] * M[3][0]) * M[1][1]
                - M[1][2] * M[2][0] * M[3][1]
                - M[1][0] * M[2][1] * M[3][2]
                + M[1][0] * M[2][2] * M[3][1]
                + M[1][2] * M[2][1] * M[3][0]
            )
            * M[0][3]
            + (
                M[0][2] * M[2][3] * M[3][0]
                - M[0][2] * M[2][0] * M[3][3]
                + M[0][0] * M[2][2] * M[3][3]
                - M[0][0] * M[2][3] * M[3][2]
            )
            * M[1][1]
            + (
                M[0][1] * M[1][3] * M[3][0]
                - M[0][1] * M[1][0] * M[3][3]
                - M[0][0] * M[1][3] * M[3][1]
            )
            * M[2][2]
            + (-M[0][1] * M[1][2] * M[2][3] - M[0][2] * M[1][3] * M[2][1]) * M[3][0]
            + (
                (M[2][3] * M[3][1] - M[2][1] * M[3][3]) * M[1][2]
                + M[2][1] * M[3][2] * M[1][3]
            )
            * M[0][0]
            + (
                M[1][0] * M[2][3] * M[3][2]
                + M[2][0] * (M[1][2] * M[3][3] - M[1][3] * M[3][2])
            )
            * M[0][1]
            + (
                (M[2][1] * M[3][3] - M[2][3] * M[3][1]) * M[1][0]
                + M[1][3] * M[2][0] * M[3][1]
            )
            * M[0][2]
        )
        a1 = (
            (
                (M[3][0] - 2.0 * M[2][2]) * M[1][1]
                - M[1][0] * M[3][1]
                + M[2][2] * M[3][0]
                + 2.0 * M[1][2] * M[2][1]
                - M[2][0] * M[3][2]
            )
            * M[0][3]
            + (
                2.0 * M[2][0] * M[3][2]
                - M[0][0] * M[3][3]
                - 2.0 * M[2][2] * M[3][0]
                + 2.0 * M[0][2] * M[2][3]
            )
            * M[1][1]
            + (-M[0][0] * M[3][3] + 2.0 * M[0][1] * M[1][3] + 2.0 * M[1][0] * M[3][1])
            * M[2][2]
            + (-M[0][1] * M[1][3] + 2.0 * M[1][2] * M[2][1] - M[0][2] * M[2][3])
            * M[3][0]
            + (M[1][3] * M[3][1] + M[2][3] * M[3][2]) * M[0][0]
            + (M[1][0] * M[3][3] - 2.0 * M[1][2] * M[2][3]) * M[0][1]
            + (M[2][0] * M[3][3] - 2.0 * M[1][3] * M[2][1]) * M[0][2]
            - 2.0 * M[1][2] * M[2][0] * M[3][1]
            - 2.0 * M[1][0] * M[2][1] * M[3][2]
        )
        a2 = (
            (2.0 * M[1][1] - M[3][0] + 2.0 * M[2][2]) * M[0][3]
            + (2.0 * M[3][0] - 4.0 * M[2][2]) * M[1][1]
            - 2.0 * M[2][0] * M[3][2]
            + 2.0 * M[2][2] * M[3][0]
            + M[0][0] * M[3][3]
            + 4.0 * M[1][2] * M[2][1]
            - 2.0 * M[0][1] * M[1][3]
            - 2.0 * M[1][0] * M[3][1]
            - 2.0 * M[0][2] * M[2][3]
        )
        a3 = -2.0 * M[3][0] + 4.0 * M[1][1] + 4.0 * M[2][2] - 2.0 * M[0][3]
        a4 = -4.0

        def func(x):
            return a0 + a1 * x + a2 * x * x + a3 * x * x * x + a4 * x * x * x * x

        def d_func(x):
            return a1 + 2 * a2 * x + 3 * a3 * x * x + 4 * a4 * x * x * x

        x0 = spopt.fsolve(func, 0.0, fprime=d_func)

        def solve_eq_sys(val, M):
            # prepare
            M[3][0] = M[3][0] + 2 * val
            M[0][3] = M[0][3] + 2 * val
            M[1][1] = M[1][1] - val
            M[2][2] = M[2][2] - val
            return np.linalg.svd(M)

        U, s, Vt = solve_eq_sys(x0[0], M)

        A_vec = Vt[np.argmin(s), :]

        xc = -A_vec[1] / (2.0 * A_vec[0])
        yc = -A_vec[2] / (2.0 * A_vec[0])
        # the term *sqrt term corrects for the constraint, because it may be altered due to numerical inaccuracies during calculation
        r0 = (
            1.0
            / (2.0 * np.absolute(A_vec[0]))
            * np.sqrt(
                A_vec[1] * A_vec[1] + A_vec[2] * A_vec[2] - 4.0 * A_vec[0] * A_vec[3]
            )
        )
        if refine_results:
            print("agebraic r0: " + str(r0))
            xc, yc, r0 = self._fit_circle_iter(z_data, xc, yc, r0)
            r0 = self._fit_circle_iter_radialweight(z_data, xc, yc, r0)
            print("iterative r0: " + str(r0))
        return xc, yc, r0

    def _guess_delay(self, f_data: FloatArray, z_data: ComplexArray) -> float:
        phase2 = np.unwrap(np.angle(z_data))
        gradient = stats.linregress(f_data, phase2)[0]
        gradient = float(gradient)  # type: ignore
        return gradient * (-1.0) / (np.pi * 2.0)

    def _fit_delay(
        self,
        f_data: FloatArray,
        z_data: ComplexArray,
        delay: float = 0.0,
        maxiter: int = 0,
    ) -> float:
        # Normalize delay so the parameter is ~O(1)
        delay_scale = max(abs(delay), 1.0 / np.mean(f_data))

        def residuals(p, x, y):
            phasedelay = p[0] * delay_scale
            z_data_temp = y * np.exp(1j * (2.0 * np.pi * phasedelay * x))
            xc, yc, r0 = self._fit_circle(z_data_temp)
            err = (
                np.sqrt((z_data_temp.real - xc) ** 2 + (z_data_temp.imag - yc) ** 2)
                - r0
            )
            return err

        p_final = spopt.leastsq(
            residuals,
            [delay / delay_scale],
            args=(f_data, z_data),
            maxfev=maxiter,
            ftol=1e-12,
            xtol=1e-12,
        )
        return p_final[0][0] * delay_scale

    def _fit_delay_alt_bigdata(
        self,
        f_data: FloatArray,
        z_data: ComplexArray,
        delay: float = 0.0,
        maxiter: int = 0,
    ) -> float:
        # Normalize delay so the parameter is ~O(1)
        delay_scale = max(abs(delay), 1.0 / np.mean(f_data))

        def residuals(p, x, y):
            phasedelay = p[0] * delay_scale
            z_data_temp = 1j * 2.0 * np.pi * phasedelay * x
            np.exp(z_data_temp, out=z_data_temp)
            np.multiply(y, z_data_temp, out=z_data_temp)
            xc, yc, r0 = self._fit_circle(z_data_temp)
            err = (
                np.sqrt((z_data_temp.real - xc) ** 2 + (z_data_temp.imag - yc) ** 2)
                - r0
            )
            return err

        p_final = spopt.leastsq(
            residuals,
            [delay / delay_scale],
            args=(f_data, z_data),
            maxfev=maxiter,
            ftol=1e-12,
            xtol=1e-12,
        )
        return p_final[0][0] * delay_scale

    def _fit_entire_model(
        self,
        f_data: FloatArray,
        z_data: ComplexArray,
        fr: float,
        absQc: float,
        Ql: float,
        phi0: float,
        delay: float,
        a: float = 1.0,
        alpha: float = 0.0,
        maxiter: int = 0,
    ) -> tuple[Any, Any, Any, str, int]:
        """
        fits the whole model: a*exp(i*alpha)*exp(-2*pi*i*f*delay) * [ 1 - {Ql/Qc*exp(i*phi0)} / {1+2*i*Ql*(f-fr)/fr} ]
        """
        # Normalization scales so all parameters are ~O(1)
        fr_scale = np.mean(f_data)
        Qc_scale = max(abs(absQc), 1.0)
        Ql_scale = max(abs(Ql), 1.0)
        delay_scale = max(abs(delay), 1.0 / fr_scale)
        a_scale = max(abs(a), 1e-12)
        x = np.array(f_data)
        y = np.array(z_data)

        def _model_vec(p_n, x):
            fr_n, Qc_n, Ql_n, phi0, delay_n, a_n, alpha = p_n
            _fr = fr_n * fr_scale
            _Qc = Qc_n * Qc_scale
            _Ql = Ql_n * Ql_scale
            _delay = delay_n * delay_scale
            _a = a_n * a_scale
            phase_delay = -2.0 * np.pi * _delay * x
            detuning = 2.0 * _Ql * (x - _fr) / _fr
            return (
                _a
                * np.exp(1j * alpha)
                * np.exp(1j * phase_delay)
                * (1.0 - (_Ql / _Qc * np.exp(1j * phi0)) / (1.0 + 1j * detuning))
            )

        def funcsqr(p_n, x):
            return np.abs(_model_vec(p_n, x)) ** 2

        def residuals(p_n, x, y):
            return np.abs(y - _model_vec(p_n, x))

        p0_n = [
            fr / fr_scale,
            absQc / Qc_scale,
            Ql / Ql_scale,
            phi0,
            delay / delay_scale,
            a / a_scale,
            alpha,
        ]
        result = spopt.leastsq(
            residuals,
            p0_n,
            args=(x, y),
            full_output=True,
            maxfev=maxiter,
            ftol=1e-12,
            xtol=1e-12,
        )
        popt_n, params_cov, infodict, errmsg, ier = result  # type: ignore
        # De-normalize optimized parameters
        popt = np.array(popt_n, dtype=np.float64)
        popt[0] *= fr_scale
        popt[1] *= Qc_scale
        popt[2] *= Ql_scale
        # popt[3] = phi0 (already ~O(1))
        popt[4] *= delay_scale
        popt[5] *= a_scale
        # popt[6] = alpha (already ~O(1))
        len_ydata = len(x)
        if (
            (len_ydata > len(p0_n)) and params_cov is not None
        ):
            # Transform covariance back to original parameter space
            scale_vec = np.array([fr_scale, Qc_scale, Ql_scale, 1.0, delay_scale, a_scale, 1.0])
            s_sq = funcsqr(popt_n, x).sum() / (len_ydata - len(p0_n))
            params_cov = params_cov * s_sq
            params_cov = np.outer(scale_vec, scale_vec) * params_cov
        else:
            params_cov = np.inf
        return popt, params_cov, infodict, errmsg, ier

    #

    def _optimizedelay(
        self,
        f_data: FloatArray,
        z_data: ComplexArray,
        Ql: float,
        fr: float,
        maxiter: int = 4,
    ) -> float:
        xc, yc, r0 = self._fit_circle(z_data)
        z_data = self._center(z_data, complex(xc, yc))
        theta, Ql, fr, slope = self._phase_fit_wslope(f_data, z_data, 0.0, Ql, fr, 0.0)
        delay = 0.0
        for i in range(maxiter - 1):  # interate to get besser phase delay term
            delay = delay - slope / (2.0 * 2.0 * np.pi)
            z_data_corr = self._remove_cable_delay(f_data, z_data, delay)
            xc, yc, r0 = self._fit_circle(z_data_corr)
            z_data_corr2 = self._center(z_data_corr, complex(xc, yc))
            theta0, Ql, fr, slope = self._phase_fit_wslope(
                f_data, z_data_corr2, 0.0, Ql, fr, 0.0
            )
        delay = delay - slope / (2.0 * 2.0 * np.pi)  # start final interation
        return delay

    def _fit_circle_iter(
        self, z_data: ComplexArray, xc: float, yc: float, rc: float
    ) -> tuple[float, float, float]:
        """
        this is the radial weighting procedure
        it improves your fitting value for the radius = Ql/Qc
        use this to improve your fit in presence of heavy noise
        after having used the standard algebraic fir_circle() function
        the weight here is: W=1/sqrt((xc-xi)^2+(yc-yi)^2)
        this works, because the center of the circle is usually much less
        corrupted by noise than the radius
        """
        xdat = z_data.real
        ydat = z_data.imag

        def fitfunc(x, y, xc, yc):
            return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

        def residuals(p, x, y):
            xc, yc, r = p
            temp = r - fitfunc(x, y, xc, yc)
            return temp

        p0 = [xc, yc, rc]
        p_final = spopt.leastsq(
            residuals,
            p0,
            args=(xdat, ydat),
            ftol=1e-12,
            xtol=1e-12,
        )
        xc, yc, rc = p_final[0]
        return xc, yc, rc

    def _fit_circle_iter_radialweight(
        self, z_data: ComplexArray, xc: float, yc: float, rc: float
    ) -> float:
        """
        this is the radial weighting procedure
        it improves your fitting value for the radius = Ql/Qc
        use this to improve your fit in presence of heavy noise
        after having used the standard algebraic fir_circle() function
        the weight here is: W=1/sqrt((xc-xi)^2+(yc-yi)^2)
        this works, because the center of the circle is usually much less
        corrupted by noise than the radius
        """
        xdat = z_data.real
        ydat = z_data.imag

        def fitfunc(x, y):
            return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

        def weight(x, y):
            try:
                res = 1.0 / np.sqrt((xc - x) ** 2 + (yc - y) ** 2)
            except Exception:
                res = 1.0
            return res

        def residuals(p, x, y):
            r = p[0]
            temp = (r - fitfunc(x, y)) * weight(x, y)
            return temp

        p0 = [rc]
        p_final = spopt.leastsq(
            residuals,
            p0,
            args=(xdat, ydat),
            ftol=1e-12,
            xtol=1e-12,
        )
        return p_final[0][0]

    # def _get_errors(
    #     self, residual: Any, xdata: Any, ydata: Any, fitparams: Any
    # ) -> tuple[float, Any]:
    #     """
    #     wrapper for get_cov, only gives the errors and chisquare
    #     """
    #     chisqr, cov = self._get_cov(residual, xdata, ydata, fitparams)
    #     if cov is not None:
    #         errors = np.sqrt(np.diagonal(cov))
    #     else:
    #         errors = None
    #     return chisqr, errors

    def _residuals_notch_full(self, p: Any, x: Any, y: Any) -> Any:
        fr, absQc, Ql, phi0, delay, a, alpha = p
        err = np.absolute(
            y
            - (
                a
                * np.exp(complex(0, alpha))
                * np.exp(complex(0, -2.0 * np.pi * delay * x))
                * (
                    1
                    - (Ql / absQc * np.exp(complex(0, phi0)))
                    / (complex(1, 2 * Ql * (x - fr) / float(fr)))
                )
            )
        )
        return err

    def _residuals_notch_ideal(self, p: Any, x: Any, y: Any) -> Any:
        fr, absQc, Ql, phi0 = p
        # if fr == 0: print(p)
        err = np.absolute(
            y
            - (
                1.0
                - (Ql / float(absQc) * np.exp(1j * phi0))
                / (1 + 2j * Ql * (x - fr) / float(fr))
            )
        )
        # if np.isinf((complex(1,2*Ql*(x-fr)/float(fr))).imag):
        #   print(complex(1,2*Ql*(x-fr)/float(fr)))
        #  print("x: " + str(x))
        # print("Ql: " +str(Ql))
        # print("fr: " +str(fr))
        return err

    def _residuals_notch_ideal_complex(self, p: Any, x: Any, y: Any) -> Any:
        fr, absQc, Ql, phi0 = p
        # if fr == 0: print(p)
        err = y - (
            1.0
            - (Ql / float(absQc) * np.exp(1j * phi0))
            / (1 + 2j * Ql * (x - fr) / float(fr))
        )
        # if np.isinf((complex(1,2*Ql*(x-fr)/float(fr))).imag):
        #   print(complex(1,2*Ql*(x-fr)/float(fr)))
        #  print("x: " + str(x))
        # print("Ql: " +str(Ql))
        # print("fr: " +str(fr))
        return err

    def _residuals_directrefl(self, p: Any, x: Any, y: Any) -> Any:
        fr, Qc, Ql = p
        # if fr == 0: print(p)
        err = y - (2.0 * Ql / Qc - 1.0 + 2j * Ql * (fr - x) / fr) / (
            1.0 - 2j * Ql * (fr - x) / fr
        )
        # if np.isinf((complex(1,2*Ql*(x-fr)/float(fr))).imag):
        #   print(complex(1,2*Ql*(x-fr)/float(fr)))
        #  print("x: " + str(x))
        # print("Ql: " +str(Ql))
        # print("fr: " +str(fr))
        return err

    def _residuals_transm_ideal(self, p: Any, x: Any, y: Any) -> Any:
        fr, Ql = p
        err = np.absolute(y - (1.0 / (complex(1, 2 * Ql * (x - fr) / float(fr)))))
        return err

    def _get_cov_fast_notch(
        self, xdata: Any, ydata: Any, fitparams: Any
    ) -> tuple[float, Any]:  # enhanced by analytical derivatives
        # derivatives of notch_ideal model with respect to parameters
        def dS21_dQl(p, f):
            fr, absQc, Ql, phi0 = p
            return -(np.exp(1j * phi0) * fr**2) / (
                absQc * (fr + 2j * Ql * f - 2j * Ql * fr) ** 2
            )

        def dS21_dQc(p, f):
            fr, absQc, Ql, phi0 = p
            return (np.exp(1j * phi0) * Ql * fr) / (
                2j * (f - fr) * absQc**2 * Ql + absQc**2 * fr
            )

        def dS21_dphi0(p, f):
            fr, absQc, Ql, phi0 = p
            return -(1j * Ql * fr * np.exp(1j * phi0)) / (
                2j * (f - fr) * absQc * Ql + absQc * fr
            )

        def dS21_dfr(p, f):
            fr, absQc, Ql, phi0 = p
            return -(2j * Ql**2 * f * np.exp(1j * phi0)) / (
                absQc * (fr + 2j * Ql * f - 2j * Ql * fr) ** 2
            )

        u = self._residuals_notch_ideal_complex(fitparams, xdata, ydata)
        chi = np.absolute(u)
        u = u / chi  # unit vector pointing in the correct direction for the derivative

        aa = dS21_dfr(fitparams, xdata)
        bb = dS21_dQc(fitparams, xdata)
        cc = dS21_dQl(fitparams, xdata)
        dd = dS21_dphi0(fitparams, xdata)

        Jt = np.array(
            [
                aa.real * u.real + aa.imag * u.imag,
                bb.real * u.real + bb.imag * u.imag,
                cc.real * u.real + cc.imag * u.imag,
                dd.real * u.real + dd.imag * u.imag,
            ]
        )
        A = np.dot(Jt, np.transpose(Jt))
        chisqr = 1.0 / float(len(xdata) - len(fitparams)) * (chi**2).sum()
        try:
            cov = np.linalg.pinv(A) * chisqr
        except Exception:
            cov = None
        return chisqr, cov

    def _get_cov_fast_directrefl(
        self, xdata: Any, ydata: Any, fitparams: Any
    ) -> tuple[float, Any]:  # enhanced by analytical derivatives
        # derivatives of notch_ideal model with respect to parameters
        def dS21_dQl(p, f):
            fr, Qc, Ql = p
            return 2.0 * fr**2 / (Qc * (2j * Ql * fr - 2j * Ql * f + fr) ** 2)

        def dS21_dQc(p, f):
            fr, Qc, Ql = p
            return 2.0 * Ql * fr / (2j * Qc**2 * (f - fr) * Ql - Qc**2 * fr)

        def dS21_dfr(p, f):
            fr, Qc, Ql = p
            return -4j * Ql**2 * f / (Qc * (2j * Ql * fr - 2j * Ql * f + fr) ** 2)

        u = self._residuals_directrefl(fitparams, xdata, ydata)
        chi = np.absolute(u)
        u = u / chi  # unit vector pointing in the correct direction for the derivative

        aa = dS21_dfr(fitparams, xdata)
        bb = dS21_dQc(fitparams, xdata)
        cc = dS21_dQl(fitparams, xdata)

        Jt = np.array(
            [
                aa.real * u.real + aa.imag * u.imag,
                bb.real * u.real + bb.imag * u.imag,
                cc.real * u.real + cc.imag * u.imag,
            ]
        )
        A = np.dot(Jt, np.transpose(Jt))
        chisqr = 1.0 / float(len(xdata) - len(fitparams)) * (chi**2).sum()
        try:
            cov = np.linalg.pinv(A) * chisqr
        except Exception:
            cov = None
        return chisqr, cov
