""" Module to perform algebraic circle fitting based on
Efficient and robust analysis of complex scattering data under noise in
microwave resonators
https://arxiv.org/pdf/1410.3365.pdf

and 

https://github.com/sebastianprobst/resonator_tools

"""

import numpy as np
import scipy.optimize
from numpy.polynomial import Polynomial

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


def calc_charpoly_coeffs(A):
    """Calculates the coefficients of the characteristic polynomial of square matrix A
    using the recursive Faddeev-LeVerrier Algorithm

    This algorithm calculates the coefficients of the polynomial in λ described by
    det(A - λI). The coefficient index matches the order of the polynomial term.
    ie. \\sum_{k=0}^n c_k * \\lambda ^ k
    """
    n = A.shape[0]
    if A.shape != (n, n):
        raise Exception(f"Expected a square matrix, got one with shape {A.shape}")
    coeffs = np.zeros(shape=(n + 1))

    # Initialize k=0 case
    coeffs[n] = 1
    temp_M = np.zeros(shape=(n, n))

    for k in np.arange(start=1, stop=n + 1):
        temp_M = np.matmul(A, temp_M) + coeffs[n - k + 1] * np.identity(n)
        coeffs[n - k] = -1 / k * np.trace(np.matmul(A, temp_M))
    return coeffs


def fit_circle(z_data):
    B = np.asmatrix(
        [
            [0.0, 0.0, 0.0, -2.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-2.0, 0.0, 0.0, 0.0],
        ]
    )
    M = calc_moments(z_data)
    MBinv = np.matmul(M, np.linalg.inv(B))
    charpoly_coeffs = np.linalg.det(B) * calc_charpoly_coeffs(MBinv)
    charpoly = Polynomial(charpoly_coeffs)
    roots = scipy.optimize.root_scalar(
        f=charpoly, method="newton", x0=0.0, fprime=charpoly.deriv(1)
    )
    x0 = roots.root
    _, eigenvals, eigenvecs = np.linalg.svd(np.asmatrix(M - x0 * B))
    eigenvec = np.array(eigenvecs[np.argmin(eigenvals), :]).flat

    return calc_center_radius(eigenvec)


def calc_center_radius(eigenvec):
    A, B, C, D = eigenvec
    xc = -B / (2.0 * A)
    yc = -C / (2.0 * A)
    r0 = 1.0 / (2.0 * np.absolute(A)) * np.sqrt(B**2 + C**2 - 4.0 * A * D)
    return xc, yc, r0
