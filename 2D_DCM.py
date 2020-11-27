#!/usr/bin/env python3.8
"""
Created on 23/11/2020
@author: Jiacheng Wu, jcwu@pku.edu.cn
"""

import numpy as np
import plot_data
from numba import jit


@jit
def generate_grid(zb, n):
    """
    Generate n points within zb
    :param zb: The boundary of x
    :param n: The number of point within a giving range
    :return: x
    """
    d = zb / n
    dhalf = d / 2
    z_grid = np.linspace(dhalf, zb - dhalf, n)
    return z_grid


@jit
def init():
    T_i = np.zeros((ny, nz), dtype=float)
    phi_i = np.zeros((ny, nz), dtype=float)
    v_i = np.zeros((ny, nz), dtype=float)
    w_i = np.zeros((ny, nz), dtype=float)
    for i in range(len(y)):
        for j in range(len(z)):
            T_i[i, j] = (Tb - Ti) / zb * z[j] + Ti + T0
            phi_i[i, j] = np.sin(np.pi * y[i]) * np.sin(np.pi * z[j])
            v_i[i, j] = -1 * np.pi * np.sin(np.pi * y[i]) * np.cos(np.pi * z[j])
            w_i[i, j] = np.pi * np.cos(np.pi * y[i]) * np.sin(np.pi * z[j])
    return T_i, phi_i, v_i, w_i


@jit
def condenstaion(q):
    s = np.zeros((ny, nz), dtype=float)
    for i in range(len(y)):
        for j in range(len(z)):
            if q[i, j] > qs[i, j]:
                s[i, j] = (q[i, j] - qs[i, j]) / tau
            else:
                s[i, j] = 0
    return s


@jit
def pqpy_c(q):
    pqpy = np.zeros((ny, nz), dtype=float)
    for i in range(len(y)):
        for j in range(len(z)):
            if i == 0:
                pqpy[i, j] = 0.0
            elif i == len(y) - 1:
                pqpy[i, j] = 0.0
            else:
                pqpy[i, j] = (q[i + 1, j] - q[i - 1, j]) / (2 * dy)
    return pqpy


@jit
def pqpz_c(q):
    pqpz = np.zeros((ny, nz), dtype=float)
    for i in range(len(y)):
        for j in range(len(z)):
            if j == 0:
                pqpz[i, j] = 0.0
            elif j == len(y) - 1:
                pqpz[i, j] = 0.0
            else:
                pqpz[i, j] = (q[i, j + 1] - q[i, j - 1]) / (2 * dz)
    return pqpz


@jit
def p2qpy2_c(q):
    p2qpy2 = np.zeros((ny, nz), dtype=float)
    for i in range(len(y)):
        for j in range(len(z)):
            if i == 0:
                p2qpy2[i, j] = 2 * (q[i + 1, j] - q[i, j]) / dy ** 2
            elif i == len(y) - 1:
                p2qpy2[i, j] = 2 * (q[i - 1, j] - q[i, j]) / dy ** 2
            else:
                p2qpy2[i, j] = (q[i + 1, j] - 2 * q[i, j] + q[i - 1, j]) / dy ** 2
    return p2qpy2


@jit
def p2qpz2_c(q):
    p2qpz2 = np.zeros((ny, nz), dtype=float)
    for i in range(len(y)):
        for j in range(len(z)):
            if j == 0:
                p2qpz2[i, j] = 0.0
            elif j == len(z) - 1:
                p2qpz2[i, j] = (q[i, j - 1] - q[i, j]) / dz ** 2
            else:
                p2qpz2[i, j] = (q[i, j + 1] - 2 * q[i, j] + q[i, j - 1]) / dz ** 2
    return p2qpz2


@jit
def tend_dcm(q):
    s = condenstaion(q)
    pqpt = -1 * (v * pqpy_c(q) + w * pqpz_c(q)) + kappa * (p2qpy2_c(q) + p2qpz2_c(q)) - s
    return pqpt


@jit
def scheme_forward(q):
    # Boundary condition
    q_new = q + dt * tend_dcm(q)
    q_new[:, 0] = q0[:, 0]
    return q_new


@jit
def scheme_rk4(q):
    # Boundary condition
    q1 = dt * tend_dcm(q)
    q2 = dt * tend_dcm(q + 1 / 2 * q1)
    q3 = dt * tend_dcm(q + 1 / 2 * q2)
    q4 = dt * tend_dcm(q + q3)
    q_new = q + 1 / 6 * (q1 + 2 * q2 + 2 * q3 + q4)
    q_new[:, 0] = q0[:, 0]
    return q_new


@jit
def integrate(q):
    for step in range(steps):
        q = scheme_forward(q)
    q_new = q
    return q_new


if __name__ == '__main__':
    Ny = 100
    Nz = 100
    yb = 1.0
    zb = 1.0
    tau = 0.001
    kappa = 0.001
    y = np.linspace(0, yb, Ny + 1)
    z = np.linspace(0, zb, Nz + 1)
    ny = len(y)
    nz = len(z)
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    dt = 0.0001
    steps = 100000

    T0 = 273.0  # unit: K
    Ti = 26
    Tb = - 50
    T, phi, v, w = init()

    es0 = 6.12 * 100  # unit: Pa

    L = 2.44e6  # unit: J/kg
    Rv = 462.0  # unit: J/(kg K)
    es = es0 * np.exp((L / Rv) * (1 / T0 - 1 / T))
    p = 1000 * 100  # unit: Pa
    epsilon = 18.014 / 29.0

    qs = epsilon * es / p * 100

    Hb = 1.0
    # q0 = np.ones((ny, nz), dtype=float) + 0.7 * qs
    q0 = np.zeros((ny, nz), dtype=float)
    q0[:, 0] = Hb * qs[:, 0]

    q_e = integrate(q0)
    H = q_e / qs
    plot = True
    # plot = False
    if plot:
        plot_data.plot_contour_sub(y, z, q_e, H, phi)


