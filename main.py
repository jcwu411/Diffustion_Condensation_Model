#!/usr/bin/env python3.8
"""
Created on 23/11/2020
@author: Jiacheng Wu, jcwu@pku.edu.cn
"""

import numpy as np
import ode_solver as ode
import pde_solver as pde
import plot_data


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


def condenstaion(q, nq):
    s = np.zeros(nq)
    for i in range(nq):
        if q[i] > qs[i]:
            s[i] = (q[i] - qs[i]) / tau
        else:
            s[i] = 0
    return s


def tend_dcm(q):
    nq = len(q)
    s = condenstaion(q, nq=nq)
    # print("s = ", s)
    pqpt = np.zeros(nq)
    for i in range(nq):
        if i == 0:
            pqpt[i] = 0.0
        elif i == nq - 1:
            pqpt[i] = 0.0
        else:
            pqpt[i] = kappa * (q[i + 1] - 2 * q[i] + q[i-1]) / dx ** 2 - s[i]
    # print("\npqpt", pqpt)
    return pqpt


def scheme_rk4(q):
    # Runge-Kutta-4th-order
    q1 = dt * tend_dcm(q)
    q2 = dt * tend_dcm(q + 1 / 2 * q1)
    q3 = dt * tend_dcm(q + 1 / 2 * q2)
    q4 = dt * tend_dcm(q + q3)
    # q_new = q + 1 / 6 * (q1 + 2 * q2 + 2 * q3 + q4)

    # Boundary condition
    q_new = q + dt * tend_dcm(q)
    q_new[0] = Hb * qs[0]
    q_new[-1] = q_new[-2]
    return q_new


def integrate(q):
    for i in range(steps):
        q = scheme_rk4(q)

    q_new = q
    return q_new


if __name__ == '__main__':
    nx = 64
    xb = 1.0
    tau = 1e-2
    kappa = 0.01
    x_grid = generate_grid(xb, nx)
    dx = x_grid[1] - x_grid[0]

    dt = 0.001
    steps = 5000

    # print("x = {0}".format(x_grid))

    T0 = 273.0  # unit: K
    Ti = 25
    Tb = - 50
    x = np.linspace(0, xb, nx)
    # print("x = {0}".format(x))
    T = (Tb - Ti) / xb * x + Ti + T0
    # print("T = {0}".format(T))

    es0 = 6.12 * 100  # unit: Pa

    L = 2.44e6  # unit: J/kg
    Rv = 462.0  # unit: J/(kg K)
    es = es0 * np.exp((L / Rv) * (1 / T0 - 1 / T))
    # print("es = {0}".format(es))
    p = 1000 * 100  # unit: Pa
    epsilon = 18.014 / 29.0

    qs = epsilon * es / p * 100

    Hb = 0.7
    q0 = np.ones(nx) * Hb * qs[0]
    q0[0] = Hb * qs[0]

    q_e = integrate(q0)
    # print(q_e)
    S = (q_e - qs) / tau
    H = q_e / qs

    plot = True
    # plot = False
    if plot:
        plot_data.plot_2d(x_grid, q_e, color="b", lw=1, lb="q_e",
                          ti="Plot", xl="X", yl="Y", legendloc=1,
                          xlim=(0, 1), ylim=(0, 1.4), ylog=False,
                          fn="plot2d.pdf", sa=False)

        plot_data.plot_2d(x_grid, S, color="b", lw=1, lb="S",
                          ti="Plot", xl="X", yl="Y", legendloc=1,
                          xlim=(0, 1), ylim=(0, 1), ylog=False,
                          fn="plot2d.pdf", sa=False)

        plot_data.plot_2d(x_grid, H, color="b", lw=1, lb="H",
                          ti="Plot", xl="X", yl="Y", legendloc=1,
                          xlim=(0, 1), ylim=(0.9, 1.1), ylog=False,
                          fn="plot2d.pdf", sa=False)

