#!/usr/bin/env python3.8
"""
Created on 23/11/2020
@author: Jiacheng Wu, jcwu@pku.edu.cn
"""

import numpy as np
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


def tend_dcm(q, kappa):
    nq = len(q)
    s = condenstaion(q, nq)

    pqpt = np.zeros(nq)
    for i in range(nq):
        if i == 0:
            pqpt[i] = 0.0
        elif i == nq - 1:
            # pqpt[i] = 0.0
            pqpt[i] = kappa * (- q[i] + q[i - 1]) / dx ** 2 - s[i]
        else:
            pqpt[i] = kappa * (q[i + 1] - 2 * q[i] + q[i - 1]) / dx ** 2 - s[i]

    return pqpt


def scheme_forward(q, kappa):
    # Boundary condition
    q_new = q + dt * tend_dcm(q, kappa)
    q_new[0] = Hb * qs[0]
    # q_new[-1] = q_new[-2]
    return q_new


def integrate(q, kappa):
    for i in range(steps):
        q = scheme_forward(q, kappa)

    q_new = q
    return q_new


if __name__ == '__main__':
    nx = 100
    xb = 1.0
    tau = 1e-2
    kappa1 = 0.001
    kappa2 = kappa1 * 2
    dx = xb / nx
    x = np.arange(0.0, xb, dx)
    dx = x[1] - x[0]
    dt = 0.01
    steps = 10000

    T0 = 273.0      # unit: K
    scheme = 2
    if scheme == 1:
        Ti = 25
        Tb = - 50
        T = (Tb - Ti) / xb * x + Ti + T0
        T_plot = T - T0
    elif scheme == 2:
        xc = 0.7
        Ti = 25
        Tc = - 50
        Tb = - 30
        x1 = np.arange(0, 0.7, dx)
        x2 = np.arange(0.7, xb - dx, dx)
        T1 = (Tc - Ti) / xc * x1 + Ti + T0
        slope2 = (Tb - Tc) / (xb - xc)
        T2 = slope2 * x2 + (Tc - slope2 * xc) + T0
        T = np.concatenate((T1, T2), axis=0)
        T_plot = T - T0
    else:
        print("The type isn't exist")
        raise SystemExit

    es0 = 6.12 * 100  # unit: Pa

    L = 2.44e6  # unit: J/kg
    Rv = 462.0  # unit: J/(kg K)
    es = es0 * np.exp((L / Rv) * (1 / T0 - 1 / T))
    p = 1000 * 100  # unit: Pa
    epsilon = 18.014 / 29.0

    qs = epsilon * es / p * 100

    if scheme == 1:
        Hb = 0.7
        q0 = np.zeros(len(qs))
        q0[0] = Hb * qs[0]
        q_e1 = integrate(q0, kappa1)
        S1 = condenstaion(q_e1, len(q_e1)) * 100
        H1 = q_e1 / qs

        q_e2 = integrate(q0, kappa2)
        S2 = condenstaion(q_e2, len(q_e2)) * 100
        H2 = q_e2 / qs

    elif scheme == 2:
        Hb = 1.0
        q0 = np.zeros(len(qs))
        q0[0] = Hb * qs[0]
        q_e = integrate(q0, kappa1)
        S = condenstaion(q_e, len(q_e)) * 100
        H = q_e / qs

    plot = True
    # plot = False
    if plot:
        if scheme == 1:
            plot_data.plot_4figure(x, [q_e1, q_e2]
                                   , [H1[0:-4], H2[0:-4]]
                                   , [T_plot, T_plot]
                                   , [S1[0:-4], S2[0:-4]])
        elif scheme == 2:
            plot_data.plot_4figure_2(x, q_e
                                     , H[0:-2]
                                     , T_plot
                                     , S[0:-2])
