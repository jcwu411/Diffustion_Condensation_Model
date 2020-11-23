#!/usr/bin/env python3.8
"""
Created on 23/11/2020
@author: Jiacheng Wu, jcwu@pku.edu.cn
"""

import numpy as np
import ode_solver as ode


def condenstaion(q):
    qs = 1
    if q > qs:
        s = (q - qs) / tau
    else:
        s = 0
    return s


def p2zpx2(q):
    c = np.fft.fft(q)
    for k in range(kmax):
        c[k] = c[k] * jj * k * (2*np.pi/nx)
        c[-k] = c[-k] * jj * (-k) * (2*np.pi/nx)
    c[kmax] = c[kmax] * jj * kmax * (2*np.pi/nx)
    pypx = np.fft.ifft(c)
    return -u0*pypx.real


if __name__ == '__main__':
    tau = 1
    print("Hello")