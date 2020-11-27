#!/usr/bin/env python3.8
"""
Created on 15/11/2020
@author: Jiacheng Wu, jcwu@pku.edu.cn
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker


def plot_2d(x, y, color, lw, lb,
            ti="Plot", xl="X", yl="Y", legendloc=4,
            xlim=(0, 1), ylim=(0, 1), ylog=False,
            fn="plot2d.pdf", sa=False):
    """
    Plot a line on x-y coordinate system
    :param x:
    :param y:
    :param color: The color of each line
    :param lw: The width of each line
    :param lb: The label of each line
    :param ti: The title of plot
    :param xl: The label of x axis
    :param yl: The label of y axis
    :param legendloc: The location of legend
    :param xlim: The range of x axis
    :param ylim: The range of y axis
    :param ylog: Using logarithmic y axis or not
    :param fn:  The saved file name
    :param sa:  Saving the file or not
    :return: None
    """

    plt.figure()
    plt.plot(x, y, color=color, linewidth=lw, label=lb)

    plt.title(ti)

    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    if ylog:
        plt.yscale('log')

    plt.legend(shadow=True, loc=legendloc)

    if sa:
        plt.savefig(fn)
    plt.show()
    plt.close()


def plot_4figure(x, q, H, T, S):
    plt.subplots(figsize=(7, 6))
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.subplot(221)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(x, q[0], '-b', linewidth=1)
    plt.plot(x, q[1], linestyle='--', color='black', linewidth=1)
    plt.title("Water vapour")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.35)

    my_y_ticks = np.arange(0, 1.5, 0.3)
    plt.yticks(my_y_ticks)

    plt.subplot(222)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(x[0:-4], H[0], '-b', linewidth=1)
    plt.plot(x[0:-4], H[1], linestyle='--', color='black', linewidth=1)
    plt.title("Relative humidity")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.7, 1.1)

    plt.subplot(223)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(x, T[0], '-b', linewidth=1)
    plt.plot(x, T[1], linestyle='--', color='black', linewidth=1)
    plt.title("Temperature")
    plt.xlim(0.0, 1.0)
    plt.ylim(-60.0, 40.0)
    plt.xlabel("Distance, x")

    plt.subplot(224)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(x[0:-4], S[0], '-b', linewidth=1)
    plt.plot(x[0:-4], S[1], linestyle='--', color='black', linewidth=1)
    plt.title("Rainfall")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 3.0)
    plt.xlabel("Distance, x")

    plt.savefig("11.pdf")
    plt.show()
    plt.close()


def plot_4figure_2(x, q, H, T, S):
    plt.subplots(figsize=(7, 6))
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.subplot(221)
    plt.yscale('log')
    plt.plot(x, q, '-b', linewidth=1)
    plt.title("Water vapour")
    plt.xlim(0.0, 1.0)
    plt.ylim(1e-3, 1e1)


    plt.subplot(222)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(x[0:-2], H, '-b', linewidth=1)
    plt.title("Relative humidity")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.1)

    plt.subplot(223)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(x, T, '-b', linewidth=1)
    plt.title("Temperature")
    plt.xlim(0.0, 1.0)
    plt.ylim(-60.0, 40.0)
    plt.xlabel("Distance, x")

    plt.subplot(224)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(x[0:-2], S, '-b', linewidth=1)
    plt.title("Rainfall")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 7.0)
    plt.xlabel("Distance, x")

    plt.savefig("12.pdf")
    plt.show()
    plt.close()


def plot_contour(y, z, q, phi, yl=r"$Latitude$", zl=r"$Altitude$"
                 , contour=True, sa=False):
    yy, zz = np.meshgrid(y, z)
    CF = plt.contourf(yy, zz, q.T, cmap='Blues', extend="both")
    plt.xlabel(yl)
    plt.ylabel(zl)
    plt.title("water vapor")
    cb = plt.colorbar(CF)
    Phi = plt.contour(yy, zz, phi.T, colors='red')
    plt.show()
    plt.close()


def plot_contour2(y, z, q, phi, yl=r"$Latitude$", zl=r"$Altitude$"
                  , contour=True, sa=False):
    yy, zz = np.meshgrid(y, z)
    CF = plt.contourf(yy, zz, q.T, cmap='Blues', levels=np.linspace(0, 1.0, 6))
    plt.xlabel(yl)
    plt.ylabel(zl)
    plt.title("Relative humidity")

    cb = plt.colorbar()
    cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    Phi = plt.contour(yy, zz, phi.T, colors='red')
    plt.show()
    plt.close()


def plot_contour_sub(y, z, q, H, phi, yl=r"$Latitude$", zl=r"$Altitude$"
                     , contour=True, sa=False):
    yy, zz = np.meshgrid(y, z)
    plt.subplots()
    plt.subplots_adjust(wspace=0.3, hspace=0)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.subplot(122)
    lvl = [2.0, 7e-1, 2.4e-1, 8.6e-2, 3.0e-2, 1.0e-2,
           3.7e-3, 1.3e-3, 4.5e-4]
    lvl_v = lvl[:: -1]
    CF = plt.contourf(yy, zz, np.log(q.T), cmap='Blues',
                      levels=np.log(lvl_v), extend="both")
    plt.xlabel(yl)
    plt.ylabel(zl)
    plt.title("Water vapor")
    cb = plt.colorbar(CF)
    Phi = plt.contour(yy, zz, phi.T, colors='red')
    cb.set_ticks(np.log(lvl_v))
    cb.set_ticklabels(["4.5e-04", "1.3e-03", "3.7e-03",
                       "1.0e-02", "3.0e-02", "8.6e-02",
                       "2.4e-01", "7.0e-01", "2.0e-00"])

    plt.subplot(121)
    CH = plt.contourf(yy, zz, H.T, cmap='Blues',
                      levels=np.linspace(0, 1.0, 6), extend="both")
    plt.xlabel(yl)
    plt.ylabel(zl)
    plt.title("Relative humidity")

    cb = plt.colorbar(CH)
    cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    Phi = plt.contour(yy, zz, phi.T, colors='red')

    plt.savefig("21.pdf")
    plt.show()
    plt.close()
