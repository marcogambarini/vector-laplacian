#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from manifold import *
from scipy.optimize import approx_fprime

try:
    plt.style.use('latexsimple')
except:
    pass

# geometric parameters
L = 1
R = 1
#theta = 100*np.pi/180
theta = 50*np.pi/180

# radius of small circles
r = (1 - np.cos(theta))/(1 + np.cos(theta)) * R

# center of left circle
xc_l = np.array((-(r+R)*np.sin(theta), r))

# center of large circle
xc = np.array((0, R))

# center of right circle
xc_r = np.array(((r+R)*np.sin(theta), r))

# chart 1: left line
x0 = np.array((xc_l[0]-L, 0))
x1 = np.array((xc_l[0], 0))
c1 = LineChart(x0, x1)

# chart 2: left small arc
c2 = ArcChart(xc_l, r, -np.pi/2, np.pi/2-theta)

# chart 3: large arc
c3 = ArcChart(xc, R, 3/2*np.pi-theta, -np.pi/2+theta)

# chart 4: right small arc
c4 = ArcChart(xc_r, r, np.pi/2+theta, 3/2*np.pi)

# chart 4: right line
x2 = np.array((xc_r[0], 0))
x3 = np.array((xc_r[0]+L, 0))
c5 = LineChart(x2, x3)

# manifold definition
charts = [c1, c2, c3, c4, c5]
m1 = Manifold(charts)

x = np.array((-0.6, 1.2))
jac = m1.prox_jac(x)
print(jac)

prox_fun = lambda x: m1.proximal_point(x)
fd_jac = approx_fprime(x, m1.proximal_point)
print(fd_jac)

fig, ax = plt.subplots()
m1.plot(ax)

plt.show()
