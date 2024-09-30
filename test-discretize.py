import numpy as np
import matplotlib.pyplot as plt
from manifold import *
from scipy.sparse.linalg import gmres

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

# mesh definition
h = 0.2
x = m1.discretize_by_glueing(h)

# let's solve an harmonic extension problem in the x direction

# scalar stiffness matrix assembly
K = m1.assemble_scalar_stiffness_matrix()

# bc application
bc_L = 0.5
bc_R = 0.
N = x.shape[0]
b = np.zeros(N)
# Dirichlet boundary condition: act on first and last line
# find all elements in first row
ind_row1 = np.where(K.row==0)[0]
for ind in ind_row1:
    if K.col[ind]==0:
        b[0] = K.data[ind]*bc_L
    else:
        K.data[ind] = 0
# find all elements in last row
ind_rowN = np.where(K.row==N-1)[0]
for ind in ind_rowN:
    if K.col[ind]==N-1:
        b[N-1] = K.data[ind]*bc_R
    else:
        K.data[ind] = 0

#plt.spy(K.todense())

sol, info = gmres(K, b)

# now let's move the grid based on the solution
vec_sol = np.zeros((N, 2))
vec_sol[:, 0] = sol
x_updated = x + vec_sol
x_projected = np.zeros((N, 2))
for ii in range(N):
    x_projected[ii, :] = m1.proximal_point(x_updated[ii, :])

fig, ax = plt.subplots(2, 1, sharex=True)
m1.plot(ax[0])
m1.plot_discretization(ax[0])
m1.plot(ax[1])
ax[1].plot(x_updated[:, 0], x_updated[:, 1], 'gx-')
ax[1].plot(x_projected[:, 0], x_projected[:, 1], 'go-', fillstyle='none')

plt.show()
