import numpy as np
import matplotlib.pyplot as plt
from manifold import *
from scipy.sparse.linalg import gmres

try:
    plt.style.use('latexsimple')
except:
    pass

# geometric parameters
L = 1
R = 1
#theta = 120*np.pi/180
#theta = 100*np.pi/180
theta = 50*np.pi/180

# mesh size
h = 0.2

# penalty coefficient for tangentiality constraint
k_penalty = 1.

# source term coefficient
k_source = 0.5

# boundary conditions
bc_L = np.array((0.5, 0.))
bc_R = np.array((0., 0.))

#------------ end of user-defined parameters ----------------#

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
# perturbation makes almost sure that there are no nodes on chart boundaries
x = m1.discretize_by_glueing(h, perturb=0.)
print('points on each chart in initial grid')
print(m1.point_distribution_on_charts(x))

# let's solve an harmonic extension problem in the x direction

# matrices assembly
K, M = m1.assemble_vector_matrices()
A = K + k_penalty/h**2 * M # system matrix

# rhs assembly
b = k_source*m1.assemble_vector_rhs()

# fig, ax = plt.subplots(1, 2)
# ax[0].spy(K.todense())
# ax[0].set_title('K')
# ax[1].spy(M.todense())
# ax[1].set_title('M')

# fig, ax = plt.subplots()
# ax.spy(A.todense())


#print(np.linalg.det(A.todense()))


# bc application
N = x.shape[0]
# Dirichlet boundary condition: act on first and last line
# the matrix is in csr format
for ii in range(2):
    # first node
    row_ind = ii*N
    row_start = A.indptr[row_ind]
    row_end = A.indptr[row_ind+1]
    for col_ind in range(row_start, row_end):
        # if it is a diagonal element, keep it and set the rhs
        if A.indices[col_ind]==row_ind:
            # done like this to try to keep some balance
            # (setting the diag element to 1 may ruin the condition number)
            b[row_ind] = A.data[col_ind]*bc_L[ii]
        # otherwise, set to zero
        else:
            A.data[col_ind] = 0
    # last node
    row_ind = N-1 + ii*N
    row_start = A.indptr[row_ind]
    row_end = A.indptr[row_ind+1]
    for col_ind in range(row_start, row_end):
        if A.indices[col_ind]==row_ind:
            b[row_ind] = A.data[col_ind]*bc_R[ii]
        else:
            A.data[col_ind] = 0

sol, info = gmres(A, b)
print('GMRES solver finished with status ', info)


# now let's move the grid based on the solution
vec_sol = np.zeros((N, 2))
vec_sol[:, 0] = sol[:N]
vec_sol[:, 1] = sol[N:]
x_updated = x + vec_sol
x_projected = np.zeros((N, 2))
for ii in range(N):
    x_projected[ii, :] = m1.proximal_point(x_updated[ii, :])
print('points on each chart in final grid')
print(m1.point_distribution_on_charts(x_projected))

fig, ax = plt.subplots()
m1.plot(ax)
m1.plot_discretization(ax)

# fig, ax = plt.subplots(3, 1)
# ax[0].plot(b[:N])
# ax[1].plot(sol[:N])
# ax[2].plot(-np.diff(np.diff(sol[:N])))
# ax[0].set_title('x component')
# ax[0].set_ylabel('source')
# ax[1].set_ylabel('sol')
# ax[2].set_ylabel('d2sol')
#
# fig, ax = plt.subplots(3, 1)
# ax[0].plot(b[N:])
# ax[1].plot(sol[N:])
# ax[2].plot(-np.diff(np.diff(sol[N:])))
# ax[0].set_title('y component')
# ax[0].set_ylabel('source')
# ax[1].set_ylabel('sol')
# ax[2].set_ylabel('d2sol')

fig, ax = plt.subplots()
m1.plot(ax)
ax.plot(x_updated[:, 0], x_updated[:, 1], 'gx-')
ax.plot(x_projected[:, 0], x_projected[:, 1], 'ro-', fillstyle='none')

plt.show()
