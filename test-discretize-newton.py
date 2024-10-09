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
k_penalty = 1e1


# boundary conditions
bc_L = np.array((0.5, 0.))
bc_R = np.array((0., 0.))

# newton tolerance
newt_tol = 1e-6
maxit = 50

inexact = True
mass_lumping = True

plots = False

# create a case name for output files based on user choices
case_name = 'case2_' + str(int(inexact)) + str(int(mass_lumping))

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

N = x.shape[0]
sol_old = np.zeros(2*N)

#------- first Newton step: no motion, no rhs ------------#
ii = 1
print('Newton')
print('\nIteration 1')

# matrices assembly
if mass_lumping:
    K, M = m1.assemble_vector_matrices()
    M_lump = m1.assemble_lumped_mass_newton(np.zeros((N, 2)),
                                            inexact=inexact)
    A = K + k_penalty/h**2 * M_lump # system matrix with mass lumping
else:
    K, M = m1.assemble_vector_matrices()
    A = K + k_penalty/h**2 * M # system matrix



# rhs assembly
b = np.zeros(2*N)
#m1.apply_bc(A, b, bc_L, bc_R, mu=k_penalty/h**2)
m1.apply_bc(A, b, bc_L, bc_R, mu=None)

sol, info = gmres(A, b)
print('GMRES solver finished with status ', info)

vec_sol = np.zeros((N, 2))
vec_sol[:, 0] = sol[:N]
vec_sol[:, 1] = sol[N:]


err = np.linalg.norm(sol - sol_old)
print('Error = ', err)

err_vec = [err]

while (err>newt_tol and ii<maxit):

    ii += 1
    print('\nIteration ', ii)
    sol_old = sol.copy()

    if mass_lumping:
        M_lump = m1.assemble_lumped_mass_newton(vec_sol,
                                                inexact=inexact)
        b = k_penalty/h**2 * (m1.assemble_newton_rhs1(u=vec_sol, method='trapz')
                                + M_lump@sol_old)
        A = K + k_penalty/h**2 * M_lump
    else:
        K, M = m1.assemble_vector_matrices(newton=True, u=vec_sol, inexact=inexact)
        b = k_penalty/h**2 * (m1.assemble_newton_rhs1(u=vec_sol, method='midpoint')
                                + M@sol_old)
        A = K + k_penalty/h**2 * M

    # m1.apply_bc(A, b, bc_L, bc_R, mu=k_penalty/h**2)
    m1.apply_bc(A, b, bc_L, bc_R, mu=None)

    sol, info = gmres(A, b)
    print('GMRES solver finished with status ', info)

    vec_sol = np.zeros((N, 2))
    vec_sol[:, 0] = sol[:N]
    vec_sol[:, 1] = sol[N:]

    err = np.linalg.norm(sol - sol_old)
    err_vec.append(err)
    print('Error = ', err)

# x_projected = np.zeros((N, 2))
# for ii in range(N):
#     x_projected[ii, :] = m1.proximal_point(x_updated[ii, :])
# print('points on each chart in final grid')
# print(m1.point_distribution_on_charts(x_projected))

x_updated = x + vec_sol

np.savez(case_name+'.npz', errvec=err_vec)

if plots:

    fig, ax = plt.subplots()
    m1.plot(ax)
    m1.plot_discretization(ax)

    fig, ax = plt.subplots()
    m1.plot(ax)
    ax.plot(x_updated[:, 0], x_updated[:, 1], 'gx-')

    # detail to show difference between lumping and midpoint
    fig, ax = plt.subplots()
    m1.plot(ax)
    ax.plot(x_updated[:, 0], x_updated[:, 1], 'gx-')
    ax.set_xlim([-1, -0.6])
    ax.set_ylim([-0.05, 0.4])

    fig, ax = plt.subplots()
    ax.semilogy(err_vec)

    plt.show()
