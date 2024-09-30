import numpy as np
from manifold import *
import matplotlib.pyplot as plt

#------------------------- line test --------------------------------#
x0 = np.array((0, 0))
x1 = np.array((2, 2))

l1 = LineChart(x0, x1)
x = np.array((1, 1.5))
x_Pl, t_Pl = l1.proximal_point(x, return_param=True)
print('line test, t_Pl = ', t_Pl)
# check that the proximal parameter is correct by recomputing the
# corresponding point
x_Pl_check = l1.ref_to_cart(t_Pl)

fig, ax = plt.subplots()
l1.plot(ax)
ax.plot(x[0], x[1], 'ok')
ax.plot(x_Pl[0], x_Pl[1], 'xk')
ax.plot(x_Pl_check[0], x_Pl_check[1], 'ok', fillstyle='none')
ax.set_aspect('equal')

#-------------------------- arc test --------------------------------#
xc = np.array((1, 1))
R = 2
alpha0 = 60*np.pi/180
alpha1 = 135*np.pi/180
a1 = ArcChart(xc, R, alpha0, alpha1)
x = np.array((0.5, 4))
x_Pa, t_Pa = a1.proximal_point(x, return_param=True)
print('arc test, t_Pa = ', t_Pa)
# check that the proximal parameter is correct by recomputing the
# corresponding point
x_Pa_check = a1.ref_to_cart(t_Pa)

fig, ax = plt.subplots()
a1.plot(ax)
ax.plot(x[0], x[1], 'ok')
ax.plot(x_Pa[0], x_Pa[1], 'xk')
ax.plot(x_Pa_check[0], x_Pa_check[1], 'ok', fillstyle='none')
ax.set_aspect('equal')


plt.show()
