import numpy as np
import matplotlib.pyplot as plt
from manifold import *
from matplotlib.backend_bases import MouseButton

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

x = np.array((0.8, -0.5))
xp, (chart_index, chart_tp) = m1.proximal_point(x, return_chart_and_param=True)

click_on = False
fig, ax = plt.subplots()
m1.plot(ax)
# ax.plot(x[0], x[1], 'o')
# ax.plot(xp[0], xp[1], 'x')
ax.set_aspect('equal')

def on_click(event):
    if (event.button is MouseButton.LEFT) and event.inaxes:
        x = np.array((event.xdata, event.ydata))
        xp, (chart_index, chart_tp) = m1.proximal_point(x, return_chart_and_param=True)
        xp_test = charts[chart_index].ref_to_cart(chart_tp)
        normal = m1.normal_vector(x)
        kn = m1.curvature(x)
        if len(ax.lines)>len(charts):
            #4 items are added, 4 must be removed
            for ii in range(4):
                ax.lines[-1].remove()
        ax.plot(x[0], x[1], 'ok')
        ax.plot(xp[0], xp[1], 'xk')
        ax.plot(xp_test[0], xp_test[1], 'ok', fillstyle='none')
        #ax.plot([xp[0], xp[0]+normal[0]], [xp[1], xp[1]+normal[1]], 'k')
        ax.plot([xp[0], xp[0]+kn[0]], [xp[1], xp[1]+kn[1]], 'k')


plt.connect('button_press_event', on_click)

plt.show()
