import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod
from functools import cached_property, lru_cache
from scipy.sparse import coo_matrix
from numpy.random import default_rng


class Chart(object):
    """A local chart defining a portion of a manifold"""

    def __init__(self, param_range):
        self.param_range = param_range

    @abstractmethod
    def proximal_point(self, x, return_param=False):
        raise NotImplementedError

    @abstractmethod
    @cached_property
    def length(self, param_interval=None):
        raise NotImplementedError

    @abstractmethod
    def ref_to_cart(self, t):
        raise NotImplementedError

    @abstractmethod
    def d1(self, t):
        """First derivative (unscaled tangent vector)"""
        raise NotImplementedError

    @abstractmethod
    def d2(self, t):
        """Second derivative"""
        raise NotImplementedError

    @abstractmethod
    def curvature(self, t):
        """Returns the curvature vector kn (see Walker sec. 3.2.4)"""
        raise NotImplementedError

    def normal_vector(self, t):
        """Scaled normal vector, to the left of the path"""
        tangent_vector = self.d1(t)
        normal_vector = np.array((-tangent_vector[1], tangent_vector[0]))
        return normal_vector / np.linalg.norm(normal_vector)

    def normal_jac(self, t):
        """Jacobian of the normal vector"""
        tvec = self.d1(t)
        contrav_tangent_vector = tvec/np.dot(tvec, tvec)
        dottvec = self.d2(t)
        n = self.normal_vector(t)
        II = np.dot(n, dottvec) # second fundamental form (scalar for a curve)
        return -II * np.outer(contrav_tangent_vector, contrav_tangent_vector)

    def prox_jac(self, x):
        """Jacobian of the proximal operator"""
        y, t = self.proximal_point(x, return_param=True)
        n = self.normal_vector(t)
        d = np.dot(x - y, n) #distance
        Jn = self.normal_jac(t)
        proj_mat = np.eye(2) - np.outer(n, n)
        dist_mat = np.eye(2) + d*Jn #matrix depending on distance
        return np.linalg.solve(dist_mat, proj_mat)


    def plot(self, ax, n=100):
        t0, t1 = self.param_range
        t = np.linspace(t0, t1, n)
        x = np.zeros(n)
        y = np.zeros(n)
        for ii, tt in enumerate(t):
            xt = self.ref_to_cart(tt)
            x[ii] = xt[0]
            y[ii] = xt[1]
        ax.plot(x, y)

    def discretize(self, h):
        """Create a mesh with target size h"""
        t0, t1 = self.param_range
        N = int(np.ceil(self.length/h)) + 1
        x = np.zeros((N, 2))
        t = np.linspace(t0, t1, N)
        for ii, tt in enumerate(t):
            x[ii, :] = self.ref_to_cart(tt)
        self.grid_coordinates = x
        return x

    def plot_discretization(self, ax):
        x = self.grid_coordinates
        ax.plot(x[:,0], x[:,1], 'ko-')


class LineChart(Chart):
    """Local chart of a line segment"""

    def __init__(self, x0, x1, param_range=(0,1)):
        """
        x0: array of shape (2)
            initial point
        x1: array of shape (2)
            final point
        param_range: tuple (2)
            line parameter corresponding to x0 and x1, respectively
            default: (0,1)
        """
        self.param_range = param_range
        self.x0 = x0
        self.x1 = x1

    def ref_to_cart(self, t):
        """Maps from the reference interval to the cartesian coordinates"""
        t0, t1 = self.param_range
        x0 = self.x0
        x1 = self.x1

        if t>=t0 and t<=t1:
            return x0 + (t-t0)/(t1-t0) * (x1-x0)
        else:
            raise ValueError('Parameter out of interval!')

    def proximal_point(self, x, return_param=False):
        """
        Finds the point on the manifold closest to x.
        With return_param=True, also returns the associated parameter.
        """
        t0, t1 = self.param_range
        x0 = self.x0
        x1 = self.x1
        t_prox = t0 - (t1-t0) * np.dot(x0-x, x1-x0) / self.length**2
        if t_prox>t0 and t_prox<t1:
            # the projection is inside the segment
            # so it coincides with the proximal point
            if return_param:
                return self.ref_to_cart(t_prox), t_prox
            else:
                return self.ref_to_cart(t_prox)
        else:
            # the projection is outside the segment
            # the proximal point must be one of the endpoints
            d0 = np.linalg.norm(x - x0)
            d1 = np.linalg.norm(x - x1)
            if d0<d1:
                if return_param:
                    return x0, t0
                else:
                    return x0
            else:
                if return_param:
                    return x1, t1
                else:
                    return x1

    @cached_property
    def length(self):
        return np.linalg.norm(self.x1 - self.x0)

    def d1(self, t):
        """Tangent vector"""
        t0, t1 = self.param_range
        if t>=t0 and t<=t1:
            x0, x1 = self.x0, self.x1
            return 1/(t1 -t0) * (x1 - x0)
        else:
            raise ValueError('Parameter out of interval!')

    def d2(self, t):
        t0, t1 = self.param_range
        if t>=t0 and t<=t1:
            return np.array((0, 0))
        else:
            raise ValueError('Parameter out of interval!')

    @lru_cache(maxsize=3)
    def curvature(self, t):
        return np.array((0, 0))


class ArcChart(Chart):
    """Local line chart of a circular arc"""

    def __init__(self, xc, R, alpha0, alpha1, param_range=(0,1)):
        """
        xc: array of shape (2)
            center of the arc
        R: float
            radius
        alpha0: float
            angle of initial point (rad)
        alpha1: float
            angle of final point (rad)
        Angles must be unwrapped
        """
        self.param_range = param_range
        self.xc = xc
        self.R = R
        self.alpha0 = alpha0
        self.alpha1 = alpha1

        self.x0 = xc + R * np.array((np.cos(alpha0), np.sin(alpha0)))
        self.x1 = xc + R * np.array((np.cos(alpha1), np.sin(alpha1)))

    def ref_to_angle(self, t):
        t0, t1 = self.param_range
        alpha0 = self.alpha0
        alpha1 = self.alpha1

        return alpha0 + (t-t0)/(t1-t0) * (alpha1-alpha0)

    def ref_to_cart(self, t):
        alpha = self.ref_to_angle(t)
        t0, t1 = self.param_range
        if t>=t0 and t<=t1:
            return self.xc + self.R * np.array((np.cos(alpha), np.sin(alpha)))
        else:
            raise ValueError('Parameter out of interval!')


    def proximal_point(self, x, return_param=False):
        """
        Finds the point on the manifold closest to x.
        With return_param=True, also returns the associated parameter.
        """
        # define a separating line
        # depending on the side of the line, the proximal point may be
        # inside the arc, or an endpoint
        x0 = self.x0
        x1 = self.x1
        xc = self.xc
        R = self.R
        alpha0 = self.alpha0
        alpha1 = self.alpha1
        t0, t1 = self.param_range

        # projection on the full circle
        xp_circle = xc + R*(x-xc)/np.linalg.norm(x-xc)
        t = x1 - x0
        d = x0 - xp_circle
        # the point is inside the arc if it is to the right of the
        # line, if clockwise, or to the left, if counterclockwise
        side_indicator = (t[0]*d[1] - t[1]*d[0]) * (alpha1 - alpha0)

        if side_indicator>0:
            if return_param:
                alpha = np.arctan2(x[1] - xc[1], x[0] - xc[0])
                alphamin = min(alpha0, alpha1)
                alphamax = max(alpha0, alpha1)
                # put alpha inside the interval by an integer number of full rotations
                while alpha<alphamin:
                    alpha += 2*np.pi
                while alpha>alphamax:
                    alpha -= 2*np.pi
                param = t0 + (alpha - alpha0)/(alpha1 - alpha0) * (t1 - t0)
                return xp_circle, param
            else:
                return xp_circle
        else:
            d0 = np.linalg.norm(x - x0)
            d1 = np.linalg.norm(x - x1)
            if d0<d1:
                if return_param:
                    return x0, t0
                else:
                    return x0
            else:
                if return_param:
                    return x1, t1
                else:
                    return x1

    @cached_property
    def length(self):
        return np.abs(self.R * (self.alpha1 - self.alpha0))

    def d1(self, t):
        """Tangent vector"""
        t0, t1 = self.param_range
        alpha0, alpha1 = self.alpha0, self.alpha1
        if t>=t0 and t<=t1:
            alpha = self.ref_to_angle(t)
            return self.R * (alpha1 - alpha0)/(t1 - t0) * (
                        np.array((-np.sin(alpha), np.cos(alpha))))
        else:
            raise ValueError('Parameter out of interval!')

    def d2(self, t):
        t0, t1 = self.param_range
        alpha0, alpha1 = self.alpha0, self.alpha1
        if t>=t0 and t<=t1:
            alpha = self.ref_to_angle(t)
            return -self.R * ((alpha1 - alpha0)/(t1 - t0))**2 * (
                        np.array((np.cos(alpha), np.sin(alpha))))
        else:
            raise ValueError('Parameter out of interval!')

    @lru_cache(maxsize=3)
    def curvature(self, t):
        dc = self.ref_to_cart(t) - self.xc # distance vector from center
        # its norm is R and the curvature is 1/R, so we get:
        return 1/self.R**2 * dc

class Manifold(object):
    def __init__(self, charts):
        """
        charts: a list of objects of type Chart (or subtypes)
        """
        self.charts = charts

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        for chart in self.charts:
            chart.plot(ax)
        ax.set_aspect('equal')

    def proximal_point(self, x, return_chart_and_param=False):
        """
        Finds the point on the manifold closest to x.
        With return_chart_and_param=True, also returns the index of the
        associated chart and the corresponding parameter.
        """
        charts = self.charts
        if return_chart_and_param:
            xp, tp = charts[0].proximal_point(x, return_param=True)
            chart_tp = tp
            chart_index = 0
        else:
            xp = charts[0].proximal_point(x)
        d = np.linalg.norm(x - xp)
        for ii in range(1, len(charts)):
            if return_chart_and_param:
                this_xp, this_tp = charts[ii].proximal_point(x, return_param=True)
            else:
                this_xp = charts[ii].proximal_point(x)
            this_d = np.linalg.norm(x - this_xp)
            if this_d<d:
                xp = this_xp
                d = this_d
                if return_chart_and_param:
                    chart_tp = this_tp
                    chart_index = ii
        if return_chart_and_param:
            return xp, (chart_index, chart_tp)
        else:
            return xp

    def point_manifold_distance(self, x):
        """
        x: array of shape (N, 2)
            coordinates of N points

        Returns the vector of distance of all points from the manifold
        """
        N = x.shape[0]
        d = np.zeros(N)
        for ii in range(N):
            d[ii] = np.linalg.norm(self.proximal_point(x[ii, :]) - x[ii, :])
        return d

    def discretize_by_glueing(self, h, tol=1e-12, perturb=0):
        charts = self.charts
        x = charts[0].discretize(h)
        for ii, chart in enumerate(charts[1:]):
            this_x = chart.discretize(h)
            #pdb.set_trace()
            gap = np.linalg.norm(this_x[0,:] - x[-1,:])
            if gap<tol:
                x = np.append(x, this_x[1:,:], axis=0)
            else:
                raise ValueError('the grids to be glued do not match')
        if perturb>0:
            # add a random perturbation to the mesh and project
            # so that nodes are not on chart boundaries
            rng = default_rng()
            displ = perturb*rng.uniform(-h, h, size=x.shape)
            x_perturbed = x + displ
            for ii in range(x.shape[0]):
                x[ii, :] = self.proximal_point(x_perturbed[ii, :])
        self.grid_coordinates = x
        return x

    def point_distribution_on_charts(self, x):
        """Computes the number of points on each chart"""
        points_on_chart = np.zeros(len(self.charts), dtype=int)
        for ii in range(x.shape[0]):
            _, (chart_index, _) = self.proximal_point(x[ii,:],
                        return_chart_and_param=True)
            points_on_chart[chart_index] += 1
        return points_on_chart

    def plot_discretization(self, ax):
        x = self.grid_coordinates
        ax.plot(x[:,0], x[:,1], 'ko-')

    def assemble_scalar_stiffness_matrix(self):
        x = self.grid_coordinates
        N = x.shape[0] - 1
        # coo format without worrying about duplicate entries!
        # see the scipy documentation
        data = []
        row = []
        col = []
        for ii in range(N):
            h = np.linalg.norm(x[ii+1] - x[ii])
            # local stiffness matrix
            K_loc = 1/h * np.array(((1, -1), (-1, 1)))
            for jj in range(2):
                for kk in range(2):
                    data.append(K_loc[jj, kk])
                    row.append(ii+jj)
                    col.append(ii+kk)
        K = coo_matrix((data, (row, col)), shape=(N+1, N+1))
        return K

    def normal_vector(self, x):
        """Returns the normal vector in the point closest to x of the manifold"""
        _, (chart_index, chart_tp) = self.proximal_point(x,
                                        return_chart_and_param=True)
        return self.charts[chart_index].normal_vector(chart_tp)

    def curvature(self, x):
        """Returns vector kn in the point closest to x of the manifold"""
        _, (chart_index, chart_tp) = self.proximal_point(x,
                                        return_chart_and_param=True)
        return self.charts[chart_index].curvature(chart_tp)

    def prox_jac(self, x):
        _, (chart_index, chart_tp) = self.proximal_point(x,
                                        return_chart_and_param=True)
        return self.charts[chart_index].prox_jac(x)

    def assemble_vector_matrices(self, newton=False, u=None, inexact=True):
        """
        Assembles stiffness and mass matrices
        If newton is True (default: False), the additional argoment u
        is needed (default: None), in the form of an array of shape (N, 2)
        which contains the displacements of the grid points in the current
        configuration with respect to the reference positions. The normals
        used to assemble the mass matrix are then evaluated at the current
        position instead of the reference position.
        inexact has effect only with Newton
        (the extension-projection method is always 'exact')
        """
        x = self.grid_coordinates
        if newton:
            x_current = x + u
        N = x.shape[0] - 1
        # coo format without worrying about duplicate entries!
        # see the scipy documentation
        Kdata = [] # stiffness matrix
        Mdata = [] # mass matrix
        row = []
        col = []
        for ii in range(N): #loop on elements
            # discrete tangent vector, used to build the discrete normal
            tangent_vector = x[ii+1] - x[ii]
            h = np.linalg.norm(tangent_vector)
            if newton:
                # use the normal to the true surface evaluated at the current midpoint
                x_current_midpoint = 0.5 * (x_current[ii] + x_current[ii+1])
                normal_vector = self.normal_vector(x_current_midpoint)
                normal_vector = normal_vector / np.linalg.norm(normal_vector)
                if inexact:
                    R = np.outer(normal_vector, normal_vector)
                else:
                    R = np.eye(2) - self.prox_jac(x_current_midpoint)
                # local mass matrix (inexact, midpoint quadrature)
                M_loc = h * np.array(((1/4, 1/4), (1/4, 1/4)))

            else:
                normal_vector = np.array((-tangent_vector[1],
                                       tangent_vector[0]))
                normal_vector = normal_vector / np.linalg.norm(normal_vector)
                # rejection matrix
                R = np.outer(normal_vector, normal_vector)
                # local mass matrix (exact)
                M_loc = h * np.array(((1/3, 1/6), (1/6, 1/3)))
            I = np.eye(2)
            # local stiffness matrix
            K_loc = 1/h * np.array(((1, -1), (-1, 1)))
            for jj in range(2): # first local index
                for kk in range(2): # second local index
                    for rr in range(2): # first global block-index
                        for ss in range(2): # second global block-index
                            # This is embarassing but clear, so I don't care!
                            Kdata.append(I[rr, ss] * K_loc[jj, kk])
                            Mdata.append(R[rr, ss] * M_loc[jj, kk])
                            row.append(ii + jj + rr*(N+1))
                            col.append(ii + kk + ss*(N+1))
        K = coo_matrix((Kdata, (row, col)), shape=(2*(N+1), 2*(N+1)))
        M = coo_matrix((Mdata, (row, col)), shape=(2*(N+1), 2*(N+1)))
        return K, M

    def assemble_lumped_mass_newton(self, u, inexact=True):
        """
        Assembles a lumped mass matrix to be used for the Newton method,
        which corresponds to requiring the grid nodes to lie on the
        curve, instead of the midpoints
        u: array of shape (N, 2)
            displacements of the grid points with respect to the reference positions
        inexact: bool
            whether to use the projection matrix instead of the full
            Jacobian of the proximal map
        """
        x = self.grid_coordinates
        x_current = x + u
        N = x.shape[0] - 1
        Mdata = []
        row = []
        col = []

        # compute the first rejection matrix
        if inexact:
            normal_vector = self.normal_vector(x_current[0,:])
            normal_vector = normal_vector / np.linalg.norm(normal_vector)
            R_old = np.outer(normal_vector, normal_vector)
        else:
            R_old = np.eye(2) - self.prox_jac(x_current[0,:])

        for ii in range(N): # loop on elements
            tangent_vector = x[ii+1] - x[ii]
            h = np.linalg.norm(tangent_vector)
            # compute normal vector for second node
            # (first node already done at previous iteration)
            if inexact:
                normal_vector = self.normal_vector(x_current[ii+1,:])
                normal_vector = normal_vector / np.linalg.norm(normal_vector)
                R = np.outer(normal_vector, normal_vector)
            else:
                R = np.eye(2) - self.prox_jac(x_current[ii+1,:])


            for rr in range(2): # first global block-index
                for ss in range(2): # second global block-index
                    # first node of the element
                    Mdata.append(0.5 * h * R_old[rr, ss])
                    row.append(ii + rr*(N+1))
                    col.append(ii + ss*(N+1))

                    # second node of the element
                    Mdata.append(0.5 * h * R[rr, ss])
                    row.append(ii + 1 + rr*(N+1))
                    col.append(ii + 1 + ss*(N+1))

            R_old = R.copy()

        M = coo_matrix((Mdata, (row, col)), shape=(2*(N+1), 2*(N+1)))
        return M

    def assemble_newton_rhs1(self, u, method='trapz'):
        """
        Assemble the first contribution to the Newton rhs vector
        It is the one with the proximal operator
        The other contribution contains the Jacobian and is known from
        the previous Newton step
        """
        x = self.grid_coordinates
        x_current = x + u
        N = x.shape[0]-1
        b = np.zeros(2*(N+1))

        if method=='trapz':
            for ii in range(N):
                h = np.linalg.norm(x[ii+1, :] - x[ii, :])
                for jj in range(2):
                    x_c_rej = self.proximal_point(x_current[ii+jj]) - x_current[ii+jj]
                    for rr in range(2):
                        b[ii + jj + rr*(N+1)] += 0.5 * h * x_c_rej[rr]
        elif method=='midpoint':
            for ii in range(N):
                h = np.linalg.norm(x[ii+1, :] - x[ii, :])
                x_current_midpoint = 0.5 * (x_current[ii] + x_current[ii+1])
                x_c_mp_proj = self.proximal_point(x_current_midpoint)
                x_c_mp_rej = x_c_mp_proj - x_current_midpoint # rejection
                for jj in range(2): # loop on element dofs
                    for rr in range(2): # loop on geometric dimensions
                        b[ii + jj + rr*(N+1)] += 0.5 * h * x_c_mp_rej[rr]

        return b

    def assemble_vector_rhs(self, method='midpoint'):
        """
        Assemble rhs vector (see Mola-Heltai-De Simone 2013, eq. 37).
        Uses trapezoidal integration (nodal values).
        """
        x = self.grid_coordinates
        N = x.shape[0]-1
        b = np.zeros(2*(N+1))

        if method=='trapz':
            for ii in range(N): #loop on elements
                h = np.linalg.norm(x[ii+1,:] - x[ii,:])
                for jj in range(2): # local index
                    kn = self.curvature(x[ii+jj, :])
                    for rr in range(2): # global index
                        b[ii + jj + rr*(N+1)] += h*kn[rr]
        elif method=='midpoint':
            for ii in range(N):
                h = np.linalg.norm(x[ii+1,:] - x[ii,:])
                x_mp = 0.5 * (x[ii,:] + x[ii+1,:])
                kn = self.curvature(x_mp)
                for jj in range(2):
                    for rr in range(2):
                        b[ii + jj + rr*(N+1)] += h*kn[rr]
        return b

    def apply_bc(self, A, b, bc_L, bc_R, mu=None):
        """
        If mu is provided, it is used to set the diagonal element
        Otherwise, the diagonal element is kept unchanged
        """
        x = self.grid_coordinates
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
                    if mu is not None:
                        A.data[col_ind] = mu
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
