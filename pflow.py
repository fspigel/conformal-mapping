import sympy as sp
import numpy as np
from sympy.utilities.lambdify import lambdify
from scipy.optimize import newton
import matplotlib.pyplot as plt

class Pflow:
    def __init__(self, phi, psi):
        self.X = None
        self.Y = None
        self.U = None
        self.phi = phi
        self.psi = psi

    @classmethod
    def from_txt(cls, phi_str, psi_str):
        return cls(sp.sympify(phi_str), sp.sympify(psi_str))

    def get_vector_field(self, xlim, ylim, n_x, n_y):
        x, y = sp.symbols('x y')

        gradPhi = [[], []]
        gradPhi[0] = sp.Derivative(self.phi, x).doit()
        gradPhi[1] = sp.Derivative(self.phi, y).doit()

        Func = lambdify((x, y), gradPhi, 'numpy')
        self.X = np.linspace(xlim[0], xlim[1], n_x)
        self.Y = np.linspace(ylim[0], ylim[1], n_y)
        self.U = np.zeros((n_x, n_y, 2))
        for j, X_current in enumerate(self.X):
            for i, Y_current in enumerate(self.Y):
                if np.linalg.norm([X_current, Y_current]) <= 1:
                    self.U[i, j] = [0, 0]
                    continue
                self.U[i, j] = Func(X_current, Y_current)

    def get_plottable(self):
        R = np.zeros(self.U.shape[:2])
        V = np.zeros(self.U.shape)
        for i in range(self.U.shape[0]):
            for j in range(self.U.shape[1]):
                norm = np.linalg.norm(self.U[i,j,:])
                if norm == 0: V[i,j] = [0, 0]
                else: V[i,j] = self.U[i,j]/norm
                R[i,j] = norm
        return R, V

    def get_streamline(self, point, xlim):
        x, y = sp.symbols('x y')
        psi_0 = self.psi.subs([(x, point[0]), (y, point[1])])
        if abs(psi_0) < 1e-6:
            raise ValueError('Invalid Streamline')
        x_spread = np.linspace(xlim[0], xlim[1], 100)
        y_spread = np.zeros(x_spread.shape[0])
        psi_lambda = lambdify((x, y), self.psi - psi_0, 'numpy')
        res = newton(lambda y_current: psi_lambda(x_spread[0], y_current), 0)
        for i, x_current in enumerate(x_spread):
            res, r = newton(func=lambda y_current: psi_lambda(x_current, y_current),
                            x0=res,
                            full_output=True)
            y_spread[i] = res
        # return x_spread, y_spread
        return np.array([x_spread, y_spread])

    def plot_all_streamlines(self, x_domain, y_domain, axis):
        alpha = np.linspace(0, 2 * np.pi, 100)
        a = x_domain[1] - x_domain[0]
        b = y_domain[1] - y_domain[0]
        r = 1.1 * np.sqrt((a ** 2 + b ** 2)) / 2
        x_circle = r * np.cos(alpha)
        y_circle = r * np.sin(alpha)
        accumulator = []

        streamlines = [get_streamline(point, x_domain) for point in zip(x_circle, y_circle)]
        for streamline in streamlines:
            axis.plot(streamline[0,:], streamline[1,:], 'C0')
        return streamlines
        # for point in zip(x_circle, y_circle):
        #     try:
        #         x_sol, y_sol = self.get_streamline(point, x_domain)
        #     except ValueError:
        #         continue
        #     # axis.plot(x_sol, y_sol, 'C0')
        #     accumulator.append([x_sol, y_sol])
        # # axis.set_xlim(x_domain)
        # # axis.set_ylim(y_domain)
        # return accumulator