import numpy as np
import matplotlib.pyplot as plt

def dy(x, y, nu=0):
  y, dydx = y[0], y[1]
  d2ydx2 = (-x * dydx - (x**2 - nu**2)*y)/x**2
  return dydx, d2ydx2

xmin, xmax = 1e-15, 10
x = np.arange(xmin, xmax, 0.1)
from scipy.integrate import solve_ivp
sol = solve_ivp(dy, t_span=[xmin, xmax], y0=[1, 0], t_eval=x)

grid = plt.GridSpec(4, 1, hspace=0)
main = plt.subplot(grid[0:3], xticklabels=[])
from scipy.special import jn, jvp
main.plot(sol.t, sol.y[0], ".k", label="numérique")
main.plot(x, jn(0, x), "-r", label="analytique")
main.set_ylabel(r"$J_0(x)$")
main.legend()
dev = plt.subplot(grid[-1])
dev.plot(sol.t, jn(0, sol.t) - sol.y[0], ".k")
dev.set_xlabel(r"$x$")
dev.set_ylabel(r"$\Delta J_0(x)$");
