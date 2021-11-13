from typing import Tuple
import taichi as ti
import numpy as np
from macgrid import sMACGrid

from taichi.examples.simulation.vortex_rings import T


STEP_SIZE = 1.

#Solve Runge-Kutta ODE of second order
def runge_kutta_2(
    pos: Tuple[float,float,float],
    vel_field: np.array(Tuple[float, float, float]),
    t: np.array() = np.linspace(1/STEP_SIZE, STEP_SIZE, 5)

    ) -> Tuple[float,float,float]:

    n = len(t)
    x = np.array([pos] * n)
    for i in range(n-1):
        h = t[i+1]- t[i] # 1/5 in our case
        # should use velocity of point x[i] sampled from velocity field
        # something like:
        k1 = h * dydx(x[i], t[i], vel_field)
        x[i+1] = x[i] + h * dydx(x[i] + k1, t[i] + h/2.0, vel_field)


    return x[n-1]



def dydx(x: Tuple[float,float,float], t: float, vel_field: np.array(Tuple[float, float, float])) -> Tuple[float,float,float]:
    # computes velocity at point x at time t given a velocity field

    # use Euler Step Method (implicit/explicit/midpoint) to solve first order ODE
    
    vel_x = vel_field.sample_velocity(x)
        ### UNDER CONSTRUCTION ###

    return tuple(0., 0., 0.)
    
# https://stackoverflow.com/questions/35258628/what-will-be-python-code-for-runge-kutta-second-method
