from typing import Tuple
import taichi as ti
import numpy as np


STEP_SIZE = 1.

#Solve Runge-Kutta ODE of second order
def runge_kutta(
    pos: Tuple[float,float,float],
    vel: Tuple[float,float,float],
    steps: np.array() = np.linspace(1/STEP_SIZE, STEP_SIZE, 5)
    ) -> Tuple[float,float,float]:
    return (0,0,0)

### UNDER CONSTRUCTION ###

def dydx(x: Tuple[float,float,float], y:Tuple[float,float,float]) :
    pos_derivative = lambda x, y : x+y/2.
    return tuple(map(pos_derivative, zip(x, y)))

# Finds value of y for a given x
# using step size h
# and initial value y0 at x0.
def rungeKutta2(x0, y0, x, h) :

    # Count number of iterations
    # using step size or
    # step height h
    n = round((x - x0) / h)
    
        # Iterate for number of iterations
    y = y0
    
    for i in range(1, n + 1) :
        
                # Apply Runge Kutta Formulas
        # to find next value of y
        k1 = h * dydx(x0, y)
        k2 = h * dydx(x0 + 0.5 * h, y + 0.5 * k1)

        # Update next value of y
        y = y + (1.0 / 6.0) * (k1 + 2 * k2)

        # Update next value of x
        x0 = x0 + h

    return y
