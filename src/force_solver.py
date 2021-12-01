import taichi as ti
import numpy as np
from macgrid import MacGrid


@ti.data_oriented
class ForceSolver(object):
    def __init__(self, mac_grid:MacGrid):
        self.mac_grid = mac_grid
        self.gravity = 9.81

    @ti.kernel
    def compute_forces(self):
        self.mac_grid.clear_field(self.mac_grid.f_y)

        for x, y, z in self.mac_grid.f_y:
             self.mac_grid.f_y[x, y, z] -=  9.81 

    
    @ti.kernel
    def apply_forces(self, dt: ti.f32):
        for x, y, z in self.mac_grid.v_y:
            self.mac_grid.v_y[x, y, z] += dt * self.mac_grid.f_y[x, y, z]
