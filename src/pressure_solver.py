import taichi as ti
import numpy as np
from macgrid import sMACGrid


@ti.data_oriented
class PressureSolver(object):
    def __init__(self, mac_grid:sMACGrid):
        self.mac_grid = mac_grid

    @ti.func
    def compute_divergence(self):
        for x, y, z in ti.ndrange(self.mac_grid.grid_size, self.mac_grid.grid_size, self.mac_grid.grid_size):
            self.mac_grid.divergence[x, y, z] = 1

    @ti.func
    def compute_pressure(self):
        self.compute_divergence()
