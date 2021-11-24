import taichi as ti
import numpy as np
from macgrid import sMACGrid


@ti.data_oriented
class ForceSolver(object):
    def __init__(self, mac_grid:sMACGrid):
        self.mac_grid = mac_grid
        self.gravity = 9.81
        self.m = 0.1

    @ti.func
    def compute_forces(self):
        self.mac_grid.clear_field(self.mac_grid.forceX_grid)
        self.mac_grid.clear_field(self.mac_grid.forceY_grid)
        self.mac_grid.clear_field(self.mac_grid.forceZ_grid)

        for x, y, z in ti.ndrange(self.mac_grid.grid_size, self.mac_grid.grid_size, self.mac_grid.grid_size):
             self.mac_grid.forceY_grid[x, y, z] += self.m * 9.81

    
    @ti.func
    def apply_forces(self, dt: ti.f32):
        for x, y, z in ti.ndrange(self.mac_grid.grid_size, self.mac_grid.grid_size, self.mac_grid.grid_size):
            self.mac_grid.velX_grid[x, y, z] += dt * self.mac_grid.forceX_grid[x, y, z]
            self.mac_grid.velY_grid[x, y, z] += dt * self.mac_grid.forceY_grid[x, y, z]
            self.mac_grid.velZ_grid[x, y, z] += dt * self.mac_grid.forceZ_grid[x, y, z]