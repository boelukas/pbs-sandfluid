import taichi as ti
import numpy as np
from macgrid import sMACGrid


@ti.data_oriented
class PressureSolver(object):
    def __init__(self, mac_grid:sMACGrid):
        self.mac_grid = mac_grid
        self.gaus_seidel_min_accuracy = 0.01
        self.gaus_seidel_max_iterations = 10
        self.rho = 1.0

    # Computes the divergence
    @ti.func
    def compute_divergence(self):
        # maybe needs to go from 1 to range not from 0
        for x, y, z in ti.ndrange(self.mac_grid.grid_size - 1, self.mac_grid.grid_size - 1, self.mac_grid.grid_size - 1):
            du_dx = (self.mac_grid.velX_grid[x + 1, y, z] - self.mac_grid.velX_grid[x, y, z]) /  self.mac_grid.dx
            du_dy = (self.mac_grid.velY_grid[x, y + 1, z] - self.mac_grid.velY_grid[x, y, z]) /  self.mac_grid.dx
            du_dz = (self.mac_grid.velZ_grid[x, y, z + 1] - self.mac_grid.velZ_grid[x, y, z]) /  self.mac_grid.dx

            self.mac_grid.divergence_grid[x, y, z] = du_dx + du_dy + du_dz

    # Computes pressure that will be needed to make the velocity divergence free
    @ti.func
    def compute_pressure(self, dt: ti.f32):
        self.mac_grid.clear_field(self.mac_grid.divergence_grid)

        self.compute_divergence()
        self.gauss_seidel(dt)

    # Computes the velocity projection to make the velocity divergence free
    @ti.func
    def project(self, dt):
        DENSITY = 1.0
        for x, y, z in ti.ndrange((1, self.mac_grid.grid_size), (1, self.mac_grid.grid_size - 1), (1, self.mac_grid.grid_size - 1)):
            self.mac_grid.velX_grid[x, y, z] -= dt * (1.0 / DENSITY) * (self.mac_grid.pressure_grid[x, y, z] - self.mac_grid.pressure_grid[x - 1, y, z]) / self.mac_grid.dx

        for x, y, z in ti.ndrange((1, self.mac_grid.grid_size - 1), (1, self.mac_grid.grid_size), (1, self.mac_grid.grid_size - 1)):
            self.mac_grid.velY_grid[x, y, z] -= dt * (1.0 / DENSITY) * (self.mac_grid.pressure_grid[x, y, z] - self.mac_grid.pressure_grid[x, y - 1, z]) / self.mac_grid.dx
        
        for x, y, z in ti.ndrange((1, self.mac_grid.grid_size - 1), (1, self.mac_grid.grid_size - 1), (1, self.mac_grid.grid_size)):
            self.mac_grid.velY_grid[x, y, z] -= dt * (1.0 / DENSITY) * (self.mac_grid.pressure_grid[x, y, z] - self.mac_grid.pressure_grid[x, y, z - 1]) / self.mac_grid.dx
    
    
    # The following code is taken and adapted to 3D from exercise 4_fluid.py from the PBS course
    # run Gauss-Seidel as long as max iterations has not been reached and accuracy is not good enough
    @ti.func
    def gauss_seidel(self, dt):
         # initial guess: p = 0
        self.mac_grid.clear_field(self.mac_grid.pressure_grid)

        dx2 = self.mac_grid.dx * self.mac_grid.dx
        res_x, res_y, res_z = self.mac_grid.pressure_grid.shape

        residual = self.gaus_seidel_min_accuracy + 1
        iterations = 0
        while iterations < self.gaus_seidel_max_iterations and residual > self.gaus_seidel_min_accuracy:
            residual = 0.0

            for y in range(1, res_y - 1):
                for x in range(1, res_x - 1):
                    for z in range(1, res_z - 1):
                        b = -self.mac_grid.divergence_grid[x, y, z] / dt * self.rho

                        self.mac_grid.pressure_grid[x, y, z] = (
                            dx2 * b
                            + self.mac_grid.pressure_grid[x - 1, y, z]
                            + self.mac_grid.pressure_grid[x + 1, y, z]
                            + self.mac_grid.pressure_grid[x, y - 1, z]
                            + self.mac_grid.pressure_grid[x, y + 1, z]
                            + self.mac_grid.pressure_grid[x, y, z + 1]
                            + self.mac_grid.pressure_grid[x, y, z - 1]
                        ) / 6.0

            for y in range(1, res_y - 1):
                for x in range(1, res_x - 1):
                    for z in range(1, res_z - 1):
                        b = -self.mac_grid.divergence_grid[x, y, z] / dt * self.rho

                        cell_residual = 0.0
                        cell_residual = (
                            b
                            - (
                                6.0 * self.mac_grid.pressure_grid[x, y, z]
                                - self.mac_grid.pressure_grid[x - 1, y, z]
                                - self.mac_grid.pressure_grid[x + 1, y, z]
                                - self.mac_grid.pressure_grid[x, y - 1, z]
                                - self.mac_grid.pressure_grid[x, y + 1, z]
                                - self.mac_grid.pressure_grid[x, y, z + 1]
                                - self.mac_grid.pressure_grid[x, y, z - 1]
                            )
                            / dx2
                        )

                        residual += cell_residual * cell_residual

            residual = ti.lang.ops.sqrt(residual)
            residual /= (res_x - 2) * (res_y - 2) * (res_z - 2)

            iterations += 1