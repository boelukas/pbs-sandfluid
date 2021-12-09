import taichi as ti
import numpy as np
from macgrid import MacGrid, CellType


@ti.data_oriented
class PressureSolver(object):
    def __init__(self, mac_grid: MacGrid):
        self.name = "Custom solver"
        self.mac_grid = mac_grid
        self.gaus_seidel_min_accuracy = 0.0001
        self.gaus_seidel_max_iterations = 10000
        self.rho = 1.0
        self.density = 1.0

    # Computes the divergence
    @ti.kernel
    def compute_divergence(self):
        for x, y, z in self.mac_grid.divergence:
            if self.mac_grid.cell_type[x, y, z] == CellType.SAND.value:
                du_dx = self.mac_grid.v_x[x + 1, y, z] - self.mac_grid.v_x[x, y, z]
                du_dy = self.mac_grid.v_y[x, y + 1, z] - self.mac_grid.v_y[x, y, z]
                du_dz = self.mac_grid.v_z[x, y, z + 1] - self.mac_grid.v_z[x, y, z]
                self.mac_grid.divergence[x, y, z] = du_dx + du_dy + du_dz

    # Computes pressure that will be needed to make the velocity divergence free
    def compute_pressure(self, dt: ti.f32):
        # self.mac_grid.clear_field(self.mac_grid.divergence)
        self.mac_grid.divergence.fill(0.0)
        self.compute_divergence()  # check

        self.mac_grid.pressure.fill(0.0)

        # self.mac_grid.clear_field(self.mac_grid.pressure)
        self.gauss_seidel(dt)
        # self.reference_ps.solve(self.mac_grid.pressure, self.mac_grid.divergence)

    # Computes the velocity projection to make the velocity divergence free
    @ti.kernel
    def project(self, dt: ti.f32):
        for x, y, z in self.mac_grid.v_x:
            if x > 0 and (
                self.mac_grid.cell_type[x, y, z] == CellType.SAND.value
                or self.mac_grid.cell_type[x - 1, y, z] == CellType.SAND.value
            ):
                self.mac_grid.v_x[x, y, z] -= (
                    (
                        self.mac_grid.pressure[x, y, z]
                        - self.mac_grid.pressure[x - 1, y, z]
                    )
                    * (1.0 / self.density)
                    * dt
                )

        for x, y, z in self.mac_grid.v_y:
            if y > 0 and (
                self.mac_grid.cell_type[x, y, z] == CellType.SAND.value
                or self.mac_grid.cell_type[x, y - 1, z] == CellType.SAND.value
            ):
                self.mac_grid.v_y[x, y, z] -= (
                    (
                        self.mac_grid.pressure[x, y, z]
                        - self.mac_grid.pressure[x, y - 1, z]
                    )
                    * (1.0 / self.density)
                    * dt
                )

        for x, y, z in self.mac_grid.v_z:
            if z > 0 and (
                self.mac_grid.cell_type[x, y, z] == CellType.SAND.value
                or self.mac_grid.cell_type[x, y, z - 1] == CellType.SAND.value
            ):
                self.mac_grid.v_z[x, y, z] -= (
                    (
                        self.mac_grid.pressure[x, y, z]
                        - self.mac_grid.pressure[x, y, z - 1]
                    )
                    * (1.0 / self.density)
                    * dt
                )

    # The following code is taken and adapted to 3D from exercise 4_fluid.py from the PBS course
    # run Gauss-Seidel as long as max iterations has not been reached and accuracy is not good enough
    @ti.kernel
    def gauss_seidel(self, dt: ti.f32):
        # initial guess: p = 0
        # self.mac_grid.clear_field(self.mac_grid.pressure)
        res_x, res_y, res_z = self.mac_grid.pressure.shape

        residual = self.gaus_seidel_min_accuracy + 1
        iterations = 0
        while (
            iterations < self.gaus_seidel_max_iterations
            and residual > self.gaus_seidel_min_accuracy
        ):
            residual = 0.0

            for y in range(1, res_y - 1):
                for x in range(1, res_x - 1):
                    for z in range(1, res_z - 1):
                        if self.mac_grid.cell_type[x, y, z] == CellType.SAND.value:

                            b = -self.mac_grid.divergence[x, y, z] / dt * self.rho

                            self.mac_grid.pressure[x, y, z] = (
                                b
                                + self.mac_grid.pressure[x - 1, y, z]
                                + self.mac_grid.pressure[x + 1, y, z]
                                + self.mac_grid.pressure[x, y - 1, z]
                                + self.mac_grid.pressure[x, y + 1, z]
                                + self.mac_grid.pressure[x, y, z + 1]
                                + self.mac_grid.pressure[x, y, z - 1]
                            ) / 6.0

            for y in range(1, res_y - 1):
                for x in range(1, res_x - 1):
                    for z in range(1, res_z - 1):
                        if self.mac_grid.cell_type[x, y, z] == CellType.SAND.value:

                            b = -self.mac_grid.divergence[x, y, z] / dt * self.rho

                            cell_residual = 0.0
                            cell_residual = b - (
                                6.0 * self.mac_grid.pressure[x, y, z]
                                - self.mac_grid.pressure[x - 1, y, z]
                                - self.mac_grid.pressure[x + 1, y, z]
                                - self.mac_grid.pressure[x, y - 1, z]
                                - self.mac_grid.pressure[x, y + 1, z]
                                - self.mac_grid.pressure[x, y, z + 1]
                                - self.mac_grid.pressure[x, y, z - 1]
                            )

                            residual += cell_residual * cell_residual

            residual = ti.lang.ops.sqrt(residual)
            residual /= (res_x - 2) * (res_y - 2) * (res_z - 2)

            iterations += 1

        print("residual: ", residual, "iterations: ", iterations)
