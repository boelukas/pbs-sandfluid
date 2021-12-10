import taichi as ti
import numpy as np
from macgrid import MacGrid, CellType


@ti.data_oriented
class SandSolver(object):
    def __init__(self, mac_grid: MacGrid):
        self.mac_grid = mac_grid
        # Phi, between 30 to 45 deg in rad
        self.internal_friction_angle = 0.534
        # rho
        self.density = 0.1
        self.delta = 1.0

    def sand_steps(self, dt):
        self.mac_grid.strain_rate.fill(0.0)
        self.mac_grid.frictional_stress_divergence.fill(0.0)
        self.mac_grid.cell_rigid.fill(0)
        self.mac_grid.frictional_stress.fill(
            ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        )
        self.mac_grid.rigid_stress.fill(
            ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        )

        self.compute_strain_rate()
        self.compute_frictional_stress()
        self.compute_rigid_stress(dt)
        self.compute_rigid_sand_cells()
        self.compute_stress_divergence()
        self.update_velocity_frictional_stess(dt)

    @ti.kernel
    def update_velocity_frictional_stess(self, dt: ti.template()):
        for x, y, z in self.mac_grid.v_x:
            if x > 0 and (
                self.mac_grid.cell_type[x, y, z] == CellType.SAND.value
                or self.mac_grid.cell_type[x - 1, y, z] == CellType.SAND.value
                or self.mac_grid.cell_type[x, y - 1, z] == CellType.SAND.value
                or self.mac_grid.cell_type[x, y, z - 1] == CellType.SAND.value
            ):
                self.mac_grid.v_x[x, y, z] += (dt / self.density) * (
                    self.mac_grid.frictional_stress_divergence[x, y, z]
                    - self.mac_grid.frictional_stress_divergence[x - 1, y, z]
                )
        for x, y, z in self.mac_grid.v_y:
            if y > 0 and (
                self.mac_grid.cell_type[x, y, z] == CellType.SAND.value
                or self.mac_grid.cell_type[x - 1, y, z] == CellType.SAND.value
                or self.mac_grid.cell_type[x, y - 1, z] == CellType.SAND.value
                or self.mac_grid.cell_type[x, y, z - 1] == CellType.SAND.value
            ):
                self.mac_grid.v_y[x, y, z] += (dt / self.density) * (
                    self.mac_grid.frictional_stress_divergence[x, y, z]
                    - self.mac_grid.frictional_stress_divergence[x, y - 1, z]
                )
        for x, y, z in self.mac_grid.v_z:
            if z > 0 and (
                self.mac_grid.cell_type[x, y, z] == CellType.SAND.value
                or self.mac_grid.cell_type[x - 1, y, z] == CellType.SAND.value
                or self.mac_grid.cell_type[x, y - 1, z] == CellType.SAND.value
                or self.mac_grid.cell_type[x, y, z - 1] == CellType.SAND.value
            ):
                self.mac_grid.v_z[x, y, z] += (dt / self.density) * (
                    self.mac_grid.frictional_stress_divergence[x, y, z]
                    - self.mac_grid.frictional_stress_divergence[x, y, z - 1]
                )

    # Computes the strain rate D
    @ti.kernel
    def compute_strain_rate(self):
        for x, y, z in self.mac_grid.strain_rate:
            if self.mac_grid.cell_type[x, y, z] == CellType.SAND.value:
                du_dx = self.mac_grid.v_x[x + 1, y, z] - self.mac_grid.v_x[x, y, z]
                du_dy = self.mac_grid.v_x[x, y + 1, z] - self.mac_grid.v_x[x, y, z]
                du_dz = self.mac_grid.v_x[x, y, z + 1] - self.mac_grid.v_x[x, y, z]

                dv_dx = self.mac_grid.v_y[x + 1, y, z] - self.mac_grid.v_y[x, y, z]
                dv_dy = self.mac_grid.v_y[x, y + 1, z] - self.mac_grid.v_y[x, y, z]
                dv_dz = self.mac_grid.v_y[x, y, z + 1] - self.mac_grid.v_y[x, y, z]

                dw_dx = self.mac_grid.v_z[x + 1, y, z] - self.mac_grid.v_z[x, y, z]
                dw_dy = self.mac_grid.v_z[x, y + 1, z] - self.mac_grid.v_z[x, y, z]
                dw_dz = self.mac_grid.v_z[x, y, z + 1] - self.mac_grid.v_z[x, y, z]

                self.mac_grid.strain_rate[x, y, z] = ti.Matrix(
                    [
                        [du_dx, 0.5 * (dv_dx + du_dy), 0.5 * (dw_dx + du_dz)],
                        [0.5 * (du_dy + dv_dx), dv_dy, 0.5 * (dw_dy + dv_dz)],
                        [0.5 * (du_dz + dw_dx), 0.5 * (dv_dz + dw_dy), dw_dz],
                    ]
                )

    # Computes the frictional stress sigma_f
    @ti.kernel
    def compute_frictional_stress(self):
        for x, y, z in self.mac_grid.frictional_stress:
            if self.mac_grid.cell_type[x, y, z] == CellType.SAND.value:
                norm_D = self.frobenius_norm(self.mac_grid.strain_rate[x, y, z])
                if norm_D > 1e-5:
                    self.mac_grid.frictional_stress[x, y, z] = (
                        -ti.sin(self.internal_friction_angle)
                        * self.mac_grid.pressure[x, y, z]
                        * (1.0 / ti.sqrt(1.0 / 3.0))
                        * (1.0 / norm_D)
                        * self.mac_grid.strain_rate[x, y, z]
                    )
                else:
                    self.mac_grid.frictional_stress[x, y, z] = ti.Matrix(
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                    )

    # Computes the rigid stress sigma_f
    @ti.kernel
    def compute_rigid_stress(self, dt: ti.f32):
        for x, y, z in self.mac_grid.rigid_stress:
            if self.mac_grid.cell_type[x, y, z] == CellType.SAND.value:
                self.mac_grid.rigid_stress[x, y, z] = (
                    -self.density * (1.0 / dt) * self.mac_grid.strain_rate[x, y, z]
                )

    @ti.kernel
    def compute_rigid_sand_cells(self):
        for x, y, z in self.mac_grid.cell_rigid:
            if self.mac_grid.cell_type[x, y, z] == CellType.SAND.value:
                # yield condition
                sigma = self.mac_grid.rigid_stress[x, y, z]
                sigma_m = sigma.trace() / 3.0
                sigma_bar = self.frobenius_norm(sigma - sigma_m * self.delta) / ti.sqrt(
                    2.0
                )
                lhs = ti.sqrt(3.0) * sigma_bar
                rhs = ti.sin(self.internal_friction_angle) * sigma_m
                # TODO: Should maybe be >=
                # print(lhs, rhs)
                if lhs < rhs:
                    self.mac_grid.cell_rigid[x, y, z] = 1
                else:
                    self.mac_grid.cell_rigid[x, y, z] = 0

    # Computes the rigid stress sigma_f
    @ti.kernel
    def compute_stress_divergence(self):
        for x, y, z in self.mac_grid.frictional_stress_divergence:
            dsig_dx = (
                self.mac_grid.frictional_stress[x + 1, y, z]
                - self.mac_grid.frictional_stress[x, y, z]
            )
            dsig_dx_col_sum = dsig_dx[0, 0] + dsig_dx[1, 0] + dsig_dx[2, 0]

            dsig_dy = (
                self.mac_grid.frictional_stress[x, y + 1, z]
                - self.mac_grid.frictional_stress[x, y, z]
            )
            dsig_dy_col_sum = dsig_dy[0, 1] + dsig_dy[1, 1] + dsig_dy[2, 1]

            dsig_dz = (
                self.mac_grid.frictional_stress[x, y, z + 1]
                - self.mac_grid.frictional_stress[x, y, z]
            )
            dsig_dz_col_sum = dsig_dz[0, 2] + dsig_dz[1, 2] + dsig_dz[2, 2]

            self.mac_grid.frictional_stress_divergence[x, y, z] = (
                dsig_dx_col_sum + dsig_dy_col_sum + dsig_dz_col_sum
            )

    # Computes the frobenius norm of m: 3x3
    @ti.func
    def frobenius_norm(self, m: ti.template()):
        return ti.sqrt(
            m[0, 0] ** 2
            + m[0, 1] ** 2
            + m[0, 2] ** 2
            + m[1, 0] ** 2
            + m[1, 1] ** 2
            + m[1, 2] ** 2
            + m[2, 0] ** 2
            + m[2, 1] ** 2
            + m[2, 2] ** 2
        )
