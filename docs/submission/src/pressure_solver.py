import taichi as ti
from macgrid import MacGrid, CellType


@ti.data_oriented
class PressureSolver(object):
    def __init__(
        self,
        mac_grid: MacGrid,
        gaus_seidel_min_accuracy: float,
        gaus_seidel_max_iterations: int,
    ):
        self.mac_grid = mac_grid
        self.gaus_seidel_min_accuracy = gaus_seidel_min_accuracy
        self.gaus_seidel_max_iterations = gaus_seidel_max_iterations
        self.rho = 1.0

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
        self.mac_grid.divergence.fill(0.0)
        self.compute_divergence()

        self.mac_grid.pressure.fill(0.0)
        self.gauss_seidel(dt)

    # Computes the velocity projection to make the velocity divergence free
    @ti.kernel
    def project(self, dt: ti.f32):
        for x, y, z in self.mac_grid.v_x:
            if x > 0 and (
                self.mac_grid.cell_type[x, y, z] == CellType.SAND.value
                or self.mac_grid.cell_type[x - 1, y, z] == CellType.SAND.value
                or self.mac_grid.cell_type[x, y - 1, z] == CellType.SAND.value
                or self.mac_grid.cell_type[x, y, z - 1] == CellType.SAND.value
            ):
                self.mac_grid.v_x[x, y, z] -= (
                    self.mac_grid.pressure[x, y, z]
                    - self.mac_grid.pressure[x - 1, y, z]
                ) * dt

        for x, y, z in self.mac_grid.v_y:
            if y > 0 and (
                self.mac_grid.cell_type[x, y, z] == CellType.SAND.value
                or self.mac_grid.cell_type[x, y - 1, z] == CellType.SAND.value
                or self.mac_grid.cell_type[x - 1, y, z] == CellType.SAND.value
                or self.mac_grid.cell_type[x, y, z - 1] == CellType.SAND.value
            ):
                self.mac_grid.v_y[x, y, z] -= (
                    self.mac_grid.pressure[x, y, z]
                    - self.mac_grid.pressure[x, y - 1, z]
                ) * dt

        for x, y, z in self.mac_grid.v_z:
            if z > 0 and (
                self.mac_grid.cell_type[x, y, z] == CellType.SAND.value
                or self.mac_grid.cell_type[x, y, z - 1] == CellType.SAND.value
                or self.mac_grid.cell_type[x - 1, y, z] == CellType.SAND.value
                or self.mac_grid.cell_type[x, y - 1, z] == CellType.SAND.value
            ):
                self.mac_grid.v_z[x, y, z] -= (
                    self.mac_grid.pressure[x, y, z]
                    - self.mac_grid.pressure[x, y, z - 1]
                ) * dt

    # Run Gauss-Seidel as long as max iterations has not been reached and accuracy is not good enough
    def gauss_seidel(self, dt: ti.f32):
        residual = self.gaus_seidel_min_accuracy + 1
        for _ in range(self.gaus_seidel_max_iterations):
            if residual <= self.gaus_seidel_min_accuracy:
                return
            else:
                residual = self.update_pressure(dt)

    @ti.kernel
    def update_pressure(self, dt: ti.f32) -> ti.f32:
        residual = 0.0
        grid_size = self.mac_grid.grid_size

        # Update pressure
        for x, y, z in self.mac_grid.pressure:
            if self.mac_grid.cell_type[x, y, z] == CellType.SAND.value:

                b = -self.mac_grid.divergence[x, y, z] / dt * self.rho
                non_solid_neighbors = 0
                # Check for non-solid cells in x-Direction
                if x != 1 and x != grid_size - 1:
                    non_solid_neighbors += 2
                elif x == 1:
                    non_solid_neighbors += 1
                else:
                    non_solid_neighbors += 1

                # Check for non-solid cells in y-Direction
                if y != 1 and y != grid_size - 1:
                    non_solid_neighbors += 2
                elif y == 1:
                    non_solid_neighbors += 1
                else:
                    non_solid_neighbors += 1

                # Check for non-solid cells in z-Direction
                if z != 1 and z != grid_size - 1:
                    non_solid_neighbors += 2
                elif z == 1:
                    non_solid_neighbors += 1
                else:
                    non_solid_neighbors += 1

                self.mac_grid.pressure[x, y, z] = (
                    b
                    + self.mac_grid.pressure[x - 1, y, z]
                    + self.mac_grid.pressure[x + 1, y, z]
                    + self.mac_grid.pressure[x, y - 1, z]
                    + self.mac_grid.pressure[x, y + 1, z]
                    + self.mac_grid.pressure[x, y, z + 1]
                    + self.mac_grid.pressure[x, y, z - 1]
                ) / non_solid_neighbors

        # Compute residual
        for x, y, z in self.mac_grid.pressure:
            if self.mac_grid.cell_type[x, y, z] == CellType.SAND.value:

                b = -self.mac_grid.divergence[x, y, z] / dt * self.rho

                cell_residual = 0.0

                non_solid_neighbors = 0
                # Check for non-solid cells in x-Direction
                if x != 1 and x != grid_size - 1:
                    non_solid_neighbors += 2
                elif x == 1:
                    non_solid_neighbors += 1
                else:
                    non_solid_neighbors += 1

                # Check for non-solid cells in y-Direction
                if y != 1 and y != grid_size - 1:
                    non_solid_neighbors += 2
                elif y == 1:
                    non_solid_neighbors += 1
                else:
                    non_solid_neighbors += 1

                # Check for non-solid cells in z-Direction
                if z != 1 and z != grid_size - 1:
                    non_solid_neighbors += 2
                elif z == 1:
                    non_solid_neighbors += 1
                else:
                    non_solid_neighbors += 1

                cell_residual = b - (
                    non_solid_neighbors * self.mac_grid.pressure[x, y, z]
                    - self.mac_grid.pressure[x - 1, y, z]
                    - self.mac_grid.pressure[x + 1, y, z]
                    - self.mac_grid.pressure[x, y - 1, z]
                    - self.mac_grid.pressure[x, y + 1, z]
                    - self.mac_grid.pressure[x, y, z + 1]
                    - self.mac_grid.pressure[x, y, z - 1]
                )

                residual += cell_residual * cell_residual

        residual = ti.lang.ops.sqrt(residual)
        residual /= (grid_size - 2) * (grid_size - 2) * (grid_size - 2)
        return residual
