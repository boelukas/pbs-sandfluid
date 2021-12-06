from types import CellType
from typing import List, Tuple, Union
from numpy.lib.function_base import diff
import taichi as ti
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from enum import Enum
import random as rd


class Particle:
    def __init__(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self.pos = position
        self.v = velocity


class CellType(Enum):
    AIR = 0
    SAND = 1
    SOLID = 2


class InvalidIndexError(Exception):
    pass


@ti.data_oriented
class MacGrid:
    def __init__(self, grid_size: int) -> None:
        # grid parameters
        self.grid_size = grid_size

        # Cell centered grids
        self.cell_type = ti.field(
            ti.f32, shape=(self.grid_size, self.grid_size, self.grid_size)
        )
        self.pressure = ti.field(
            ti.f32, shape=(self.grid_size, self.grid_size, self.grid_size)
        )
        self.divergence = ti.field(
            ti.f32, shape=(self.grid_size, self.grid_size, self.grid_size)
        )

        # Edge centered grids
        self.v_x = ti.field(
            ti.f32, shape=(self.grid_size + 1, self.grid_size, self.grid_size)
        )
        self.v_y = ti.field(
            ti.f32, shape=(self.grid_size, self.grid_size + 1, self.grid_size)
        )
        self.v_z = ti.field(
            ti.f32, shape=(self.grid_size, self.grid_size, self.grid_size + 1)
        )

        self.f_x = ti.field(
            ti.f32, shape=(self.grid_size + 1, self.grid_size, self.grid_size)
        )
        self.f_y = ti.field(
            ti.f32, shape=(self.grid_size, self.grid_size + 1, self.grid_size)
        )
        self.f_z = ti.field(
            ti.f32, shape=(self.grid_size, self.grid_size, self.grid_size + 1)
        )

        self.splat_x_weights = ti.field(
            ti.f32, shape=(self.grid_size + 1, self.grid_size, self.grid_size)
        )
        self.splat_y_weights = ti.field(
            ti.f32, shape=(self.grid_size, self.grid_size + 1, self.grid_size)
        )
        self.splat_z_weights = ti.field(
            ti.f32, shape=(self.grid_size, self.grid_size, self.grid_size + 1)
        )

        # particles
        self.particle_pos = ti.Vector.field(
            3,
            ti.f32,
            shape=(self.grid_size * 2, self.grid_size * 2, self.grid_size * 2),
        )
        self.particle_v = ti.Vector.field(
            3,
            ti.f32,
            shape=(self.grid_size * 2, self.grid_size * 2, self.grid_size * 2),
        )
        self.particle_active = ti.field(
            ti.f32, shape=(self.grid_size * 2, self.grid_size * 2, self.grid_size * 2)
        )

        # Initialize grids and particles
        self.reset_fields()

    @ti.kernel
    def test(self):
        for i, j, k in self.pressure:
            self.pressure[i, j, k] = i
        print(1.0 == self.sample_cell_centered(self.pressure, 1.5, 0.5, 0.5))
        print(0.0 == self.sample_cell_centered(self.pressure, 0.5, 0.5, 0.5))
        print(0.75 == self.sample_cell_centered(self.pressure, 1.25, 0.5, 0.5))
        for i, j, k in self.pressure:
            self.pressure[i, j, k] = j
        print(1.0 == self.sample_cell_centered(self.pressure, 0.5, 1.5, 0.5))
        print(0.0 == self.sample_cell_centered(self.pressure, 0.5, 0.5, 0.5))
        print(0.75 == self.sample_cell_centered(self.pressure, 0.5, 1.25, 0.5))

        for i, j, k in self.pressure:
            self.pressure[i, j, k] = k
        print(1.0 == self.sample_cell_centered(self.pressure, 0.5, 0.5, 1.5))
        print(0.0 == self.sample_cell_centered(self.pressure, 0.5, 0.5, 0.5))
        print(0.75 == self.sample_cell_centered(self.pressure, 0.5, 0.5, 1.25))

    @ti.kernel
    def reset_fields(self):
        # Cell centered grids
        self.clear_field(self.cell_type)
        self.clear_field(self.pressure)
        self.clear_field(self.divergence)

        # Edge centered grids
        self.clear_field(self.v_x)
        self.clear_field(self.v_y)
        self.clear_field(self.v_z)

        self.clear_field(self.f_x)
        self.clear_field(self.f_y)
        self.clear_field(self.f_z)

        # particles
        self.clear_field(self.particle_pos, [0.0, 0.0, 0.0])
        self.clear_field(self.particle_v, [0.0, 0.0, 0.0])
        self.clear_field(self.particle_active)

        # Initialize grids and particles
        self.init_cell_type()
        self.init_particles()

    # Initializes the cell types. The border of the grid is always solid
    # The cells with CellType SAND will have active particles
    @ti.func
    def init_cell_type(self):
        for i, j, k in self.cell_type:
            if (
                i == 0
                or j == 0
                or k == 0
                or i == self.grid_size - 1
                or j == self.grid_size - 1
                or k == self.grid_size - 1
            ):
                self.cell_type[i, j, k] = CellType.SOLID.value
            elif 1 <= i <= 2 and 2 <= j <= 3 and 1 <= k <= 2:
                self.cell_type[i, j, k] = CellType.SAND.value
            else:
                self.cell_type[i, j, k] = CellType.AIR.value

    @ti.func
    def reset_cell_type(self):
        self.clear_field(self.cell_type)

        for i, j, k in self.cell_type:
            if (
                i == 0
                or j == 0
                or k == 0
                or i == self.grid_size - 1
                or j == self.grid_size - 1
                or k == self.grid_size - 1
            ):
                self.cell_type[i, j, k] = CellType.SOLID.value

    # Initializes the particles to 8 particles per grid cell.
    # The positions are for grid cell i: i + 0.25 + rand and i + 0.75 + rand
    # In all dimensions and for a rand jitter between [-0.25, 0.25]
    # Only particles in SAND grid cells will be active
    @ti.func
    def init_particles(self):
        for i, j, k in self.particle_pos:
            grid_idx_i = i // 2
            grid_idx_j = j // 2
            grid_idx_k = k // 2
            even_i = i % 2 == 0
            even_j = j % 2 == 0
            even_k = k % 2 == 0
            if (
                self.cell_type[grid_idx_i, grid_idx_j, grid_idx_k]
                == CellType.SAND.value
            ):
                self.particle_active[i, j, k] = 1
                particle_pos_x, particle_pos_y, particle_pos_z = 0.0, 0.0, 0.0
                if even_i:
                    particle_pos_x = (
                        float(grid_idx_i) + 0.25 + ((ti.random(int) % 50) - 25) / 100.0
                    )
                else:
                    particle_pos_x = (
                        grid_idx_i + 0.75 + ((ti.random(int) % 50) - 25) / 100.0
                    )
                if even_j:
                    particle_pos_y = (
                        float(grid_idx_j) + 0.25 + ((ti.random(int) % 50) - 25) / 100.0
                    )
                else:
                    particle_pos_y = (
                        grid_idx_j + 0.75 + ((ti.random(int) % 50) - 25) / 100.0
                    )
                if even_k:
                    particle_pos_z = (
                        float(grid_idx_k) + 0.25 + ((ti.random(int) % 50) - 25) / 100.0
                    )
                else:
                    particle_pos_z = (
                        grid_idx_k + 0.75 + ((ti.random(int) % 50) - 25) / 100.0
                    )

                self.particle_pos[i, j, k] = [
                    particle_pos_x,
                    particle_pos_y,
                    particle_pos_z,
                ]
            else:
                self.particle_active[i, j, k] = 0

    # Taken from 4_fluid.py from the exercises
    @ti.func
    def clear_field(self, f: ti.template(), v: ti.template() = 0):
        for x, y, z in ti.ndrange(*f.shape):
            f[x, y, z] = v

    @ti.func
    def clamp(self, x, min_bound, max_bound):
        return max(min_bound, min(x, max_bound))

    @ti.kernel
    def update_cell_types(self):
        # Wipe out old cell_type grid and initialize the new grid with domain bounds.
        self.reset_cell_type()

        # Mark cells that contain at least one particle with SAND
        for i, j, k in self.particle_pos:
            if self.particle_active[i, j, k] == 1:
                p_pos = self.particle_pos[i, j, k]
                # Get cell idx in which the particle currently resides
                grid_i = self.clamp(int(p_pos[0]), 0, self.grid_size - 1)
                grid_j = self.clamp(int(p_pos[1]), 0, self.grid_size - 1)
                grid_k = self.clamp(int(p_pos[2]), 0, self.grid_size - 1)

                # Check whether cell is solid (boundary)
                if self.cell_type[grid_i, grid_j, grid_k] != CellType.SOLID.value:
                    self.cell_type[grid_i, grid_j, grid_k] = CellType.SAND.value

        # Mark the uninitialized cells with AIR
        for i, j, k in self.cell_type:
            if (
                self.cell_type[i, j, k] != CellType.SOLID.value
                and self.cell_type[i, j, k] != CellType.SAND.value
            ):
                self.cell_type[i, j, k] = CellType.AIR.value

    @ti.kernel
    def neumann_boundary_conditions(self):
        for i, j, k in ti.ndrange(*self.cell_type.shape):
            if self.cell_type[i, j, k] == CellType.SOLID.value:
                self.v_x[i, j, k] = 0.0
                self.v_x[i + 1, j, k] = 0.0
                self.v_y[i, j, k] = 0.0
                self.v_y[i, j + 1, k] = 0.0
                self.v_z[i, j, k] = 0.0
                self.v_z[i, j, k + 1] = 0.0

    # Sample grid with grid origin at (x_offset, y_offset, z_offset)
    @ti.func
    def sample(
        self,
        grid,
        x,
        y,
        z,
        x_offset,
        y_offset,
        z_offset,
        x_resolution,
        y_resolution,
        z_resolution,
    ):
        x_down = self.clamp(int(x - x_offset), 0, x_resolution - 1)
        y_down = self.clamp(int(y - y_offset), 0, y_resolution - 1)
        z_down = self.clamp(int(z - z_offset), 0, z_resolution - 1)
        # print("x_down, y_down, z_down", x_down, y_down, z_down)
        x_up = self.clamp(x_down + 1, 0, x_resolution - 1)
        y_up = self.clamp(y_down + 1, 0, y_resolution - 1)
        z_up = self.clamp(z_down + 1, 0, z_resolution - 1)
        # print("x_down, y_down, z_down", x_up, y_up, z_up)

        diff_x = self.clamp(x - x_offset - x_down, 0.0, 1.0)
        diff_y = self.clamp(y - y_offset - y_down, 0.0, 1.0)
        diff_z = self.clamp(z - z_offset - z_down, 0.0, 1.0)
        # print("diff_x, diff_y, diff_z", diff_x, diff_y, diff_z)

        x_val_front_down = (
            grid[x_down, y_down, z_down] * (1 - diff_x)
            + grid[x_up, y_down, z_down] * diff_x
        )
        x_val_back_down = (
            grid[x_down, y_down, z_up] * (1 - diff_x)
            + grid[x_up, y_down, z_up] * diff_x
        )
        x_val_front_up = (
            grid[x_down, y_up, z_down] * (1 - diff_x)
            + grid[x_up, y_up, z_down] * diff_x
        )
        x_val_back_up = (
            grid[x_down, y_up, z_up] * (1 - diff_x) + grid[x_up, y_up, z_up] * diff_x
        )
        # print("x_val_front_down, x_val_back_down, x_val_front_up, x_val_back_up",x_val_front_down, x_val_back_down, x_val_front_up, x_val_back_up)

        xz_val_down = x_val_front_down * (1 - diff_z) + x_val_back_down * diff_z
        xz_val_up = x_val_front_up * (1 - diff_z) + x_val_back_up * diff_z
        # print("xz_val_down, xz_val_up",xz_val_down, xz_val_up)

        return xz_val_down * (1 - diff_y) + xz_val_up * diff_y

    @ti.func
    def sample_cell_centered(self, grid, x, y, z):
        return self.sample(
            grid, x, y, z, 0.5, 0.5, 0.5, self.grid_size, self.grid_size, self.grid_size
        )

    @ti.func
    def sample_x_edged(self, grid, x, y, z):
        return self.sample(
            grid,
            x,
            y,
            z,
            0,
            0.5,
            0.5,
            self.grid_size + 1,
            self.grid_size,
            self.grid_size,
        )

    @ti.func
    def sample_y_edged(self, grid, x, y, z):
        return self.sample(
            grid,
            x,
            y,
            z,
            0.5,
            0.0,
            0.5,
            self.grid_size,
            self.grid_size + 1,
            self.grid_size,
        )

    @ti.func
    def sample_z_edged(self, grid, x, y, z):
        return self.sample(
            grid,
            x,
            y,
            z,
            0.5,
            0.5,
            0.0,
            self.grid_size,
            self.grid_size,
            self.grid_size + 1,
        )

    # Brings the velocity from the grid to particles. The velocity fields are sampled at the location of the particles.
    @ti.kernel
    def grid_to_particles(self):
        for i, j, k in self.particle_pos:
            if self.particle_active[i, j, k] == 1:
                p = self.particle_pos[i, j, k]
                self.particle_v[i, j, k] = [
                    self.sample_x_edged(self.v_x, p[0], p[1], p[2]),
                    self.sample_y_edged(self.v_y, p[0], p[1], p[2]),
                    self.sample_z_edged(self.v_z, p[0], p[1], p[2]),
                ]

    @ti.func
    def splat(
        self,
        target_field,
        particle_x,
        particle_y,
        particle_z,
        particle_value,
        weights,
        x_offset,
        y_offset,
        z_offset,
        x_resolution,
        y_resolution,
        z_resolution,
    ):
        x_down = self.clamp(int(particle_x - x_offset), 0, x_resolution - 1)
        y_down = self.clamp(int(particle_y - y_offset), 0, y_resolution - 1)
        z_down = self.clamp(int(particle_z - z_offset), 0, z_resolution - 1)

        x_up = self.clamp(x_down + 1, 0, x_resolution - 1)
        y_up = self.clamp(y_down + 1, 0, y_resolution - 1)
        z_up = self.clamp(z_down + 1, 0, z_resolution - 1)

        diff_x = self.clamp(particle_x - x_offset - x_down, 0.0, 1.0)
        diff_y = self.clamp(particle_y - y_offset - y_down, 0.0, 1.0)
        diff_z = self.clamp(particle_z - z_offset - z_down, 0.0, 1.0)

        target_field[x_down, y_down, z_down] += (
            particle_value * (1 - diff_x) * (1 - diff_y) * (1 - diff_z)
        )
        weights[x_down, y_down, z_down] += (1 - diff_x) * (1 - diff_y) * (1 - diff_z)

        target_field[x_up, y_down, z_down] += (
            particle_value * diff_x * (1 - diff_y) * (1 - diff_z)
        )
        weights[x_up, y_down, z_down] += diff_x * (1 - diff_y) * (1 - diff_z)

        target_field[x_down, y_up, z_down] += (
            particle_value * (1 - diff_x) * diff_y * (1 - diff_z)
        )
        weights[x_down, y_up, z_down] += (1 - diff_x) * diff_y * (1 - diff_z)

        target_field[x_down, y_down, z_up] += (
            particle_value * (1 - diff_x) * (1 - diff_y) * diff_z
        )
        weights[x_down, y_down, z_up] += (1 - diff_x) * (1 - diff_y) * diff_z

        target_field[x_up, y_down, z_up] += (
            particle_value * diff_x * (1 - diff_y) * diff_z
        )
        weights[x_up, y_down, z_up] += diff_x * (1 - diff_y) * diff_z

        target_field[x_up, y_up, z_down] += (
            particle_value * diff_x * diff_y * (1 - diff_z)
        )
        weights[x_up, y_up, z_down] += diff_x * diff_y * (1 - diff_z)

        target_field[x_down, y_up, z_up] += (
            particle_value * (1 - diff_x) * diff_y * diff_z
        )
        weights[x_down, y_up, z_up] += (1 - diff_x) * diff_y * diff_z

        target_field[x_up, y_up, z_up] += particle_value * diff_x * diff_y * diff_z
        weights[x_up, y_up, z_up] += diff_x * diff_y * diff_z

    @ti.func
    def splat_cell_centered(self, grid, x, y, z, value, weights):
        return self.splat(
            grid,
            x,
            y,
            z,
            value,
            weights,
            0.5,
            0.5,
            0.5,
            self.grid_size,
            self.grid_size,
            self.grid_size,
        )

    @ti.func
    def splat_x_edged(self, grid, x, y, z, value):
        return self.splat(
            grid,
            x,
            y,
            z,
            value,
            self.splat_x_weights,
            0,
            0.5,
            0.5,
            self.grid_size + 1,
            self.grid_size,
            self.grid_size,
        )

    @ti.func
    def splat_y_edged(self, grid, x, y, z, value):
        return self.splat(
            grid,
            x,
            y,
            z,
            value,
            self.splat_y_weights,
            0.5,
            0.0,
            0.5,
            self.grid_size,
            self.grid_size + 1,
            self.grid_size,
        )

    @ti.func
    def splat_z_edged(self, grid, x, y, z, value):
        return self.splat(
            grid,
            x,
            y,
            z,
            value,
            self.splat_z_weights,
            0.5,
            0.5,
            0.0,
            self.grid_size,
            self.grid_size,
            self.grid_size + 1,
        )

    # Splats the velocity of active particles to the grid.
    # Adds its weighted velocity to each surrounding grid vertex and in the end divides each grid vertex by the sum of all weights
    # that were applied to it.
    @ti.kernel
    def particles_to_grid(self):
        self.clear_field(self.splat_x_weights)
        self.clear_field(self.splat_y_weights)
        self.clear_field(self.splat_z_weights)

        for i, j, k in self.particle_pos:
            if self.particle_active[i, j, k] == 1:
                p = self.particle_pos[i, j, k]
                v = self.particle_v[i, j, k]
                self.splat_x_edged(self.v_x, p[0], p[1], p[2], v[0])
                self.splat_y_edged(self.v_y, p[0], p[1], p[2], v[1])
                self.splat_z_edged(self.v_z, p[0], p[1], p[2], v[2])

        for i, j, k in self.splat_x_weights:
            if self.splat_x_weights[i, j, k] > 0.0:
                self.v_x[i, j, k] /= self.splat_x_weights[i, j, k]

        for i, j, k in self.splat_y_weights:
            if self.splat_y_weights[i, j, k] > 0.0:
                self.v_y[i, j, k] /= self.splat_y_weights[i, j, k]

        for i, j, k in self.splat_y_weights:
            if self.splat_y_weights[i, j, k] > 0.0:
                self.v_z[i, j, k] /= self.splat_y_weights[i, j, k]

    # Explicite euler step to advect particles
    @ti.kernel
    def advect_particles(self, dt: ti.f32):
        for i, j, k in self.particle_pos:
            if self.particle_active[i, j, k] == 1:
                self.particle_pos[i, j, k] += dt * self.particle_v[i, j, k]

    # move particles with midpoint euler from grid velocity
    @ti.kernel
    def advect_particles_midpoint(self, dt: ti.f32):
        for i, j, k in self.particle_pos:
            if self.particle_active[i, j, k] == 1:
                start_pos = self.particle_pos[i, j, k]
                midpos = start_pos + self.velocity_interpolation(
                    start_pos, self.v_x, self.v_y, self.v_z
                ) * (dt * 0.5)
                step = (
                    self.velocity_interpolation(midpos, self.v_x, self.v_y, self.v_z)
                    * dt
                )
                self.particle_pos[i, j, k] += step

    @ti.func
    def velocity_interpolation(self, pos, vel_x, vel_y, vel_z):
        _ux = self.sample_x_edged(vel_x, pos.x, pos.y, pos.z)
        _uy = self.sample_y_edged(vel_y, pos.x, pos.y, pos.z)
        _uz = self.sample_z_edged(vel_z, pos.x, pos.y, pos.z)
        return ti.Vector([_ux, _uy, _uz])

    def show_v_y(self):
        vely_numpy = self.v_y.to_numpy()
        resolution = min(vely_numpy.shape)
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.set_ylabel("z")
        ax.set_zlabel("y")
        ax.set_xlabel("x")

        x, y, z = np.meshgrid(
            np.arange(0, resolution, 1),
            np.arange(0, resolution, 1),
            np.arange(0, resolution, 1),
        )

        u = np.zeros((resolution, resolution, resolution))
        w = np.zeros((resolution, resolution, resolution))
        v = vely_numpy[:resolution, :resolution, :resolution]

        ax.quiver(y, z, x, u, w, v, length=1, color="black")
        plt.show()

    # Plots the pressure. Upwards pointing arrows mean positive pressure
    def show_pressure(self):
        p_numpy = self.pressure.to_numpy()
        resolution = min(p_numpy.shape)
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.set_ylabel("z")
        ax.set_zlabel("y")
        ax.set_xlabel("x")
        plt.xlim([0, resolution - 1])
        plt.ylim([0, resolution - 1])
        ax.set_zlim(0, resolution - 1)
        x, y, z = np.meshgrid(
            np.arange(0, resolution, 1),
            np.arange(0, resolution, 1),
            np.arange(0, resolution, 1),
        )

        u = np.zeros((resolution, resolution, resolution))
        w = np.zeros((resolution, resolution, resolution))
        v = p_numpy[:resolution, :resolution, :resolution]

        ax.quiver(y, z, x, u, w, v, length=1, color="black")
        # print(np.unravel_index(p_numpy.argmax(), p_numpy.shape))
        # print(np.max(p_numpy))
        plt.show()

    # Plots divergence. Upwards pointing values are positive.
    def show_divergence(self):
        div_numpy = self.divergence.to_numpy()
        resolution = min(div_numpy.shape)
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.set_ylabel("z")
        ax.set_zlabel("y")
        ax.set_xlabel("x")
        plt.xlim([0, resolution - 1])
        plt.ylim([0, resolution - 1])
        ax.set_zlim(0, resolution - 1)
        x, y, z = np.meshgrid(
            np.arange(0, resolution, 1),
            np.arange(0, resolution, 1),
            np.arange(0, resolution, 1),
        )

        u = np.zeros((resolution, resolution, resolution))
        w = np.zeros((resolution, resolution, resolution))
        v = div_numpy[:resolution, :resolution, :resolution]

        ax.quiver(y, z, x, u, w, v, length=1, color="black")
        # print(np.unravel_index(div_numpy.argmax(), div_numpy.shape))
        # print(np.max(div_numpy))
        plt.show()


class sMACGrid:
    def __init__(self, resolution: int) -> None:
        # size of the cubic simulation grid_size
        # determines the number of voxels the grid_size is divided into.
        self.voxel_size = 1.0
        # grid that stores velocity and pressure attributes
        self.grid_size = resolution
        self.dx = 1
        # Velocity is stored at the faces of the cell/voxel along the corresponding axis
        self.velX_grid = ti.field(
            ti.f32, shape=(self.grid_size + 1, self.grid_size, self.grid_size)
        )
        self.velY_grid = ti.field(
            ti.f32, shape=(self.grid_size, self.grid_size + 1, self.grid_size)
        )
        self.velZ_grid = ti.field(
            ti.f32, shape=(self.grid_size, self.grid_size, self.grid_size + 1)
        )

        self.forceX_grid = ti.field(
            ti.f32, shape=(self.grid_size + 1, self.grid_size, self.grid_size)
        )
        self.forceY_grid = ti.field(
            ti.f32, shape=(self.grid_size, self.grid_size + 1, self.grid_size)
        )
        self.forceZ_grid = ti.field(
            ti.f32, shape=(self.grid_size, self.grid_size, self.grid_size + 1)
        )
        # Pressure is sampled at the cell center
        self.pressure_grid = ti.field(
            ti.f32, shape=(self.grid_size, self.grid_size, self.grid_size)
        )
        # Indicates which cell has fluid particles (1) and which not (0)
        # NOTE: Has to be initialized/set after the advection of particles on the grid
        self.has_particles = ti.field(
            ti.f32, shape=(self.grid_size, self.grid_size, self.grid_size)
        )
        self.divergence_grid = ti.field(
            ti.f32, shape=(self.grid_size, self.grid_size, self.grid_size)
        )

        # Weight factor for SPH Kernel
        self.sph_kernel = 315.0 / (64.0 * np.pi * (self.voxel_size ** 9))

    # Returns index of the voxel in which the given Particle is present
    def get_voxel(self, position: np.ndarray) -> Tuple[int, int, int]:
        voxel_size = self.voxel_size
        i = int(position[0] // voxel_size)
        j = int(position[1] // voxel_size)
        k = int(position[2] // voxel_size)

        if i >= self.grid_size or j >= self.grid_size or k >= self.grid_size:
            raise InvalidIndexError("Particle is out of grid_size bounds.")

        return (i, j, k)

    # Returns index of the voxel in which the given Particle is present
    def get_voxel_particle(self, particle: Particle) -> Tuple[int, int, int]:
        voxel_size = self.voxel_size
        i = int(particle.pos[0] // voxel_size)
        j = int(particle.pos[1] // voxel_size)
        k = int(particle.pos[2] // voxel_size)

        if i >= self.grid_size or j >= self.grid_size or k >= self.grid_size:
            raise InvalidIndexError("Particle is out of grid_size bounds.")

        return (i, j, k)

    # Returns the global position of a staggered grid point
    def gridindex_to_position(
        self, i: int, j: int, k: int, grid: str
    ) -> Tuple[float, float, float]:

        if grid == "velX":
            pos = (
                i * self.voxel_size,
                (j + 0.5) * self.voxel_size,
                (k + 0.5) * self.voxel_size,
            )
        elif grid == "velY":
            pos = (
                (i + 0.5) * self.voxel_size,
                j * self.voxel_size,
                (k + 0.5) * self.voxel_size,
            )
        elif grid == "velZ":
            pos = (
                (i + 0.5) * self.voxel_size,
                (j + 0.5) * self.voxel_size,
                k * self.voxel_size,
            )
        elif grid == "pressure":
            pos = (
                (i + 0.5) * self.voxel_size,
                (j + 0.5) * self.voxel_size,
                (k + 0.5) * self.voxel_size,
            )
        else:
            print("No grid specified.")
            return None

        return pos

    # Returns index of velocity array to which the given Particle is closest (kernel size = 1 voxel)
    def get_splat_index(self, pos: np.ndarray, grid: str) -> Tuple[int, int, int]:
        voxel_size = self.voxel_size

        if grid == "velX":
            i = int(round(pos[0]))
            j = int(round(pos[1] - 0.5))
            k = int(round(pos[2] - 0.5))
        elif grid == "velY":
            i = int(round(pos[0] - 0.5))
            j = int(round(pos[1]))
            k = int(round(pos[2] - 0.5))
        elif grid == "velZ":
            i = int(round(pos[0] - 0.5))
            j = int(round(pos[1] - 0.5))
            k = int(round(pos[2]))
        else:
            print("No grid specified.")
            return None

        if i >= self.grid_size or j >= self.grid_size or k >= self.grid_size:
            raise InvalidIndexError("Particle is out of grid_size bounds.")

        return (i, j, k)

    def weigthed_average(
        self, particles: List[Tuple[np.ndarray, float]], pos: np.ndarray
    ) -> float:
        weights = []
        vel = []

        for (p, v) in particles:
            vel.append(v)
            dist = np.linalg.norm(p - pos)
            w = self.sph_kernel * (self.voxel_size ** 2 - dist ** 2)
            weights.append(w)

        vel = np.asarray(vel)
        weights = np.asarray(weights)
        avg = np.average(a=vel, weights=weights)
        return avg

    # Transfer values of particle velocities on the grid with weighted neighbourhood averaging.
    def splat_velocity(self, particles: List[Particle], small_kernel=True) -> None:
        # initialize array for final grid values
        size = self.grid_size
        valuesX = np.zeros((size + 1, size, size))
        valuesY = np.zeros((size, size + 1, size))
        valuesZ = np.zeros((size, size, size + 1))

        # initialize bins as lists
        binsX = {}
        binsY = {}
        binsZ = {}

        for p in particles:
            pos = p.pos
            (u, v, w) = tuple(p.v)

            if small_kernel:
                # kernel of size 1: only assigns particles to closest gridpoint
                idx = self.get_splat_index(pos, "velX")
                for x in range(max(0, idx[0] - 1), min(self.grid_size, idx[0] + 1)):
                    for y in range(max(0, idx[1] - 1), min(self.grid_size, idx[1] + 1)):
                        for z in range(
                            max(0, idx[2] - 1), min(self.grid_size, idx[2] + 1)
                        ):
                            temp = (x, y, z)
                            if temp in binsX:
                                binsX[temp].append((pos, u))
                            else:
                                binsX[temp] = [(pos, u)]

                idx = self.get_splat_index(pos, "velY")
                for x in range(max(0, idx[0] - 1), min(self.grid_size, idx[0] + 1)):
                    for y in range(max(0, idx[1] - 1), min(self.grid_size, idx[1] + 1)):
                        for z in range(
                            max(0, idx[2] - 1), min(self.grid_size, idx[2] + 1)
                        ):
                            temp = (x, y, z)
                            if temp in binsY:
                                binsY[temp].append((pos, v))
                            else:
                                binsY[temp] = [(pos, v)]

                idx = self.get_splat_index(pos, "velZ")
                for x in range(max(0, idx[0] - 1), min(self.grid_size, idx[0] + 1)):
                    for y in range(max(0, idx[1] - 1), min(self.grid_size, idx[1] + 1)):
                        for z in range(
                            max(0, idx[2] - 1), min(self.grid_size, idx[2] + 1)
                        ):
                            temp = (x, y, z)
                            if temp in binsZ:
                                binsZ[temp].append((pos, w))
                            else:
                                binsZ[temp] = [(pos, w)]

            else:
                # kernel of size 3: also assigns particles to adjecent gridpoints
                idx = self.get_splat_index(pos, "velX")
                raise NotImplementedError

        for (i, j, k) in list(binsX.keys()):
            closest_particles = binsX[(i, j, k)]
            grid_pos = np.asarray(self.gridindex_to_position(i, j, k, "velX"))
            valuesX[i, j, k] = self.weigthed_average(
                particles=closest_particles, pos=grid_pos
            )
        for (i, j, k) in list(binsY.keys()):
            closest_particles = binsY[(i, j, k)]
            grid_pos = np.asarray(self.gridindex_to_position(i, j, k, "velY"))
            valuesY[i, j, k] = self.weigthed_average(
                particles=closest_particles, pos=grid_pos
            )
        for (i, j, k) in list(binsZ.keys()):
            closest_particles = binsZ[(i, j, k)]
            grid_pos = np.asarray(self.gridindex_to_position(i, j, k, "velZ"))
            valuesZ[i, j, k] = self.weigthed_average(
                particles=closest_particles, pos=grid_pos
            )

        self.set_grid_velocity(valuesX, valuesY, valuesZ)

        return None

    # Set new grid velocity
    def set_grid_velocity(
        self, valuesX: np.ndarray, valuesY: np.ndarray, valuesZ: np.ndarray
    ) -> None:
        self.velX_grid.from_numpy(valuesX)
        self.velY_grid.from_numpy(valuesY)
        self.velZ_grid.from_numpy(valuesZ)
        return None

    def index_in_bounds(self, i: int, j: int, k: int, grid: ti.template()) -> bool:
        shape = grid.shape
        if i >= shape[0] or j >= shape[1] or k >= shape[2]:
            # raise InvalidIndexError("Index is out of bounds.")
            return False
        return True

    # Returns the velocity evaluated at the center of the voxel (where pressure is stored)
    def sample_velocity_at_center(
        self, i: int, j: int, k: int
    ) -> Tuple[float, float, float]:
        is_in_bound = self.index_in_bounds(i, j, k, self.pressure_grid)
        if not is_in_bound:
            raise InvalidIndexError("Index is out of bounds.")

        # Linearly interpolated between faces of the voxel
        velX = 0.5 * (self.velX_grid[i, j, k] + self.velX_grid[i + 1, j, k])
        velY = 0.5 * (self.velY_grid[i, j, k] + self.velY_grid[i, j + 1, k])
        velZ = 0.5 * (self.velZ_grid[i, j, k] + self.velZ_grid[i, j, k + 1])
        return (velX, velY, velZ)

    """
    def trilinear_interpolation(self, values: list, position: Tuple[float,float,float]) -> float:
        x, y, z = position
        res = values[0] * (1 - x) * (1 - y) * (1 - z) + \
           values[1] * x * (1 - y) * (1 - z) + \
           values[2] * (1 - x) * y * (1 - z) + \
           values[3] * (1 - x) * (1 - y) * z + \
           values[4] * x * (1 - y) * z + \
           values[5] * (1 - x) * y * z + \
           values[6] * x * y * (1 - z) + \
           values[7] * x * y * z
        return res
    """

    # Description: Given grid coordinates of a particle return its sample velocity with trilinear interpolation
    # NOTE: The "Particles" argument must contain all particles within a particular grid cell!!
    def sample_velocity(
        self, particles: Union[List[Particle], Tuple[float, float, float]], RK2=False
    ) -> Tuple[float, float, float]:
        def get_sample_points(i, j, k, grid):

            values = np.zeros(shape=(2, 2, 2))
            if self.index_in_bounds(i, j, k, grid):
                values[0, 0, 0] = grid[i, j, k]
            if self.index_in_bounds(i, j, k + 1, grid):
                values[0, 0, 1] = grid[i, j, k + 1]
            if self.index_in_bounds(i, j + 1, k, grid):
                values[0, 1, 0] = grid[i, j + 1, k]
            if self.index_in_bounds(i, j + 1, k + 1, grid):
                values[0, 1, 1] = grid[i, j + 1, k + 1]
            if self.index_in_bounds(i + 1, j, k, grid):
                values[1, 0, 0] = grid[i + 1, j, k]
            if self.index_in_bounds(i + 1, j, k + 1, grid):
                values[1, 0, 1] = grid[i + 1, j, k + 1]
            if self.index_in_bounds(i + 1, j + 1, k, grid):
                values[1, 1, 0] = grid[i + 1, j + 1, k]
            if self.index_in_bounds(i + 1, j + 1, k + 1, grid):
                values[1, 1, 1] = grid[i + 1, j + 1, k + 1]

            return values

        particles_position_list = []
        idx = None

        if RK2:
            pos = particles
            particles_position_list.append(pos)
            idx = self.get_voxel(pos)
        else:
            # Extract position and store in list
            for p in particles:
                idx_curr = self.get_voxel_particle(particle=p)
                if idx:
                    assert idx_curr == idx
                else:
                    idx = idx_curr
                particles_position_list.append(p.pos)

        i, j, k = idx
        values_X = get_sample_points(i, j, k, self.velX_grid)
        values_Y = get_sample_points(i, j, k, self.velY_grid)
        values_Z = get_sample_points(i, j, k, self.velZ_grid)
        x_axis = np.linspace(0, self.voxel_size, 2)
        y_axis = np.linspace(0, self.voxel_size, 2)
        z_axis = np.linspace(0, self.voxel_size, 2)

        grid_relative_pos_x = []
        grid_relative_pos_y = []
        grid_relative_pos_z = []

        for p_pos in particles_position_list:
            x, y, z = tuple(p_pos)

            # Grid position of velocity sampling grid point of the x component at given index (i,j,k)
            nx, ny, nz = self.gridindex_to_position(i, j, k, "velX")
            # Grid-relative coordinates required for interpolation
            gx, gy, gz = max(0, x - nx), max(0, y - ny), max(0, z - nz)
            grid_relative_pos_x.append([gx, gy, gz])

            # Grid position of velocity sampling grid point of the y component at given index (i,j,k)
            nx, ny, nz = self.gridindex_to_position(i, j, k, "velY")
            gx, gy, gz = max(0, x - nx), max(0, y - ny), max(0, z - nz)
            grid_relative_pos_y.append([gx, gy, gz])

            # Grid position of velocity sampling grid point of the z component at given index (i,j,k)
            nx, ny, nz = self.gridindex_to_position(i, j, k, "velZ")
            gx, gy, gz = max(0, x - nx), max(0, y - ny), max(0, z - nz)
            grid_relative_pos_z.append([gx, gy, gz])

        interpolated_X = RegularGridInterpolator(
            points=(x_axis, y_axis, z_axis), values=values_X
        )(grid_relative_pos_x)
        interpolated_Y = RegularGridInterpolator(
            points=(x_axis, y_axis, z_axis), values=values_Y
        )(grid_relative_pos_y)
        interpolated_Z = RegularGridInterpolator(
            points=(x_axis, y_axis, z_axis), values=values_Z
        )(grid_relative_pos_z)

        # interpolated_X = self.trilinear_interpolation(values=values_X.flatten().tolist(),position=tuple(grid_relative_pos_x[0]))
        # interpolated_Y = self.trilinear_interpolation(values=values_Y.flatten().tolist(),position=tuple(grid_relative_pos_y[0]))
        # interpolated_Z = self.trilinear_interpolation(values=values_Z.flatten().tolist(),position=tuple(grid_relative_pos_z[0]))

        particle_velocities = [
            list(vel) for vel in zip(interpolated_X, interpolated_Y, interpolated_Z)
        ]
        return np.array(particle_velocities)

    # Velocity projection step of PIC solver. Update grid velocities after solving pressure equations.
    def grid_to_particles(self, particles: List[Particle]) -> None:

        # Gather all particles that belong in the same grid cell
        bins = {}
        for p in particles:
            grid_cell_idx = self.get_voxel_particle(p)
            if grid_cell_idx in bins:
                bins[grid_cell_idx] += [p]
            else:
                bins[grid_cell_idx] = [p]

        # Interpolate velocities for particles per grid cell
        for cell_idx in list(bins.keys()):
            interpolated_velocity = self.sample_velocity(bins[cell_idx])

            # Transfer the interpolated velocity to individual particles
            for i in range(len(bins[cell_idx])):
                particle = bins[cell_idx][i]
                particle.v = interpolated_velocity[i]

        return None

    # Given grid coordinates
    def sample_pressure(self, particle: Particle) -> float:
        i, j, k = self.get_voxel_particle(particle)
        # Currently: This returns the pressure stored within the center of the voxel in which a given particle is present
        pressure = self.pressure_grid[i, j, k]
        return pressure

    def midpoint_euler(self, pos: np.ndarray, step_size: float) -> np.ndarray:
        timestep = step_size / 2
        temp_vel = self.dxdt(
            pos + timestep * self.sample_velocity(pos, RK2=True)[0], timestep
        )
        expl_pos = pos + step_size * temp_vel
        impl_pos = pos + step_size * self.dxdt(0.5 * (pos + expl_pos), timestep)

        return impl_pos

    # Solve Runge-Kutta ODE of second order
    # https://stackoverflow.com/questions/35258628/what-will-be-python-code-for-runge-kutta-second-method

    def runge_kutta_2(self, pos: np.ndarray, dt: float) -> np.ndarray:

        t = np.linspace(0.2 * dt, dt, 5)
        n = len(t)
        x = np.array([pos] * n)
        for i in range(n - 1):
            h = t[i + 1] - t[i]  # 1/5 in our case

            k1 = h * self.dxdt(x[i], t[i])
            x[i + 1] = x[i] + h * self.dxdt(x[i] + k1, t[i] + h / 2.0)

        # return x
        return x[n - 1]

    def dxdt(self, x: np.ndarray, t: float) -> np.ndarray:
        # computes velocity at point x at time t given a velocity field
        # x_bound = np.maximum(np.zeros(3), np.minimum(np.array([self.grid_size] * 3), x))
        vel_x = self.sample_velocity(x, RK2=True)

        # forward
        y = x + t * vel_x[0]

        # y_bound = np.maximum(np.zeros(3), np.minimum(np.array([self.grid_size] * 3), y))
        dx = self.sample_velocity(y, RK2=True)

        return dx[0]

    @ti.func
    def clear_field(self, target_field: ti.template(), zero_value: ti.template() = 0):
        for x, y, z in ti.ndrange(
            target_field.shape[0], target_field.shape[1], target_field.shape[2]
        ):
            target_field[x, y, z] = zero_value

    # Plots the Y velocity without the last Y layer
    def show_velY(self):
        vely_numpy = self.velY_grid.to_numpy()
        resolution = min(vely_numpy.shape)
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.set_ylabel("z")
        ax.set_zlabel("y")
        ax.set_xlabel("x")

        x, y, z = np.meshgrid(
            np.arange(0, resolution, 1),
            np.arange(0, resolution, 1),
            np.arange(0, resolution, 1),
        )

        u = np.zeros((resolution, resolution, resolution))
        w = np.zeros((resolution, resolution, resolution))
        v = vely_numpy[:resolution, :resolution, :resolution]

        ax.quiver(y, z, x, u, w, v, length=1, color="black")
        plt.show()
