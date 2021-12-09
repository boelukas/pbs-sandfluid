# Reference pressure solver implementation adapted to 3D from https://github.com/Wimacs/taichi_code
import taichi as ti
import numpy as np
from macgrid import MacGrid, CellType


@ti.data_oriented
class ReferencePressureSolver(object):
    def __init__(self, mac_grid: MacGrid):
        self.name = "Reference solver"
        self.mac_grid = mac_grid
        self.gaus_seidel_min_accuracy = 0.0001
        self.gaus_seidel_max_iterations = 10000
        self.rho = 1.0
        self.density = 0.6
        self.reference_ps = MultigridPCGPoissonSolver(
            self.mac_grid.cell_type,
            self.mac_grid.grid_size,
            self.mac_grid.grid_size,
            self.mac_grid.grid_size,
        )

    # Computes the divergence
    @ti.kernel
    def compute_divergence(self):
        for x, y, z in self.mac_grid.divergence:
            if self.mac_grid.cell_type[x, y, z] == CellType.SAND.value:
                du_dx = self.mac_grid.v_x[x, y, z] - self.mac_grid.v_x[x + 1, y, z]
                du_dy = self.mac_grid.v_y[x, y, z] - self.mac_grid.v_y[x, y + 1, z]
                du_dz = self.mac_grid.v_z[x, y, z] - self.mac_grid.v_z[x, y, z + 1]
                self.mac_grid.divergence[x, y, z] = du_dx + du_dy + du_dz

    def compute_pressure(self, dt: ti.f32):
        # self.mac_grid.clear_field(self.mac_grid.divergence)
        self.mac_grid.divergence.fill(0.0)
        self.compute_divergence()  # check

        self.mac_grid.pressure.fill(0.0)
        # self.gauss_seidel(dt)
        self.reference_ps.solve(self.mac_grid.pressure, self.mac_grid.divergence)

    @ti.kernel
    def project(self, dt: ti.f32):
        # self.density = 0.6
        for x, y, z in self.mac_grid.v_x:
            if x > 0 and (
                self.mac_grid.cell_type[x, y, z] == CellType.SAND.value
                or self.mac_grid.cell_type[x - 1, y, z] == CellType.SAND.value
            ):
                self.mac_grid.v_x[x, y, z] += (
                    self.mac_grid.pressure[x - 1, y, z]
                    - self.mac_grid.pressure[x, y, z]
                ) * (1.0 / self.density)

        for x, y, z in self.mac_grid.v_y:
            if y > 0 and (
                self.mac_grid.cell_type[x, y, z] == CellType.SAND.value
                or self.mac_grid.cell_type[x, y - 1, z] == CellType.SAND.value
            ):
                self.mac_grid.v_y[x, y, z] += (
                    self.mac_grid.pressure[x, y - 1, z]
                    - self.mac_grid.pressure[x, y, z]
                ) * (1.0 / self.density)

        for x, y, z in self.mac_grid.v_z:
            if z > 0 and (
                self.mac_grid.cell_type[x, y, z] == CellType.SAND.value
                or self.mac_grid.cell_type[x, y, z - 1] == CellType.SAND.value
            ):
                self.mac_grid.v_z[x, y, z] += (
                    self.mac_grid.pressure[x, y, z - 1]
                    - self.mac_grid.pressure[x, y, z]
                ) * (1.0 / self.density)


@ti.data_oriented
class MultigridPCGPoissonSolver:
    def __init__(self, marker, nx, ny, nz):
        shape = (nx, ny, nz)
        self.nx, self.ny, self.nz = shape
        print(f"nx, ny, nz = {nx}, {ny}, {nz}")

        self.dim = 3
        self.max_iters = 300
        self.n_mg_levels = 4
        self.pre_and_post_smoothing = 2
        self.bottom_smoothing = 10
        self.use_multigrid = True

        def _res(l):
            return (nx // (2 ** l), ny // (2 ** l), nz // (2 ** l))

        self.r = [
            ti.field(dtype=ti.f32, shape=_res(_)) for _ in range(self.n_mg_levels)
        ]  # residual
        self.z = [
            ti.field(dtype=ti.f32, shape=_res(_)) for _ in range(self.n_mg_levels)
        ]  # M^-1 r
        self.d = [
            ti.field(dtype=ti.f32, shape=_res(_)) for _ in range(self.n_mg_levels)
        ]  # temp
        self.f = [marker] + [
            ti.field(dtype=ti.i32, shape=_res(_)) for _ in range(self.n_mg_levels - 1)
        ]  # marker
        self.L = [
            ti.Vector.field(8, dtype=ti.f32, shape=_res(_))
            for _ in range(self.n_mg_levels)
        ]  # -L operator

        self.x = ti.field(dtype=ti.f32, shape=shape)  # solution
        self.p = ti.field(dtype=ti.f32, shape=shape)  # conjugate gradient
        self.Ap = ti.field(dtype=ti.f32, shape=shape)  # matrix-vector product
        self.alpha = ti.field(dtype=ti.f32, shape=())  # step size
        self.beta = ti.field(dtype=ti.f32, shape=())  # step size
        self.sum = ti.field(dtype=ti.f32, shape=())  # storage for reductions

        for _ in range(self.n_mg_levels):
            print(f"r[{_}].shape = {self.r[_].shape}")
        for _ in range(self.n_mg_levels):
            print(f"L[{_}].shape = {self.L[_].shape}")

    @ti.func
    def is_fluid(self, f, i, j, k, nx, ny, nz):
        return (
            i >= 0
            and i < nx
            and j >= 0
            and j < ny
            and k >= 0
            and k < nz
            and CellType.SAND.value == f[i, j, k]
        )

    @ti.func
    def is_solid(self, f, i, j, k, nx, ny, nz):
        return (
            i >= 0
            and i < nx
            and j >= 0
            and j < ny
            and k >= 0
            and k < nz
            and CellType.SOLID.value == f[i, j, k]
        )

    @ti.func
    def is_air(self, f, i, j, k, nx, ny, nz):
        return (
            i >= 0
            and i < nx
            and j >= 0
            and j < ny
            and k >= 0
            and k < nz
            and CellType.AIR.value == f[i, j, k]
        )

    @ti.func
    def neighbor_sum(self, L, x, f, i, j, k, nx, ny, nz):
        ret = x[(i - 1 + nx) % nx, j, k] * L[i, j, k][2]
        ret += x[(i + 1 + nx) % nx, j, k] * L[i, j, k][3]
        ret += x[i, (j - 1 + ny) % ny, k] * L[i, j, k][4]
        ret += x[i, (j + 1 + ny) % ny, k] * L[i, j, k][5]
        ret += x[i, j, (k - 1 + nz) % nz] * L[i, j, k][6]
        ret += x[i, j, (k + 1 + nz) % nz] * L[i, j, k][7]
        return ret

    # -L matrix : 0-diagonal, 1-diagonal inverse, 2...-off diagonals
    @ti.kernel
    def init_L(self, l: ti.template()):
        _nx, _ny, _nz = self.nx // (2 ** l), self.ny // (2 ** l), self.nz // (2 ** l)
        for i, j, k in self.L[l]:
            if CellType.SAND.value == self.f[l][i, j, k]:
                s = 6.0
                s -= float(self.is_solid(self.f[l], i - 1, j, k, _nx, _ny, _nz))
                s -= float(self.is_solid(self.f[l], i + 1, j, k, _nx, _ny, _nz))
                s -= float(self.is_solid(self.f[l], i, j - 1, k, _nx, _ny, _nz))
                s -= float(self.is_solid(self.f[l], i, j + 1, k, _nx, _ny, _nz))
                s -= float(self.is_solid(self.f[l], i, j, k - 1, _nx, _ny, _nz))
                s -= float(self.is_solid(self.f[l], i, j, k + 1, _nx, _ny, _nz))
                self.L[l][i, j, k][0] = s
                self.L[l][i, j, k][1] = 1.0 / s
            self.L[l][i, j, k][2] = float(
                self.is_fluid(self.f[l], i - 1, j, k, _nx, _ny, _nz)
            )
            self.L[l][i, j, k][3] = float(
                self.is_fluid(self.f[l], i + 1, j, k, _nx, _ny, _nz)
            )
            self.L[l][i, j, k][4] = float(
                self.is_fluid(self.f[l], i, j - 1, k, _nx, _ny, _nz)
            )
            self.L[l][i, j, k][5] = float(
                self.is_fluid(self.f[l], i, j + 1, k, _nx, _ny, _nz)
            )
            self.L[l][i, j, k][6] = float(
                self.is_fluid(self.f[l], i, j, k - 1, _nx, _ny, _nz)
            )
            self.L[l][i, j, k][7] = float(
                self.is_fluid(self.f[l], i, j, k + 1, _nx, _ny, _nz)
            )

    def solve(self, x, rhs):
        tol = 1e-12

        self.r[0].copy_from(rhs)
        self.x.fill(0.0)

        self.Ap.fill(0.0)
        self.p.fill(0.0)

        for l in range(1, self.n_mg_levels):
            self.downsample_f(
                self.f[l - 1],
                self.f[l],
                self.nx // (2 ** l),
                self.ny // (2 ** l),
                self.nz // (2 ** l),
            )
        for l in range(self.n_mg_levels):
            self.L[l].fill(0.0)
            self.init_L(l)

        self.sum[None] = 0.0
        self.reduction(self.r[0], self.r[0])
        initial_rTr = self.sum[None]

        print(f"init rtr = {initial_rTr}")

        if initial_rTr < tol:
            print(f"converged: init rtr = {initial_rTr}")
        else:
            # r = b - Ax = b    since x = 0
            # p = r = r + 0 p
            if self.use_multigrid:
                self.apply_preconditioner()
            else:
                self.z[0].copy_from(self.r[0])

            self.update_p()

            self.sum[None] = 0.0
            self.reduction(self.z[0], self.r[0])
            old_zTr = self.sum[None]

            iter = 0
            for i in range(self.max_iters):
                # alpha = rTr / pTAp
                self.apply_L(0, self.p, self.Ap)

                self.sum[None] = 0.0
                self.reduction(self.p, self.Ap)
                pAp = self.sum[None]

                self.alpha[None] = old_zTr / pAp

                # x = x + alpha p
                # r = r - alpha Ap
                self.update_x_and_r()

                # check for convergence
                self.sum[None] = 0.0
                self.reduction(self.r[0], self.r[0])
                rTr = self.sum[None]
                if rTr < initial_rTr * tol:
                    break

                # z = M^-1 r
                if self.use_multigrid:
                    self.apply_preconditioner()
                else:
                    self.z[0].copy_from(self.r[0])

                # beta = new_rTr / old_rTr
                self.sum[None] = 0.0
                self.reduction(self.z[0], self.r[0])
                new_zTr = self.sum[None]

                self.beta[None] = new_zTr / old_zTr

                # p = z + beta p
                self.update_p()
                old_zTr = new_zTr

                iter = i
            print(f"converged to {rTr} in {iter} iters")

        x.copy_from(self.x)

    @ti.kernel
    def apply_L(self, l: ti.template(), x: ti.template(), Ax: ti.template()):
        _nx, _ny, _nz = self.nx // (2 ** l), self.ny // (2 ** l), self.nz // (2 ** l)
        for i, j, k in Ax:
            if CellType.SAND.value == self.f[l][i, j, k]:
                r = x[i, j, k] * self.L[l][i, j, k][0]
                r -= self.neighbor_sum(self.L[l], x, self.f[l], i, j, k, _nx, _ny, _nz)
                Ax[i, j, k] = r

    @ti.kernel
    def reduction(self, p: ti.template(), q: ti.template()):
        for I in ti.grouped(p):
            if CellType.SAND.value == self.f[0][I]:
                self.sum[None] += p[I] * q[I]

    @ti.kernel
    def update_x_and_r(self):
        a = float(self.alpha[None])
        for I in ti.grouped(self.p):
            if CellType.SAND.value == self.f[0][I]:
                self.x[I] += a * self.p[I]
                self.r[0][I] -= a * self.Ap[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            if CellType.SAND.value == self.f[0][I]:
                self.p[I] = self.z[0][I] + self.beta[None] * self.p[I]

    # ------------------ multigrid ---------------
    @ti.kernel
    def downsample_f(
        self,
        f_fine: ti.template(),
        f_coarse: ti.template(),
        nx: ti.template(),
        ny: ti.template(),
        nz: ti.template(),
    ):
        for i, j, k in f_coarse:
            i2 = i * 2
            j2 = j * 2
            k2 = k * 2

            if (
                CellType.AIR.value == f_fine[i2, j2, k2]
                or CellType.AIR.value == f_fine[i2 + 1, j2, k2]
                or CellType.AIR.value == f_fine[i2, j2 + 1, k2]
                or CellType.AIR.value == f_fine[i2, j2, k2 + 1]
                or CellType.AIR.value == f_fine[i2 + 1, j2 + 1, k2]
                or CellType.AIR.value == f_fine[i2 + 1, j2, k2 + 1]
                or CellType.AIR.value == f_fine[i2, j2 + 1, k2 + 1]
                or CellType.AIR.value == f_fine[i2 + 1, j2 + 1, k2 + 1]
            ):
                f_coarse[i, j, k] = CellType.AIR.value
            else:
                if (
                    CellType.SAND.value == f_fine[i2, j2, k2]
                    or CellType.SAND.value == f_fine[i2 + 1, j2, k2]
                    or CellType.SAND.value == f_fine[i2, j2 + 1, k2]
                    or CellType.SAND.value == f_fine[i2, j2, k2 + 1]
                    or CellType.SAND.value == f_fine[i2 + 1, j2 + 1, k2]
                    or CellType.SAND.value == f_fine[i2 + 1, j2, k2 + 1]
                    or CellType.SAND.value == f_fine[i2, j2 + 1, k2 + 1]
                    or CellType.SAND.value == f_fine[i2 + 1, j2 + 1, k2 + 1]
                ):
                    f_coarse[i, j, k] = CellType.SAND.value
                else:
                    f_coarse[i, j, k] = CellType.SOLID.value

    @ti.kernel
    def restrict(self, l: ti.template()):
        _nx, _ny, _nz = self.nx // (2 ** l), self.ny // (2 ** l), self.nz // (2 ** l)
        for i, j, k in self.r[l]:
            if CellType.SAND.value == self.f[l][i, j, k]:
                Az = self.L[l][i, j, k][0] * self.z[l][i, j, k]
                Az -= self.neighbor_sum(
                    self.L[l], self.z[l], self.f[l], i, j, k, _nx, _ny, _nz
                )
                res = self.r[l][i, j, k] - Az
                self.r[l + 1][i // 2, j // 2, k // 2] += res

    @ti.kernel
    def prolongate(self, l: ti.template()):
        for I in ti.grouped(self.z[l]):
            self.z[l][I] += self.z[l + 1][I // 2]

    # Gause-Seidel
    @ti.kernel
    def smooth(self, l: ti.template(), phase: ti.template()):
        _nx, _ny, _nz = self.nx // (2 ** l), self.ny // (2 ** l), self.nz // (2 ** l)
        for i, j, k in self.r[l]:
            if CellType.SAND.value == self.f[l][i, j, k] and (i + j) & 1 == phase:
                self.z[l][i, j, k] = (
                    self.r[l][i, j, k]
                    + self.neighbor_sum(
                        self.L[l], self.z[l], self.f[l], i, j, k, _nx, _ny, _nz
                    )
                ) * self.L[l][i, j, k][1]

    def apply_preconditioner(self):

        self.z[0].fill(0)
        for l in range(self.n_mg_levels - 1):
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 0)
                self.smooth(l, 1)
            self.z[l + 1].fill(0)
            self.r[l + 1].fill(0)
            self.d[l].fill(0.0)
            self.restrict(l)

        for i in range(self.bottom_smoothing // 2):
            self.smooth(self.n_mg_levels - 1, 0)
            self.smooth(self.n_mg_levels - 1, 1)
        for i in range(self.bottom_smoothing // 2):
            self.smooth(self.n_mg_levels - 1, 1)
            self.smooth(self.n_mg_levels - 1, 0)

        for l in reversed(range(self.n_mg_levels - 1)):
            self.prolongate(l)
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 1)
                self.smooth(l, 0)
