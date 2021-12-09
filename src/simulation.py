import taichi as ti
import open3d as o3d
import numpy as np
from macgrid import MacGrid
from macgrid import Particle
from pressure_solver import PressureSolver
from reference_pressure_solver import ReferencePressureSolver
from force_solver import ForceSolver
from particle_visualization import ParticleVisualization
from pathlib import Path
from time import gmtime, strftime

# For debugging
# ti.init(debug=True, arch=ti.cpu)
ti.init(arch=ti.gpu)


@ti.data_oriented
class Simulation(object):
    def __init__(self):
        self.dt = 1e-2
        self.t = 0.0
        self.grid_size = 64
        self.dx = 1.0
        self.paused = True
        self.draw_convex_hull = False

        # Set this flag to true to export images for every time step
        self.export_images = True
        self.scale = 1.0
        self.alternative_mac_grid = MacGrid(self.grid_size)
        self.particles_vis = ParticleVisualization(
            self.alternative_mac_grid, self.scale
        )
        self.pressure_solver = PressureSolver(self.alternative_mac_grid)
        self.force_solver = ForceSolver(self.alternative_mac_grid)

    def init(self):
        self.alternative_mac_grid.reset_fields()
        self.t = 0.0

    def advance(self, dt: ti.f32, t: ti.f32):
        # # Compute mac_grid.VelX_grid, mac_grid.VelY_grid, mac_grid.VelZ_grid as
        # # self.mac_grid.splat_velocity(self.particles)
        # self.alternative_mac_grid.v_x.fill(0.0)
        # self.alternative_mac_grid.v_y.fill(0.0)
        # self.alternative_mac_grid.v_z.fill(0.0)

        # self.alternative_mac_grid.particles_to_grid()
        # # print(self.mac_grid.velY_grid)

        # # Adds gravity to the fluid
        # # -> velocity changed
        # self.force_solver.compute_forces()
        # self.force_solver.apply_forces(dt)
        # # print(self.mac_grid.velY_grid)

        # # Ensure the fluid stays incompressible:
        # # Add enough pressure to the fluid to make the velocity field have divergence 0
        # # -> velocity changed
        # self.pressure_solver.compute_pressure(dt)
        # # print(
        # #     np.min(self.alternative_mac_grid.pressure.to_numpy().transpose(1, 0, 2)),
        # #     np.max(self.alternative_mac_grid.pressure.to_numpy().transpose(1, 0, 2)),
        # # )
        # self.pressure_solver.project(dt)
        # # self.alternative_mac_grid.show_divergence()
        # # self.alternative_mac_grid.show_pressure()

        # # Apply boundary conditions so that particles do not disappear out of the domain
        # self.alternative_mac_grid.neumann_boundary_conditions()

        # # Bring the new velocity to the particles
        # # self.mac_grid.grid_to_particles(self.particles)
        # self.alternative_mac_grid.grid_to_particles()

        # # TODO: Replace with RK2 step
        # # Update the particle position with the new velocity by stepping in the velocity direction

        # self.alternative_mac_grid.advect_particles_midpoint(dt)
        # # self.alternative_mac_grid.advect_particles(dt)

        # # Re-Mark the cells after advection step into SOLID, SAND or AIR
        # self.alternative_mac_grid.update_cell_types()
        # # self.alternative_mac_grid.print_particles()

        ## FLIP
        self.force_solver.compute_forces()
        self.force_solver.apply_forces(dt)
        self.pressure_solver.compute_pressure(dt)
        self.pressure_solver.project(dt)
        self.alternative_mac_grid.neumann_boundary_conditions()
        self.alternative_mac_grid.update_from_grid()
        self.alternative_mac_grid.advect_particles_midpoint(dt)
        self.alternative_mac_grid.update_cell_types()

        self.alternative_mac_grid.particles_to_grid()
        self.alternative_mac_grid.save_velocities()

    def step(self):
        if self.paused:
            return
        self.t += self.dt
        self.advance(self.dt, self.t)
        self.particles_vis.update_particles()
        # self.alternative_mac_grid.show_pressure()

        # self.alternative_mac_grid.show_divergence()


def main():
    sim = Simulation()

    # setup gui
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    if sim.export_images:
        time = strftime("%Y-%m-%d_%H_%M_%S", gmtime())
        picture_dir = "./sim_" + time
        Path(picture_dir).mkdir(parents=True, exist_ok=True)
        with open(picture_dir + "/sim_settings", "w") as f:
            f.write("Simulation Settings:\n")
            f.write("Grid Size: {}\n".format(sim.grid_size))
            f.write("dt: {}\n".format(sim.dt))
            f.write(
                "Particle (x, y, z) range: {}\n".format(
                    sim.alternative_mac_grid.initial_sand_cells
                )
            )
            f.write("Pressure solver: {}\n".format(sim.pressure_solver.name))
            f.write("Pressure solver density: {}\n".format(sim.pressure_solver.density))
            f.write(
                "Pressure solver gauss seidel min accuracy: {}\n".format(
                    sim.pressure_solver.gaus_seidel_min_accuracy
                )
            )
            f.write(
                "Pressure solver gauss seidel max iterations: {}\n".format(
                    sim.pressure_solver.gaus_seidel_max_iterations
                )
            )
            f.write(
                "Flip viscosity: {}\n".format(sim.alternative_mac_grid.flip_viscosity)
            )

    def init(vis):
        print("reset simulation")
        sim.init()
        vis.reset_view_point(True)

    def pause(vis):
        sim.paused = not sim.paused

    vis.register_key_callback(ord("R"), init)
    vis.register_key_callback(ord(" "), pause)  # space

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0]
    )
    vis.add_geometry(coordinate_frame)  # coordinate frame

    points = (
        [[i * sim.scale, 0, 0] for i in range(sim.grid_size + 1)]
        + [
            [i * sim.scale, 0, (sim.grid_size) * sim.scale]
            for i in range(sim.grid_size + 1)
        ]
        + [[0, 0, i * sim.scale] for i in range(sim.grid_size + 1)]
        + [
            [(sim.grid_size) * sim.scale, 0, i * sim.scale]
            for i in range(sim.grid_size + 1)
        ]
    )
    lines = [[i, i + (sim.grid_size + 1)] for i in range(sim.grid_size + 1)] + [
        [i + 2 * (sim.grid_size + 1), i + 2 * (sim.grid_size + 1) + (sim.grid_size + 1)]
        for i in range(sim.grid_size + 1)
    ]
    colors = [[0.7, 0.7, 0.7] for i in range(len(lines))]
    ground_plane = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ground_plane.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(ground_plane, True)  # ground plane

    aabb = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.array([0, 0, 0]),
        max_bound=np.array(
            [
                (sim.grid_size) * sim.scale,
                (sim.grid_size) * sim.scale,
                (sim.grid_size) * sim.scale,
            ]
        ),
    )
    aabb.color = [0.7, 0.7, 0.7]
    vis.add_geometry(aabb)  # bounding box

    vis.add_geometry(sim.particles_vis.point_cloud)

    if sim.draw_convex_hull:
        convex_hull = sim.particles_vis.point_cloud.compute_convex_hull()[0]
        convex_hull.orient_triangles()
        vis.add_geometry(convex_hull)
    frame_idx = 0
    while True:
        sim.step()

        vis.update_geometry(sim.particles_vis.point_cloud)
        if sim.draw_convex_hull:
            convex_hull = sim.particles_vis.point_cloud.compute_convex_hull()[0]
            convex_hull.orient_triangles()
            vis.update_geometry(convex_hull)

        if not vis.poll_events():
            break
        vis.update_renderer()
        if not sim.paused and sim.export_images:
            frame = picture_dir + "/frame_" + str(frame_idx).zfill(5) + ".png"
            vis.capture_screen_image(frame, True)
            frame_idx += 1


if __name__ == "__main__":
    main()
