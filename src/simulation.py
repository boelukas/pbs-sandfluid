import taichi as ti
import open3d as o3d
import numpy as np
from macgrid import MacGrid
from pressure_solver import PressureSolver
from force_solver import ForceSolver
from particle_visualization import ParticleVisualization
from pathlib import Path
from time import gmtime, strftime

from sand_solver import SandSolver

# For debugging
# ti.init(debug=True, arch=ti.cpu)
ti.init(arch=ti.gpu)


@ti.data_oriented
class Simulation(object):
    def __init__(self):
        # Simulation parameters
        self.dt = 1e-2
        self.grid_size = 15
        # All cells in the x, y, z range will be marked as sand
        self.initial_sand_cells = ((5, 10), (1, 10), (5, 10))
        self.pic_fraction = 0.0
        self.gaus_seidel_min_accuracy = 0.0001
        self.gaus_seidel_max_iterations = 10000

        # Visualization parameters
        self.paused = True
        self.draw_alpha_surface = False
        self.draw_convex_hull = False
        self.t = 0.0
        self.mesh = None
        self.scale = 1.0
        update_edge_particles = self.draw_alpha_surface or self.draw_convex_hull

        # Set this flag to true to export images for every time step
        self.export_images = True

        # Grid and solvers
        self.mac_grid = MacGrid(
            self.grid_size, self.initial_sand_cells, self.pic_fraction
        )
        # Compute particle on the edges of the object for surface reconstruction
        self.particles_vis = ParticleVisualization(
            self.mac_grid, self.scale, update_edge_particles
        )
        self.pressure_solver = PressureSolver(
            self.mac_grid,
            self.gaus_seidel_min_accuracy,
            self.gaus_seidel_max_iterations,
        )
        self.force_solver = ForceSolver(self.mac_grid)
        self.sand_solver = SandSolver(self.mac_grid)

    def init(self):
        self.mac_grid.reset_fields()
        self.t = 0.0

    def advance(self, dt: ti.f32, t: ti.f32):
        self.mac_grid.particles_to_grid()
        self.mac_grid.save_velocities()

        # Non advection steps of fluid solver
        self.force_solver.compute_forces()
        self.force_solver.apply_forces(dt)
        self.pressure_solver.compute_pressure(dt)
        self.pressure_solver.project(dt)
        self.mac_grid.neumann_boundary_conditions()

        # TODO Fix sand solver
        # self.sand_solver.sand_steps(dt)
        # print(self.alternative_mac_grid.frictional_stress)
        self.mac_grid.grid_to_particles()
        self.mac_grid.advect_particles_midpoint(dt)
        self.mac_grid.update_cell_types()

    def step(self):
        if self.paused:
            return
        self.t += self.dt
        self.advance(self.dt, self.t)
        self.particles_vis.update_particles()
        # self.alternative_mac_grid.show_rigid_cells()

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
                "Particle (x, y, z) range: {}\n".format(sim.mac_grid.initial_sand_cells)
            )
            f.write("Pressure solver: {}\n".format(sim.pressure_solver.name))
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
            f.write("PIC fraction: {}\n".format(sim.mac_grid.pic_fraction))

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

    # pivot_radii = [8.0]
    alpha = 5.0

    if sim.draw_alpha_surface:
        sim.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            sim.particles_vis.point_cloud, alpha
        )
        sim.mesh.orient_triangles()
        vis.add_geometry(sim.mesh)
    elif sim.draw_convex_hull:
        temp_hull = sim.particles_vis.point_cloud.compute_convex_hull()[0]
        temp_hull.orient_triangles()
        sim.convex_hull = temp_hull
        vis.add_geometry(sim.convex_hull)
    else:
        vis.add_geometry(sim.particles_vis.point_cloud)

    frame_idx = 0
    while True:
        sim.step()

        if sim.draw_alpha_surface:
            sim.particles_vis.point_cloud_edge.estimate_normals()
            # temp_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(sim.particles_vis.point_cloud_edge, o3d.utility.DoubleVector(pivot_radii))
            # temp_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(sim.particles_vis.point_cloud_edge, depth=20)
            temp_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                sim.particles_vis.point_cloud_edge, alpha
            )
            temp_mesh.orient_triangles()
            # temp_mesh = temp_mesh.filter_smooth_simple()

            vis.remove_geometry(sim.mesh, False)
            sim.mesh = temp_mesh
            vis.add_geometry(sim.mesh, False)
        elif sim.draw_convex_hull:
            temp_hull = sim.particles_vis.point_cloud.compute_convex_hull()[0]
            temp_hull.orient_triangles()
            vis.remove_geometry(sim.mesh, False)
            sim.mesh = temp_hull
            vis.add_geometry(sim.mesh, False)
        else:
            vis.update_geometry(sim.particles_vis.point_cloud)

        if not vis.poll_events():
            break
        vis.update_renderer()
        if not sim.paused and sim.export_images:
            frame = picture_dir + "/frame_" + str(frame_idx).zfill(5) + ".png"
            vis.capture_screen_image(frame, True)
            frame_idx += 1


if __name__ == "__main__":
    main()
