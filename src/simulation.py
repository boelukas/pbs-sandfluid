import taichi as ti
import open3d as o3d
import numpy as np
from macgrid import sMACGrid
from macgrid import Particle
from pressure_solver import PressureSolver
from force_solver import ForceSolver
from particle_visualization import ParticleVisualization

# For debugging
# ti.init(debug=True, arch=ti.cpu)
ti.init(arch=ti.gpu)

@ti.data_oriented
class Simulation(object):
    def __init__(self):
        self.dt = 3e-3
        self.t = 0.0
        self.resolution = (20, 20, 20)
        self.dx = 1.0
        self.paused = True
        self.draw_convex_hull = False
        self.scale = 0.5
        #self.particles = ParticleField(start_pos = ti.Vector((2, 2, 2)), scale = 0.5, shape=(20,20,20))
        self.particles = []
        particle_start_pos = (2, 2, 2)
        for i, j, k in np.ndindex((5, 5, 5)):
            self.particles += [Particle(np.array([particle_start_pos[0] + self.scale * i,
                                particle_start_pos[0] + self.scale * j,
                                particle_start_pos[0] + self.scale * k]), 
                                np.array([0.1, 0.1, 0.1]))]
        self.particles_vis = ParticleVisualization(self.particles)
        self.mac_grid = sMACGrid(domain=self.resolution[0], scale=1)
        self.pressure_solver = PressureSolver(self.mac_grid)
        self.force_solver = ForceSolver(self.mac_grid)

        self.init()

    def init(self):
        self.t = 0.0

    def advance(self, dt: ti.f32, t: ti.f32):
        # TODO: Compute mac_grid.VelX_grid, mac_grid.VelY_grid, mac_grid.VelZ_grid as
        # weighted average over the particle velocities

        # Adds gravity to the fluid 
        # -> velocity changed
        self.force_solver.compute_forces()
        self.force_solver.apply_forces(dt)

        # Ensure the fluid stays incompressible: 
        # Add enough pressure to the fluid to make the velocity field have divergence 0
        # -> velocity changed
        self.pressure_solver.compute_pressure(dt)
        self.pressure_solver.project(dt)

        # TODO: Bring the new velocity to the particles
        self.mac_grid.sample_velocity(self.particles)

        # TODO: Replace with RK2 step
        # Update the particle position with the new velocity by stepping in the velocity direction
        for p in self.particles:
            p.pos += dt * p.v

    def step(self):
        if self.paused:
            return
        self.t += self.dt
        self.advance(
            self.dt,
            self.t
        )
        self.particles_vis.update_particles(self.particles)


def main():
    sim = Simulation()

    # setup gui
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

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
        [[i, 0, 0] for i in range(sim.resolution[1])]
        + [[i, 0, sim.resolution[2]- 1] for i in range(sim.resolution[1])]
        + [[0, 0, i] for i in range(sim.resolution[2])]
        + [[sim.resolution[1] - 1, 0, i] for i in range(sim.resolution[2])]
    )
    lines = [[i, i + sim.resolution[1]]
             for i in range(sim.resolution[1])] + [[i + 2 *sim.resolution[1], i + 2 *sim.resolution[1] + sim.resolution[2]] for i in range(sim.resolution[2])]
    colors = [[0.7, 0.7, 0.7] for i in range(len(lines))]
    ground_plane = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ground_plane.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(ground_plane, True)  # ground plane

    aabb = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.array([0, 0, 0]), max_bound=np.array([sim.resolution[0]-1, sim.resolution[1]-1, sim.resolution[2]-1])
    )
    aabb.color = [0.7, 0.7, 0.7]
    vis.add_geometry(aabb)  # bounding box

    vis.add_geometry(sim.particles_vis.point_cloud)

    if(sim.draw_convex_hull):
        convex_hull = sim.particles_vis.point_cloud.compute_convex_hull()[0]
        convex_hull.orient_triangles()
        vis.add_geometry(convex_hull)

    while True:
        sim.step()

        vis.update_geometry(sim.particles_vis.point_cloud)
        if(sim.draw_convex_hull):
            convex_hull = sim.particles_vis.point_cloud.compute_convex_hull()[0]
            convex_hull.orient_triangles()
            vis.update_geometry(convex_hull)

        if not vis.poll_events():
            break
        vis.update_renderer()

if __name__ == "__main__":
    main()