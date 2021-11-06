import taichi as ti
import open3d as o3d
import numpy as np
from particle_field import ParticleField

# For debugging
# ti.init(debug=True, arch=ti.cpu)
ti.init(arch=ti.gpu)

@ti.data_oriented
class Simulation(object):
    def __init__(self):
        self.dt = 3e-3
        self.t = 0.0
        self.paused = True
        self.gravity = np.array([0.0, -9.81, 0.0])
        self.particles = ParticleField(start_pos = ti.Vector((0, 5, 0)), scale = 0.5, shape=(6,6,6))
        self.draw_convex_hull = False
        self.init()

    def init(self):
        self.t = 0.0

    @ti.kernel
    def advance(self, dt: ti.f32, t: ti.f32):
        # TODO: Update particle positions self.particles.pos here
        pass

    def step(self):
        if self.paused:
            return
        self.t += self.dt
        self.advance(
            self.dt,
            self.t
        )
        self.particles.update_new_positions()


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
        [[i, 0, -10] for i in range(-10, 11)]
        + [[i, 0, 10] for i in range(-10, 11)]
        + [[-10, 0, i] for i in range(-10, 11)]
        + [[10, 0, i] for i in range(-10, 11)]
    )
    lines = [[i, i + 21]
             for i in range(21)] + [[i + 42, i + 63] for i in range(21)]
    colors = [[0.7, 0.7, 0.7] for i in range(len(lines))]
    ground_plane = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ground_plane.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(ground_plane, True)  # ground plane

    aabb = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.array([-10, 0, -10]), max_bound=np.array([10, 20, 10])
    )
    aabb.color = [0.7, 0.7, 0.7]
    vis.add_geometry(aabb)  # bounding box

    vis.add_geometry(sim.particles.point_cloud)

    if(sim.draw_convex_hull):
        convex_hull = sim.particles.point_cloud.compute_convex_hull()[0]
        convex_hull.orient_triangles()
        vis.add_geometry(convex_hull)

    while True:
        sim.step()

        vis.update_geometry(sim.particles.point_cloud)
        if(sim.draw_convex_hull):
            convex_hull = sim.particles.point_cloud.compute_convex_hull()[0]
            convex_hull.orient_triangles()
            vis.update_geometry(convex_hull)

        if not vis.poll_events():
            break
        vis.update_renderer()

if __name__ == "__main__":
    main()