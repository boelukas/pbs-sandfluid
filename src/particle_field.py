import taichi as ti
import open3d as o3d
import numpy as np
@ti.data_oriented
class ParticleField(object):
    def __init__(self, start_pos = ti.Vector((0, 5, 0)), scale = 1, shape = (3, 3, 3)):    
        self.pos = ti.Vector.field(3, dtype=ti.f32, shape=(np.prod(list(shape)),))
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=(np.prod(list(shape)),))
        pos_pointer = 0
        for i, j, k in np.ndindex(shape):
            self.pos[pos_pointer] = ti.Vector([start_pos[0] + i * scale, start_pos[1] + j * scale, start_pos[2] + k * scale])
            self.velocity[pos_pointer] = ti.Vector([0.1, 0.1, 0.1])
            pos_pointer += 1
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(self.pos.to_numpy())
    
    @ti.func
    def set_pos(self, p, idx=ti.Vector([])):
        self.pos[idx] = p
    
    @ti.func
    def set_velocity(self, v, idx=ti.Vector([])):
        self.velocity[idx] = v

    @ti.func
    def step_in_velocity_direction(self, dt=ti.f32, idx=ti.Vector([])):
        for x in ti.ndrange(self.pos.shape[0]):
            self.pos[x] = self.pos[x] + dt * self.velocity[x]


    def update_new_positions(self):
        self.point_cloud.points = o3d.utility.Vector3dVector(self.pos.to_numpy())


