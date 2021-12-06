from typing import List
import numpy as np
import open3d as o3d


from macgrid import MacGrid, Particle


class ParticleVisualization(object):
    def __init__(self, mac_grid: MacGrid, scale: float):
        self.scale = scale
        self.mac_grid = mac_grid
        self.point_cloud = o3d.geometry.PointCloud()
        self.update_particles()

    def update_particles(self):
        particle_positions = self.mac_grid.particle_pos.to_numpy().reshape(
            ((self.mac_grid.grid_size * 2) ** 3, 3)
        )
        filtered_pos = [
            self.scale * x
            for x in particle_positions
            if not np.array_equal(x, np.array([0.0, 0.0, 0.0]))
        ]
        self.point_cloud.points = o3d.utility.Vector3dVector(np.array(filtered_pos))
