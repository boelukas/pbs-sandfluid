from typing import List
import numpy as np
import open3d as o3d


from macgrid import MacGrid, Particle


class ParticleVisualization(object):
    def __init__(self, mac_grid: MacGrid, scale: float, update_edges: bool):
        self.scale = scale
        self.mac_grid = mac_grid
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud_edge = o3d.geometry.PointCloud()
        self.update_edges = update_edges
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

        if self.update_edges:
            self.mac_grid.particle_edge_pos.copy_from(self.mac_grid.particle_pos)
            self.mac_grid.update_particle_edge_pos()
            edge_positions = self.mac_grid.particle_edge_pos.to_numpy().reshape(
                ((self.mac_grid.grid_size * 2) ** 3, 3)
            )
            filtered_edge = [
                self.scale * x
                for x in edge_positions
                if not np.array_equal(x, np.array([0.0, 0.0, 0.0]))
            ]
            self.point_cloud_edge.points = o3d.utility.Vector3dVector(np.array(filtered_edge))