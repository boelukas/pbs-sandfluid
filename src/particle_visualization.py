from typing import List
import numpy as np
import open3d as o3d

from macgrid import Particle
class ParticleVisualization(object):
    def __init__(self, particles: List[Particle]):    
        self.point_cloud = o3d.geometry.PointCloud()
        self.update_particles(particles)

    def update_particles(self, particles: List[Particle]):
        positions = [p.get_position() for p in particles]
        self.point_cloud.points = o3d.utility.Vector3dVector(np.array(positions))
