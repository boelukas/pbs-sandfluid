from typing import Tuple
import taichi as ti
import numpy as np

from RK2 import runge_kutta

class Particle:
    def __init__(self, x:float, y:float, z:float, velocity:Tuple[float,float,float],radius=0.1) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.velocity = velocity
        self.radius = radius

    def get_position(self) -> Tuple[float,float,float]:
        return (self.x, self.y, self.z)

    def get_velocity(self) -> Tuple[float,float,float]:
        return self.velocity

    def get_radius(self) -> float:
        return self.radius
    
    def set_radius(self, radius:float) -> None:
        self.radius = radius
        return None
    
    def update_velocity(self, new_velocity:Tuple[float,float,float]) -> None:
        self.velocity = new_velocity
        return None

    #Update position after updating velocity
    def update_position(self) -> None:
        pos = self.get_position()
        vel_field = sMACGrid().vel_grid
        
        (self.x, self.y, self.z) = runge_kutta(pos = pos, vel_field = vel_field)
        return None
    


#Store all particle instances within here
#Compute averaging over nearest neighbor particles
class ParticleField:
    def __init__(self) -> None:
        pass
    


class sMACGrid:
    def __init__(self, domain:int, scale:float) -> None:
        #size of the cubic simulation domain
        self.domain = domain
        #determines the voxel size
        self.scale = scale
        #grid that stores velocity and pressure attributes
        self.grid_size = int(domain * scale)
        self.vel_grid = np.zeros((self.grid_size+1,self.grid_size+1,self.grid_size+1))
        self.pressure_grid = np.zeros((self.grid_size,self.grid_size,self.grid_size))

    #Returns index of the voxel in which the given Particle is present
    def get_voxel(self, particle: Particle) -> Tuple[int,int,int]:
        return NotImplementedError

    #Given grid coordinates return sample velocity point with trilinear interpolation
    def sample_velocity(self, particle: Particle) -> float:
        return NotImplementedError
    
    #Update velocity
    def update_velocity(self) -> None:
        return NotImplementedError

    #Given grid coordinates 
    def sample_pressure(self, particle: Particle) -> float:
        i,j,k = self.get_voxel(particle)
        #Get coordinates of the center of the voxel
        cx,cy,cz = (i + 1/2) * self.scale, (j + 1/2) * self.scale, (k + 1/2) * self.scale
        #Sample pressure at (Cx,Cy,Cz)
        return NotImplementedError
    
    #Update Pressure
    def update_pressure(self) -> None:
        return NotImplementedError

