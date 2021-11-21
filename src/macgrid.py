from typing import Tuple
import taichi as ti
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from RK2 import runge_kutta_2

# ti.init(debug=True, arch=ti.cpu)
ti.init(arch=ti.cpu)
#ti.init(arch=ti.gpu)

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
        grid = sMACGrid()
        
        (self.x, self.y, self.z) = runge_kutta_2(pos = pos, vel_field = grid)
        return None
    
    
class InvalidIndexError(Exception):
    pass


class sMACGrid:
    def __init__(self, domain:int, scale:float) -> None:
        #size of the cubic simulation domain
        self.domain = domain
        #determines the voxel size
        self.scale = scale
        #grid that stores velocity and pressure attributes
        self.grid_size = int(domain * scale)
        #Velocity is stored at the faces of the cell/voxel along the corresponding axis
        self.velX_grid = ti.field(ti.f32,shape=(self.grid_size+1,self.grid_size,self.grid_size))
        self.velY_grid = ti.field(ti.f32,shape=(self.grid_size,self.grid_size+1,self.grid_size))
        self.velZ_grid = ti.field(ti.f32,shape=(self.grid_size,self.grid_size,self.grid_size+1))
        #Pressure is sampled at the cell center
        self.pressure_grid = ti.field(ti.f32,shape=(self.grid_size,self.grid_size,self.grid_size))

    #Returns index of the voxel in which the given Particle is present
    def get_voxel(self, particle: Particle) -> Tuple[int,int,int]:
        voxel_size = self.scale
        i = particle.x // voxel_size
        j = particle.y // voxel_size
        k = particle.z // voxel_size
        
        if(i >= self.domain or j >= self.domain or k >= self.domain):
            raise InvalidIndexError("Particle is out of domain bounds.")

        return (i,j,k)

    #Transfer values of particle velocities on the grid with weighest nearest neighbour averaging.
    def set_grid_velocity(self, valuesX: np.ndarray, valuesY: np.ndarray, valuesZ: np.ndarray) -> None:
        self.velX_grid.from_numpy(valuesX)
        self.velY_grid.from_numpy(valuesY)
        self.velZ_grid.from_numpy(valuesZ)
        return None

    def trilinear_interpolation(values: list, position: Tuple[int,int,int]) -> float:
        x, y, z = position
        res = values[0] * (1 - x) * (1 - y) * (1 - z) + \
           values[1] * x * (1 - y) * (1 - z) + \
           values[2] * (1 - x) * y * (1 - z) + \
           values[3] * (1 - x) * (1 - y) * z + \
           values[4] * x * (1 - y) * z + \
           values[5] * (1 - x) * y * z + \
           values[6] * x * y * (1 - z) + \
           values[7] * x * y * z
        return res

    def index_in_bounds(self, i:int, j:int, k:int, grid:ti.lang.field.ScalarField) -> bool:
        shape = grid.shape
        if(i >= shape[0] or j >= shape[1] or k >= shape[2]):
                #raise InvalidIndexError("Index is out of bounds.")
                return False
        return True

    #Returns the velocity evaluated at the center of the voxel (where pressure is stored)
    def sample_velocity_at_center(self, i:int, j:int, k:int) -> Tuple[float,float,float]:
        is_in_bound = self.index_in_bounds(i,j,k,self.pressure_grid)
        if not is_in_bound:
            raise InvalidIndexError("Index is out of bounds.")
        
        velX = 0.5 * (self.velX_grid[i,j,k] + self.velX_grid[i+1,j,k])
        velY = 0.5 * (self.velY_grid[i,j,k] + self.velY_grid[i,j+1,k])
        velZ = 0.5 * (self.velZ_grid[i,j,k] + self.velZ_grid[i,j,k+1])
        return (velX,velY,velZ)

    #Given grid coordinates of a particle return its sample velocity with trilinear interpolation
    def sample_velocity(self, particle: Particle) -> Tuple[float,float,float]:

        def get_sample_points(i,j,k,grid):

            values = np.zeros((2,2,2))
            if(self.index_in_bounds(i,j,k,grid)):
                values[0,0,0] = grid[i,j,k]
            if(self.index_in_bounds(i,j,k+1,grid)):
                values[0,0,1] = grid[i,j,k+1]
            if(self.index_in_bounds(i,j+1,k,grid)):
                values[0,1,0] = grid[i,j+1,k]
            if(self.index_in_bounds(i,j+1,k+1,grid)):
                values[0,1,1] = grid[i,j+1,k]
            if(self.index_in_bounds(i+1,j,k,grid)):
                values[1,0,0] = grid[i,j,k]
            if(self.index_in_bounds(i+1,j,k+1,grid)):
                values[1,0,1] = grid[i,j,k+1]
            if(self.index_in_bounds(i+1,j+1,k,grid)):
                values[1,1,0] = grid[i,j+1,k]
            if(self.index_in_bounds(i+1,j+1,k+1,grid)):
                values[1,1,1] = grid[i,j+1,k]

            return values

        x,y,z = particle.get_position()
        i,j,k = self.get_voxel(particle)

        values_X = get_sample_points(i,j,k,self.velX_grid)
        values_Y = get_sample_points(i,j,k,self.velY_grid)
        values_Z = get_sample_points(i,j,k,self.velZ_grid)
        x_axis = np.linspace(0,self.scale,2)
        y_axis = np.linspace(0,self.scale,2)
        z_axis = np.linspace(0,self.scale,2)

        #Grid-relative coordinates
        y_stag,z_stag = y-0.5*self.scale, z-0.5*self.scale
        gx,gy,gz = x-i*self.scale, y_stag-j*self.scale, z_stag-k*self.scale
        interpolated = RegularGridInterpolator(points=(x_axis,y_axis,z_axis),values=values_X)([gx,gy,gz])[0]

        x_stag,z_stag = x-0.5*self.scale, z-0.5*self.scale
        gx,gy,gz = x_stag-i*self.scale, y-j*self.scale, z_stag-k*self.scale
        interpolated_Y = RegularGridInterpolator(points=(x_axis,y_axis,z_axis),values=values_Y)([gx,gy,gz])[0]

        x_stag,y_stag = x-0.5*self.scale, y-0.5*self.scale
        gx,gy,gz = x_stag-i*self.scale, y_stag-j*self.scale, z-k*self.scale
        interpolated_Z = RegularGridInterpolator(points=(x_axis,y_axis,z_axis),values=values_Z)([gx,gy,gz])[0]
        
        particle_velocity = [interpolated,interpolated_Y,interpolated_Z]
        
        return particle_velocity


    #Update velocity after numerically solving NS equations on the grid.
    def update_velocity(self) -> None:
        return NotImplementedError

    #Given grid coordinates 
    def sample_pressure(self, particle: Particle) -> float:
        i,j,k = self.get_voxel(particle)
        #Currently: This returns the pressure stored within the center of the voxel in which a given particle is present
        pressure = self.pressure_grid[i,j,k]
        return pressure
    
    #Update Pressure
    def update_pressure(self) -> None:
        return NotImplementedError

    #Sample interpolated velocity for (x,y,z) coordinates
    #Used in RK2 method
    def get_interpolated_velocity(pos: Tuple[float,float,float]) -> Tuple[float,float,float]:
        return NotImplementedError

