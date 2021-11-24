from typing import List, Tuple, Union
import taichi as ti
import numpy as np
from scipy.interpolate import RegularGridInterpolator

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
        
        # (self.x, self.y, self.z) = runge_kutta_2(pos = pos, vel_field = grid)
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
        self.dx = 1
        #Velocity is stored at the faces of the cell/voxel along the corresponding axis
        self.velX_grid = ti.field(ti.f32,shape=(self.grid_size+1,self.grid_size,self.grid_size))
        self.velY_grid = ti.field(ti.f32,shape=(self.grid_size,self.grid_size+1,self.grid_size))
        self.velZ_grid = ti.field(ti.f32,shape=(self.grid_size,self.grid_size,self.grid_size+1))
        
        self.forceX_grid = ti.field(ti.f32,shape=(self.grid_size+1,self.grid_size,self.grid_size))
        self.forceY_grid = ti.field(ti.f32,shape=(self.grid_size,self.grid_size+1,self.grid_size))
        self.forceZ_grid = ti.field(ti.f32,shape=(self.grid_size,self.grid_size,self.grid_size+1))
        #Pressure is sampled at the cell center
        self.pressure_grid = ti.field(ti.f32,shape=(self.grid_size,self.grid_size,self.grid_size))
        #Indicates which cell has fluid particles (1) and which not (0)
        #NOTE: Has to be initialized/set after the advection of particles on the grid
        self.has_particles = ti.field(ti.f32,shape=(self.grid_size,self.grid_size,self.grid_size))
        self.divergence_grid = ti.field(ti.f32, shape=(self.grid_size,self.grid_size,self.grid_size))


    #Returns index of the voxel in which the given Particle is present
    def get_voxel(self, position: Tuple[float,float,float]) -> Tuple[int,int,int]:
        voxel_size = self.scale
        x,y,z = position

        i = x // voxel_size
        j = y // voxel_size
        k = z // voxel_size
        
        if(i >= self.domain or j >= self.domain or k >= self.domain):
            raise InvalidIndexError("Particle is out of domain bounds.")

        return (i,j,k)

    #Returns index of the voxel in which the given Particle is present
    def get_voxel_particle(self, particle: Particle) -> Tuple[int,int,int]:
        voxel_size = self.scale
        i = particle.x // voxel_size
        j = particle.y // voxel_size
        k = particle.z // voxel_size
        
        if(i >= self.domain or j >= self.domain or k >= self.domain):
            raise InvalidIndexError("Particle is out of domain bounds.")

        return (i,j,k)

    #Returns the global position of a staggered grid point
    def gridindex_to_position(self, i:int, j:int, k:int, grid:str) -> Tuple[float,float,float]:

        if(grid == "velX"):
            pos = (i*self.scale, (j + 0.5)*self.scale, (k + 0.5)*self.scale)
        elif(grid == "velY"):
            pos = ((i + 0.5)*self.scale, j*self.scale, (k + 0.5)*self.scale)
        elif(grid == "velZ"):
            pos = ((i + 0.5)*self.scale, (j + 0.5)*self.scale, k*self.scale)
        elif(grid == "pressure"):
            pos = ((i + 0.5)*self.scale, (j + 0.5)*self.scale, (k + 0.5)*self.scale)
        else:
            print("No grid specified.")
            return None
            
        return pos

    def weigthed_average(self, particles: list(Tuple[Tuple[float,float,float], float]), pos: Tuple[float,float,float]) -> float:
        #TODO: implement weighted average according to slides
        return None

    #Transfer values of particle velocities on the grid with weighted neighbourhood averaging.
    def splat_velocity(self, particles: list(Particle)) -> None:
        #initialize array for final grid values
        size = self.grid_size
        valuesX = np.ndarray(size+1,size,size)
        valuesY = np.ndarray(size,size+1,size)
        valuesZ = np.ndarray(size,size,size+1)

        #initialize bins as lists
        binsX = [ [ []*size ]*size ] * (size+1)
        binsY = [ [ []*size ]* (size+1) ] * size
        binsZ = [ [ []*(size+1) ]*size ] * size

        for p in particles:
            assert(p is Particle)
            (x, y, z) = p.get_position()
            (u, v, w) = p.get_velocity()

            #kernel of size 1: only assigns particles to closest gridpoint
            (x1, y1, z1)= self.get_X_splat_index((x,y,z))
            binsX[x1][y1][z1].append(((x,y,z), u))

            #TODO: for binsY and binsZ
        
        for i in range(len(binsX)):
            for j in range(len(binsX[i])):
                for k in range(len(binsX[i][j])):
                    particles = binsX[i][j][k]
                    if particles: #len > 0
                        #position of gridpoint is in the middle of x-surface
                        valuesX[i,j,k] = self.weighted_average(particles = particles, pos = (float(i), float(j)-0.5, float(k)-0.5))

        #TODO: for valuesY and valuesZ

        self.set_grid_velocity(valuesX, valuesY, valuesZ)

        return None

    #Set new grid velocity
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

    def index_in_bounds(self, i:int, j:int, k:int, grid:ti.template()) -> bool:
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
        
        #Linearly interpolated between faces of the voxel
        velX = 0.5 * (self.velX_grid[i,j,k] + self.velX_grid[i+1,j,k])
        velY = 0.5 * (self.velY_grid[i,j,k] + self.velY_grid[i,j+1,k])
        velZ = 0.5 * (self.velZ_grid[i,j,k] + self.velZ_grid[i,j,k+1])
        return (velX,velY,velZ)

    #Description: Given grid coordinates of a particle return its sample velocity with trilinear interpolation
    #NOTE: The "Particles" argument must contain all particles within a particular grid cell!! 
    def sample_velocity(self, particles: Union[List[Particle],Tuple[float,float,float]],RK2=False) -> Tuple[float,float,float]:

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

        particles_position_list = []
        idx = None

        if(RK2):
            pos = particles
            particles_position_list.append(pos)
            idx = self.get_voxel(pos)
        else:
            #Extract position and store in list
            for p in particles:
                pos = p.get_position()
                idx_curr = self.get_voxel_particle(particle=p)
                if(idx):
                    assert idx_curr == idx
                else:
                    idx = idx_curr
                particles_position_list.append(pos)

        i,j,k = idx
        values_X = get_sample_points(i,j,k,self.velX_grid)
        values_Y = get_sample_points(i,j,k,self.velY_grid)
        values_Z = get_sample_points(i,j,k,self.velZ_grid)
        x_axis = np.linspace(0,self.scale,2)
        y_axis = np.linspace(0,self.scale,2)
        z_axis = np.linspace(0,self.scale,2)

        grid_relative_pos_x = []
        grid_relative_pos_y = []
        grid_relative_pos_z = []
        
        for p_pos in particles_position_list:
            x,y,z = p_pos

            #Grid position of velocity sampling grid point of the x component at given index (i,j,k)
            nx,ny,nz = self.gridindex_to_position(i,j,k,"velX")
            #Grid-relative coordinates required for interpolation
            gx,gy,gz = x-nx, y-ny, z-nz
            grid_relative_pos_x.append([gx,gy,gz])
            
            #Grid position of velocity sampling grid point of the y component at given index (i,j,k)
            nx,ny,nz = self.gridindex_to_position(i,j,k,"velY")
            gx,gy,gz = x-nx, y-ny, z-nz
            grid_relative_pos_y.append([gx,gy,gz])
            
            #Grid position of velocity sampling grid point of the z component at given index (i,j,k)
            nx,ny,nz = self.gridindex_to_position(i,j,k,"velZ")
            gx,gy,gz = x-nx, y-ny, z-nz
            grid_relative_pos_z.append([gx,gy,gz])
            
        interpolated_X = RegularGridInterpolator(points=(x_axis,y_axis,z_axis),values=values_X)(grid_relative_pos_x)
        interpolated_Y = RegularGridInterpolator(points=(x_axis,y_axis,z_axis),values=values_Y)(grid_relative_pos_y)
        interpolated_Z = RegularGridInterpolator(points=(x_axis,y_axis,z_axis),values=values_Z)(grid_relative_pos_z)

        particle_velocities = list(zip(interpolated_X,interpolated_Y,interpolated_Z))
        return particle_velocities


    #Update velocity after numerically solving NS equations on the grid.
    def update_velocity(self) -> None:
        return NotImplementedError

    #Velocity projection step of PIC solver. Update grid velocities after solving pressure equations.
    def grid_to_particles(self, particles:List[Particle]) -> None:
        
        #Gather all particles that belong in the same grid cell
        bins = {}
        for p in particles:
            grid_cell_idx = self.get_voxel_particle(p)
            bins[grid_cell_idx] += [p]
        
        #Interpolate velocities for particles per grid cell
        for cell_idx in list(bins.keys()):
            interpolated_velocity = self.sample_velocity(bins[cell_idx])

            #Transfer the interpolated velocity to individual particles
            for i in range(len(bins[cell_idx])):
                particle = bins[cell_idx][i]
                vel_update = interpolated_velocity[i]
                particle.update_velocity(vel_update)
        
        return None

    #Given grid coordinates 
    def sample_pressure(self, particle: Particle) -> float:
        i,j,k = self.get_voxel_particle(particle)
        #Currently: This returns the pressure stored within the center of the voxel in which a given particle is present
        pressure = self.pressure_grid[i,j,k]
        return pressure
    
    #Update Pressure
    def update_pressure(self) -> None:
        return NotImplementedError

    # Solve Runge-Kutta ODE of second order
    # https://stackoverflow.com/questions/35258628/what-will-be-python-code-for-runge-kutta-second-method
    
    STEP_SIZE = 1.
    """
        #Template Code
        rk2_position = (1.0,2.0,3.0)
        rk2_vel = self.sample_velocity(rk2_position,RK2=True)
    """

    def runge_kutta_2(
        self,
        pos: Tuple[float,float,float],
        t: np.array() = np.linspace(1/STEP_SIZE, STEP_SIZE, 5)

        ) -> Tuple[float,float,float]:

        n = len(t)
        x = np.array([pos] * n)
        for i in range(n-1):
            h = t[i+1]- t[i] # 1/5 in our case

            k1 = h * self.dxdt(x[i], t[i])
            x[i+1] = x[i] + h * self.dxdt(x[i] + k1, t[i] + h/2.0)

        # return x
        return x[n-1]

    def dxdt(self, x: Tuple[float,float,float], t: float) -> Tuple[float,float,float]:
        # computes velocity at point x at time t given a velocity field

        # use Euler Step Method (implicit/explicit/midpoint) to solve first order ODE
        # TODO: Change to more stable solution
        vel_x = self.trilinear_interpolation(x)
        
        # forward
        y = x + t*vel_x
        dx = self.trilinear_interpolation(y)

        return dx

    @ti.func
    def clear_field(self, target_field: ti.template(), zero_value: ti.template() = 0):
        for x, y, z in ti.ndrange(target_field.shape[0], target_field.shape[1], target_field.shape[2]):
            target_field[x, y, z] = zero_value
