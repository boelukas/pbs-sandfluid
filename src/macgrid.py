from typing import List, Tuple, Union
import taichi as ti
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# ti.init(debug=True, arch=ti.cpu)
# ti.init(arch=ti.cpu)
ti.init(arch=ti.gpu)

class Particle:
    def __init__(self, position:np.ndarray, velocity:np.ndarray) -> None:
        self.pos = position
        self.v = velocity
    
    
class InvalidIndexError(Exception):
    pass


class sMACGrid:
    def __init__(self, resolution:int) -> None:
        #size of the cubic simulation grid_size
        #determines the number of voxels the grid_size is divided into.
        self.voxel_size = 1.0
        #grid that stores velocity and pressure attributes
        self.grid_size = resolution
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
        
        #Weight factor for SPH Kernel
        self.sph_kernel = 315./(64.*np.pi*(self.voxel_size**9))

    #Returns index of the voxel in which the given Particle is present
    def get_voxel(self, position: np.ndarray) -> Tuple[int,int,int]:
        voxel_size = self.voxel_size
        i = int(position[0] // voxel_size)
        j = int(position[1] // voxel_size)
        k = int(position[2] // voxel_size)
        
        if(i >= self.grid_size or j >= self.grid_size or k >= self.grid_size):
            raise InvalidIndexError("Particle is out of grid_size bounds.")

        return (i,j,k)

    #Returns index of the voxel in which the given Particle is present
    def get_voxel_particle(self, particle: Particle) -> Tuple[int,int,int]:
        voxel_size = self.voxel_size
        i = int(particle.pos[0] // voxel_size)
        j = int(particle.pos[1] // voxel_size)
        k = int(particle.pos[2] // voxel_size)
        
        if(i >= self.grid_size or j >= self.grid_size or k >= self.grid_size):
            raise InvalidIndexError("Particle is out of grid_size bounds.")

        return (i,j,k)

    #Returns the global position of a staggered grid point
    def gridindex_to_position(self, i:int, j:int, k:int, grid:str) -> Tuple[float,float,float]:

        if(grid == "velX"):
            pos = (i*self.voxel_size, (j + 0.5)*self.voxel_size, (k + 0.5)*self.voxel_size)
        elif(grid == "velY"):
            pos = ((i + 0.5)*self.voxel_size, j*self.voxel_size, (k + 0.5)*self.voxel_size)
        elif(grid == "velZ"):
            pos = ((i + 0.5)*self.voxel_size, (j + 0.5)*self.voxel_size, k*self.voxel_size)
        elif(grid == "pressure"):
            pos = ((i + 0.5)*self.voxel_size, (j + 0.5)*self.voxel_size, (k + 0.5)*self.voxel_size)
        else:
            print("No grid specified.")
            return None
            
        return pos

    #Returns index of velocity array to which the given Particle is closest (kernel size = 1 voxel)
    def get_splat_index(self, pos: np.ndarray, grid:str) -> Tuple[int,int,int]:
        voxel_size = self.voxel_size

        if(grid == "velX"):
            i = int(round(pos[0]))
            j = int(round(pos[1]-0.5))
            k = int(round(pos[2]-0.5))
        elif(grid == "velY"):
            i = int(round(pos[0]-0.5))
            j = int(round(pos[1]))
            k = int(round(pos[2]-0.5))
        elif(grid == "velZ"):
            i = int(round(pos[0]-0.5))
            j = int(round(pos[1]-0.5))
            k = int(round(pos[2]))
        else:
            print("No grid specified.")
            return None
            
        
        if(i >= self.grid_size or j >= self.grid_size or k >= self.grid_size):
            raise InvalidIndexError("Particle is out of grid_size bounds.")

        return (i,j,k)

    def weigthed_average(self, particles: List[Tuple[np.ndarray, float]], pos: np.ndarray) -> float:
        weights = []
        vel = []

        for (p, v) in particles:
            vel.append(v)
            dist = np.linalg.norm(p-pos)
            w = self.sph_kernel*(self.voxel_size**2 - dist**2)
            weights.append(w)

        vel = np.asarray(vel)
        weights = np.asarray(weights)
        avg = np.average(a = vel, weights = weights)
        return avg

    #Transfer values of particle velocities on the grid with weighted neighbourhood averaging.
    def splat_velocity(self, particles: List[Particle], small_kernel = True) -> None:
        #initialize array for final grid values
        size = self.grid_size
        valuesX = np.zeros((size+1,size,size))
        valuesY = np.zeros((size,size+1,size))
        valuesZ = np.zeros((size,size,size+1))

        #initialize bins as lists
        binsX = {}
        binsY = {}
        binsZ = {}

        for p in particles:
            pos = p.pos
            (u, v, w) = tuple(p.v)

            if small_kernel:
                #kernel of size 1: only assigns particles to closest gridpoint
                idx = self.get_splat_index(pos, "velX")
                for x in range(max(0, idx[0]-1), min(self.grid_size, idx[0]+1)):
                    for y in range(max(0, idx[1]-1), min(self.grid_size, idx[1]+1)):
                        for z in range(max(0, idx[2]-1), min(self.grid_size, idx[2]+1)):
                            temp =(x, y, z)
                            if(temp in binsX):
                                binsX[temp].append((pos, u))
                            else:
                                binsX[temp] = [(pos, u)]

                idx = self.get_splat_index(pos, "velY")
                for x in range(max(0, idx[0]-1), min(self.grid_size, idx[0]+1)):
                    for y in range(max(0, idx[1]-1), min(self.grid_size, idx[1]+1)):
                        for z in range(max(0, idx[2]-1), min(self.grid_size, idx[2]+1)):
                            temp =(x, y, z)
                            if(temp in binsY):
                                binsY[temp].append((pos, v))
                            else:
                                binsY[temp] = [(pos, v)]

                idx = self.get_splat_index(pos, "velZ")
                for x in range(max(0, idx[0]-1), min(self.grid_size, idx[0]+1)):
                    for y in range(max(0, idx[1]-1), min(self.grid_size, idx[1]+1)):
                        for z in range(max(0, idx[2]-1), min(self.grid_size, idx[2]+1)):
                            temp =(x, y, z)
                            if(temp in binsZ):
                                binsZ[temp].append((pos, w))
                            else:
                                binsZ[temp] = [(pos, w)]

            else:
                #kernel of size 3: also assigns particles to adjecent gridpoints
                idx = self.get_splat_index(pos, "velX")
                raise NotImplementedError

        for (i, j, k) in list(binsX.keys()):
            closest_particles = binsX[(i, j, k)]
            grid_pos = np.asarray(self.gridindex_to_position(i, j, k, "velX"))
            valuesX[i,j,k] = self.weigthed_average(particles = closest_particles, pos = grid_pos)
        for (i, j, k) in list(binsY.keys()):
            closest_particles = binsY[(i, j, k)]
            grid_pos = np.asarray(self.gridindex_to_position(i, j, k, "velY"))
            valuesY[i,j,k] = self.weigthed_average(particles = closest_particles, pos = grid_pos)
        for (i, j, k) in list(binsZ.keys()):
            closest_particles = binsZ[(i, j, k)]
            grid_pos = np.asarray(self.gridindex_to_position(i, j, k, "velZ"))
            valuesZ[i,j,k] = self.weigthed_average(particles = closest_particles, pos = grid_pos)

        self.set_grid_velocity(valuesX, valuesY, valuesZ)

        return None

    #Set new grid velocity
    def set_grid_velocity(self, valuesX: np.ndarray, valuesY: np.ndarray, valuesZ: np.ndarray) -> None:
        self.velX_grid.from_numpy(valuesX)
        self.velY_grid.from_numpy(valuesY)
        self.velZ_grid.from_numpy(valuesZ)
        return None

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

    """
    def trilinear_interpolation(self, values: list, position: Tuple[float,float,float]) -> float:
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
    """

    #Description: Given grid coordinates of a particle return its sample velocity with trilinear interpolation
    #NOTE: The "Particles" argument must contain all particles within a particular grid cell!! 
    def sample_velocity(self, particles: Union[List[Particle],Tuple[float,float,float]],RK2=False) -> Tuple[float,float,float]:

        def get_sample_points(i,j,k,grid):

            
            values = np.zeros(shape=(2,2,2))
            if(self.index_in_bounds(i,j,k,grid)):
                values[0,0,0] = grid[i,j,k]
            if(self.index_in_bounds(i,j,k+1,grid)):
                values[0,0,1] = grid[i,j,k+1]
            if(self.index_in_bounds(i,j+1,k,grid)):
                values[0,1,0] = grid[i,j+1,k]
            if(self.index_in_bounds(i,j+1,k+1,grid)):
                values[0,1,1] = grid[i,j+1,k+1]
            if(self.index_in_bounds(i+1,j,k,grid)):
                values[1,0,0] = grid[i+1,j,k]
            if(self.index_in_bounds(i+1,j,k+1,grid)):
                values[1,0,1] = grid[i+1,j,k+1]
            if(self.index_in_bounds(i+1,j+1,k,grid)):
                values[1,1,0] = grid[i+1,j+1,k]
            if(self.index_in_bounds(i+1,j+1,k+1,grid)):
                values[1,1,1] = grid[i+1,j+1,k+1]

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
                idx_curr = self.get_voxel_particle(particle=p)
                if(idx):
                    assert idx_curr == idx
                else:
                    idx = idx_curr
                particles_position_list.append(p.pos)

        i,j,k = idx
        values_X = get_sample_points(i,j,k,self.velX_grid)
        values_Y = get_sample_points(i,j,k,self.velY_grid)
        values_Z = get_sample_points(i,j,k,self.velZ_grid)
        x_axis = np.linspace(0,self.voxel_size,2)
        y_axis = np.linspace(0,self.voxel_size,2)
        z_axis = np.linspace(0,self.voxel_size,2)

        grid_relative_pos_x = []
        grid_relative_pos_y = []
        grid_relative_pos_z = []
        
        for p_pos in particles_position_list:
            x,y,z = tuple(p_pos)

            #Grid position of velocity sampling grid point of the x component at given index (i,j,k)
            nx,ny,nz = self.gridindex_to_position(i,j,k,"velX")
            #Grid-relative coordinates required for interpolation
            gx,gy,gz = max(0,x-nx), max(0,y-ny), max(0,z-nz)
            grid_relative_pos_x.append([gx,gy,gz])
            
            #Grid position of velocity sampling grid point of the y component at given index (i,j,k)
            nx,ny,nz = self.gridindex_to_position(i,j,k,"velY")
            gx,gy,gz = max(0,x-nx), max(0,y-ny), max(0,z-nz)
            grid_relative_pos_y.append([gx,gy,gz])
            
            #Grid position of velocity sampling grid point of the z component at given index (i,j,k)
            nx,ny,nz = self.gridindex_to_position(i,j,k,"velZ")
            gx,gy,gz = max(0,x-nx), max(0,y-ny), max(0,z-nz)
            grid_relative_pos_z.append([gx,gy,gz])

        interpolated_X = RegularGridInterpolator(points=(x_axis,y_axis,z_axis),values=values_X)(grid_relative_pos_x)
        interpolated_Y = RegularGridInterpolator(points=(x_axis,y_axis,z_axis),values=values_Y)(grid_relative_pos_y)
        interpolated_Z = RegularGridInterpolator(points=(x_axis,y_axis,z_axis),values=values_Z)(grid_relative_pos_z)

        #interpolated_X = self.trilinear_interpolation(values=values_X.flatten().tolist(),position=tuple(grid_relative_pos_x[0]))
        #interpolated_Y = self.trilinear_interpolation(values=values_Y.flatten().tolist(),position=tuple(grid_relative_pos_y[0]))
        #interpolated_Z = self.trilinear_interpolation(values=values_Z.flatten().tolist(),position=tuple(grid_relative_pos_z[0]))

        particle_velocities = [list(vel) for vel in zip(interpolated_X,interpolated_Y,interpolated_Z)]
        return np.array(particle_velocities)


    #Velocity projection step of PIC solver. Update grid velocities after solving pressure equations.
    def grid_to_particles(self, particles:List[Particle]) -> None:
        
        #Gather all particles that belong in the same grid cell
        bins = {}
        for p in particles:
            grid_cell_idx = self.get_voxel_particle(p)
            if(grid_cell_idx in bins):
                bins[grid_cell_idx] += [p]
            else:
                bins[grid_cell_idx] = [p]
        
        #Interpolate velocities for particles per grid cell
        for cell_idx in list(bins.keys()):
            interpolated_velocity = self.sample_velocity(bins[cell_idx])

            #Transfer the interpolated velocity to individual particles
            for i in range(len(bins[cell_idx])):
                particle = bins[cell_idx][i]
                particle.v = interpolated_velocity[i]
        
        return None

    #Given grid coordinates 
    def sample_pressure(self, particle: Particle) -> float:
        i,j,k = self.get_voxel_particle(particle)
        #Currently: This returns the pressure stored within the center of the voxel in which a given particle is present
        pressure = self.pressure_grid[i,j,k]
        return pressure

    def midpoint_euler(self, pos: np.ndarray, step_size: float) -> np.ndarray:
        timestep = step_size/2
        expl_pos = pos + step_size * self.dxdt(pos + step_size/2 * self.sample_velocity(pos, RK2=True)[0], timestep)
        impl_pos = pos + step_size * self.dxdt(0.5 * (pos + expl_pos), timestep)

        return impl_pos

    # Solve Runge-Kutta ODE of second order
    # https://stackoverflow.com/questions/35258628/what-will-be-python-code-for-runge-kutta-second-method

    def runge_kutta_2(
        self,
        pos: np.ndarray,
        dt: float
        ) -> np.ndarray:

        t = np.linspace(0.2*dt, dt, 5)
        n = len(t)
        x = np.array([pos] * n)
        for i in range(n-1):
            h = t[i+1]- t[i] # 1/5 in our case

            k1 = h * self.dxdt(x[i], t[i])
            x[i+1] = x[i] + h * self.dxdt(x[i] + k1, t[i] + h/2.0)

        # return x
        return x[n-1]

    def dxdt(self, x: np.ndarray, t: float) -> np.ndarray:
        # computes velocity at point x at time t given a velocity field

        # use Euler Step Method (implicit/explicit/midpoint) to solve first order ODE
        # TODO: Change to more stable solution
        vel_x = self.sample_velocity(x, RK2=True)
        
        # forward
        y = x + t*vel_x[0]
        dx = self.sample_velocity(y, RK2=True)

        return dx[0]

    @ti.func
    def clear_field(self, target_field: ti.template(), zero_value: ti.template() = 0):
        for x, y, z in ti.ndrange(target_field.shape[0], target_field.shape[1], target_field.shape[2]):
            target_field[x, y, z] = zero_value
