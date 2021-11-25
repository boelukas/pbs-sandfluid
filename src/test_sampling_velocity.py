from macgrid import sMACGrid, Particle
import taichi as ti
import numpy as np


def test():
    sim_domain = sMACGrid(domain=4,scale=4)
    test_particle = Particle(position=np.array([1.125,1.125,1.125]),velocity=np.zeros(3))
    test_particle2 = Particle(position=np.array([1.625,1.375,1.875]),velocity=np.zeros(3))
    particles = [test_particle,test_particle2]
    grid_size = sim_domain.grid_size
    
    test_velocities_X = np.arange(grid_size**2*(grid_size+1)).reshape(grid_size+1,grid_size,grid_size)
    test_velocities_Y = np.arange(grid_size**2*(grid_size+1)).reshape(grid_size,grid_size+1,grid_size)
    test_velocities_Z = np.arange(grid_size**2*(grid_size+1)).reshape(grid_size,grid_size,grid_size+1)

    sim_domain.set_grid_velocity(valuesX=test_velocities_X,valuesY=test_velocities_Y,valuesZ=test_velocities_Z)
    interpolated_vel = sim_domain.sample_velocity(particles=particles)
    print(interpolated_vel)
    
    sim_domain.grid_to_particles([test_particle,test_particle2])
    print(test_particle.v)
    print(test_particle2.v)
    return

if __name__ == '__main__':
    test()