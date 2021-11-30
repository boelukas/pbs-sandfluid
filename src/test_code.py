from macgrid import sMACGrid, Particle
import taichi as ti
import numpy as np


def test_particle_velocity():
    sim_domain = sMACGrid(resolution = 20)
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

def test_splatting():
    sim_domain = sMACGrid(resolution = 20)
    test_particle = Particle(position=np.array([0.5,0.4,0.75]),velocity=np.array([0.5, 1.1, 0.6]))
    test_particle2 = Particle(position=np.array([0.5,0.6,1.25]),velocity=np.array([0.7, 0.9, 0.8]))
    particles = [test_particle,test_particle2]
    grid_size = sim_domain.grid_size
    
    test_velocities_X = np.arange(grid_size**2*(grid_size+1)).reshape(grid_size+1,grid_size,grid_size)
    test_velocities_Y = np.arange(grid_size**2*(grid_size+1)).reshape(grid_size,grid_size+1,grid_size)
    test_velocities_Z = np.arange(grid_size**2*(grid_size+1)).reshape(grid_size,grid_size,grid_size+1)
    sim_domain.set_grid_velocity(valuesX=test_velocities_X,valuesY=test_velocities_Y,valuesZ=test_velocities_Z)

    sim_domain.splat_velocity(particles=particles)
    print("--- Vel X ---")
    print(sim_domain.velX_grid)
    print("--- Vel Y ---")
    print(sim_domain.velY_grid)
    print("--- Vel Z ---")
    print(sim_domain.velZ_grid)
    return

def test_runge_kutta_euler():
    sim_domain = sMACGrid(resolution = 20)
    test_particle  = Particle(position=np.array([0., 0.5, 0.5]),velocity=np.array([0.1, 0.1, 0.1]))
    test_particle2 = Particle(position=np.array([1., 0.5, 0.5]),velocity=np.array([0.1, 0.1, 0.1]))
    test_particle3 = Particle(position=np.array([0.5, 0., 0.5]),velocity=np.array([0.1, 0.1, 0.1]))
    test_particle4 = Particle(position=np.array([0.5, 1., 0.5]),velocity=np.array([0.1, 0.1, 0.1]))
    test_particle5 = Particle(position=np.array([0.5, 0.5, 0.]),velocity=np.array([0.1, 0.1, 0.1]))
    test_particle6 = Particle(position=np.array([0.5, 0.5, 1.]),velocity=np.array([0.1, 0.1, 0.1]))
    particles = [test_particle, test_particle2, test_particle3, test_particle4, test_particle5, test_particle6]

    sim_domain.splat_velocity(particles=particles)

    test_pos = np.array([0.5, 0.5, 0.5])
    vel = sim_domain.runge_kutta_2(test_pos, 3e-1)
    print("RK2: " + str(vel))

    vel = sim_domain.midpoint_euler(test_pos, 3e-1)
    print("Euler: " + str(vel))

    return

if __name__ == '__main__':
    # test_particle_velocity()
    # test_splatting()
    test_runge_kutta_euler()