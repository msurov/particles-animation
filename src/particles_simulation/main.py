from dynamics import (
  ParticlesDynamics,
  ParticlesParameters,
  ParticleParameters,
  pack_state,
  unpack_state
)
from anim import ParticlesAnimation
from typing import Tuple
from sim import Simulator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def gen_random_particles(nparticles : int, radius_diap : Tuple[float,float], mass_diap : Tuple[float,float]):
  masses,radiuses = np.random.rand(2, nparticles)
  masses = masses * (mass_diap[1] - mass_diap[0]) + mass_diap[0]
  radiuses = radiuses * (radius_diap[1] - radius_diap[0]) + radius_diap[0]
  return [ParticleParameters(r, m) for m,r in zip(masses, radiuses)]

def main():
  syspar = ParticlesParameters(
    particles = gen_random_particles(15, [0.01, 0.05], [0.001, 0.01]),
    gravity_accel = 1.0,
    elasticity_coef = 4.
  )
  nparticles = len(syspar.particles)
  dynamics = ParticlesDynamics(syspar)
  initial_positions = np.random.rand(nparticles, 2)
  initial_velocities = np.zeros((nparticles, 2))
  initial_state = pack_state(initial_positions, initial_velocities)
  sim = Simulator(dynamics, 0., initial_state, max_step=1e-2, atol=1e-5, rtol=1e-5)

  fig, ax = plt.subplots(1, 1)
  ax.set_aspect(1)
  ax.grid(True)
  anim = ParticlesAnimation(ax, syspar, initial_positions)
  
  fps = 30
  nframes = 10000

  def updater(iframe):
    t = iframe / fps
    _,state = sim.update(t)
    positions,_ = unpack_state(state)
    anim.move_objects(positions)
    return anim.circles

  a = animation.FuncAnimation(fig, updater, frames=nframes, interval=1000/fps, blit=True)
  plt.show()

if __name__ == '__main__':
  main()
