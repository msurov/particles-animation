import numpy as np
from typing import Tuple
from .particles_parameters import ParticlesParameters


def pack_state(positions : np.ndarray, velocities : np.ndarray) -> np.ndarray:
  R"""
    pack particles positions and velocities into 1-d array
  """
  return np.concatenate((
    np.reshape(positions, (-1,)),
    np.reshape(velocities, (-1,))
  ))

def unpack_state(state : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  R"""
    unpack 1-d array to particles positions and velocities
  """
  n = len(state)
  positions = np.reshape(state[:n//2], (-1, 2))
  velocities = np.reshape(state[n//2:], (-1, 2))
  return positions, velocities

class ParticlesDynamics:
  R"""
    Dynamics of particles in box
  """

  def __init__(self, par : ParticlesParameters):
    self.nparticles = len(par.particles)
    self.eps = 1e-8
    self.elasticity_coef = par.elasticity_coef
    self.gravity_accel = par.gravity_accel
    self.masses = np.array([part.mass for part in par.particles], float)
    self.radiuses = np.array([part.radius for part in par.particles], float)
    self.ex, self.ey = np.eye(2)

  def compute_forces(self, positions : np.ndarray) -> np.ndarray:
    forces = np.zeros((self.nparticles, 2))

    for i in range(self.nparticles):
      ri = self.radiuses[i]
      mi = self.masses[i]
      forces[i] += -mi * self.gravity_accel * self.ey

      # collisions
      for j in range(i + 1, self.nparticles):
        l = positions[i] - positions[j]
        nl = np.linalg.norm(l)
        rj = self.radiuses[j]
        d = nl - ri - rj
        if d < 0:
          d = -d
          f = self.elasticity_coef * d / (ri + rj - d)
          direction = l / (np.linalg.norm(l) + self.eps)
          f = f * direction
          forces[i,:] += f
          forces[j,:] -= f

      # left wall
      d = positions[i][0] - ri
      if d < 0:
        d = abs(d)
        f = self.elasticity_coef * d / max(ri - d, self.eps)
        forces[i] += f * self.ex

      # right wall
      d = 1 - positions[i][0] - ri
      if d < 0:
        d = abs(d)
        f = self.elasticity_coef * d / max(ri - d, self.eps)
        forces[i] += f * -self.ex

      # floor
      d = positions[i][1] - ri
      if d < 0:
        d = abs(d)
        f = self.elasticity_coef * d / max(ri - d, self.eps)
        forces[i] += f * self.ey

    return forces

  def __call__(self, _, st : np.ndarray) -> np.ndarray:
    positions, velocities = unpack_state(st)
    forces = self.compute_forces(positions)
    accelerations = forces / self.masses[:,np.newaxis]
    dst = pack_state(velocities, accelerations)
    return dst
