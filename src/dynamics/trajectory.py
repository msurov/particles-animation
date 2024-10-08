import numpy as np
from dataclasses import dataclass


@dataclass
class Trajectory:
  time : np.ndarray
  positions : np.ndarray
  velocities : np.ndarray

  @property
  def nparicles(self):
    return np.shape(self.positions)[1]

  def particle_position(self, iparticle : int) -> np.ndarray:
    return self.positions[:,iparticle,:]

  def particle_velocity(self, iparticle : int) -> np.ndarray:
    return self.positions[:,iparticle,:]

def make_trajectory(time, states):
  nt, ns = np.shape(states)
  nparts = ns // 4
  positions = states[:,0:2*nparts]
  positions = np.reshape(positions, (nt, nparts, 2))
  velocities = states[:,2*nparts:]
  velocities = np.reshape(velocities, (nt, nparts, 2))
  traj = Trajectory(
      time = time,
      positions = positions,
      velocities = velocities
  )
  return traj
