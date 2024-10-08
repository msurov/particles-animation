from dynamics import ParticlesParameters
from typing import Callable
import numpy as np
from matplotlib.patches import Circle


Updater = Callable[[float], np.ndarray]

class ParticlesAnimation:
  def __init__(self, ax, parameters : ParticlesParameters, initial_positions : np.ndarray):
    self.circles = self.make_objs(ax, parameters, initial_positions)
    for obj in self.circles:
      ax.add_patch(obj)
  
  def make_objs(self, ax, parameters : ParticlesParameters, positions : np.ndarray):
    circles = []
    for particle, position in zip(parameters.particles, positions):
      c = Circle(position, particle.radius)
      ax.add_patch(c)
      circles.append(c)
    return circles

  def move_objects(self, positions : np.ndarray):
    nballs = len(self.circles)
    assert (nballs, 2) == np.shape(positions)
    for circle, position in zip(self.circles, positions):
      circle.set_center(position)
