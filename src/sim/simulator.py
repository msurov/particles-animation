from scipy.integrate import ode
import numpy as np
from copy import copy
from dataclasses import dataclass
from typing import Callable, Tuple

class Simulator:
  def __init__(self, 
                sys : Callable[[float, np.ndarray], np.ndarray],
                initial_time : float,
                initial_state : np.ndarray,
                **integrator_args
            ):
    integrator = ode(sys)
    integrator.set_initial_value(initial_state, initial_time)
    integrator.set_integrator('dopri5', **integrator_args)
    self.integrator = integrator

  def update(self, t) -> Tuple[float, np.ndarray]:
    self.integrator.integrate(t)
    self.integrator.y
    assert self.integrator.successful(), f'Can\'t integrate at {t}sec'
    x = self.integrator.y
    t = self.integrator.t
    return t, x
