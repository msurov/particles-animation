from dataclasses import dataclass
from typing import List

@dataclass
class ParticleParameters:
  radius : float
  mass : float

@dataclass
class ParticlesParameters:
  particles : List[ParticleParameters]
  gravity_accel : float
  elasticity_coef : float
