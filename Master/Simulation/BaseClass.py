
from .common import *


class SimulationBaseClass:


    def simulate_all(self) -> None:
        for i in range(self.steps):
            self.sim_step(i)

    def simulate_step(self) -> None:
        self.sim_step(self.current_step)
