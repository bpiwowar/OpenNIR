from experimaestro import config, param
from cached_property import cached_property
import numpy as np

@param("seed", default=0)
@config()
class Random:
    @cached_property
    def state(self):
        return np.random.RandomState(self.seed)
