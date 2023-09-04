import numpy
from random import Random

class RandomChoiceGenerator:
    def __init__(self, seed=2023):
        self.gen = Random(seed)

    def choice(self, p):
        if type(p) is not numpy.ndarray:
            # expect cupy.ndarray
            p = p.get()
        return self.gen.choices(range(len(p)), p, k=1)[0]
