import numpy

class RandomChoiceGenerator:
    def __init__(self, seed=2023):
        self.rng = numpy.random.default_rng(seed)
    
    def choice(self, p):
        if type(p) is not numpy.ndarray:
            p = p.get()
        return int(self.rng.choice(len(p), 1, p=p)[0])
