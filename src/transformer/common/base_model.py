import pickle
import os
from .layers import BaseLayer
from .config import GPU
from .np import np, to_numpy, to_cupy

class BaseModel(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
    
    def generate(self):
        raise NotImplementedError
    
    def save_params(self, file_name=None):
        if file_name is None:
            file_name = self.__class__.__name__ + '.pkl'

        params = [p.astype(np.float16) for p in self.params]
        if GPU:
            params = [to_numpy(p) for p in params]

        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
    
    def load_params(self, file_name=None):
        if file_name is None:
            file_name = self.__class__.__name__ + '.pkl'

        if '/' in file_name:
            file_name = file_name.replace('/', os.sep)

        if not os.path.exists(file_name):
            raise IOError('No file: ' + file_name)

        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        params = [p.astype('f') for p in params]
        if GPU:
            params = [to_cupy(p) for p in params]

        for i, param in enumerate(self.params):
            param[...] = params[i]
