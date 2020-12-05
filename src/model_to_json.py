import json 
from copy import deepcopy
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj,np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def export(model):
    model_copy = deepcopy(model.__dict__)
    with open('model.json', 'w') as outfile:
        json.dump(model_copy,outfile,cls=NumpyEncoder)
        