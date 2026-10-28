import numpy as np
from collections import defaultdict


class MapRatioEstimator:
    _instance = None

    def __init__(self):
        self.data = defaultdict(list)

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def update(self, pos, input_len, output_len):
        if input_len <= 0:
            return

        ratio = output_len / input_len
        self.data[pos].append(ratio)

    def get_ratio(self, pos, percentile=95):
        if pos not in self.data:
            return None
        return np.percentile(self.data[pos], percentile) * 1.1
    
    def reset(self):
        self.data.clear()