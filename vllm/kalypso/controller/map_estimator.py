import numpy as np
from collections import defaultdict


import contextvars
_QUERY_ESTIMATOR = contextvars.ContextVar("kalypso_query_estimator", default=None)


class MapRatioEstimator:
    _instance = None
    # Output bound = percentile of observed output/input ratios x margin. Defaults are the paper's
    # setting; multi-query tests may override them per query (eval_runs/admit_cfg_<host>).
    percentile = 95
    margin = 1.1

    def __init__(self):
        self.data = defaultdict(list)

    @classmethod
    def instance(cls):
        local = _QUERY_ESTIMATOR.get()  # multi-query mode: per-query estimator
        if local is not None:
            return local
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def update(self, pos, input_len, output_len):
        if input_len <= 0:
            return

        ratio = output_len / input_len
        self.data[pos].append(ratio)

    def get_ratio(self, pos, percentile=None):
        if pos not in self.data:
            return None
        percentile = self.percentile if percentile is None else percentile
        return np.percentile(self.data[pos], percentile) * self.margin
    
    def reset(self):
        self.data.clear()