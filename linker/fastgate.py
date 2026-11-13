"""Exact numpy evaluator for the merge-gate CatBoost models (oblivious trees, numeric features).

CatBoost's per-call overhead (param JSON parsing + thread pool) dominates single-row predictions
inside the sequential Kruskal replay; this evaluates the same trees in microseconds."""

import json
import os
import tempfile

import numpy as np


class FastModel:
    def __init__(self, cb_model):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        cb_model.save_model(path, format="json")
        js = json.load(open(path))
        os.remove(path)
        ff = js["features_info"]["float_features"]
        flat = [f["flat_feature_index"] for f in ff]  # float_feature_index -> column position
        nan_max = {i for i, f in enumerate(ff) if f.get("nan_value_treatment") == "AsTrue"}
        assert not js["features_info"].get("categorical_features")
        trees = js["oblivious_trees"]
        depth = max(len(t["splits"]) for t in trees)
        T = len(trees)
        self.feat = np.zeros((T, depth), np.int64)
        self.border = np.full((T, depth), np.inf)  # padded splits of shallower trees never fire
        self.leaves = np.zeros((T, 1 << depth))
        for i, t in enumerate(trees):
            for d, sp in enumerate(t["splits"]):
                self.feat[i, d] = flat[sp["float_feature_index"]]
                assert sp["float_feature_index"] not in nan_max
                self.border[i, d] = np.float32(sp["border"])
            lv = np.asarray(t["leaf_values"], float)
            self.leaves[i, : len(lv)] = lv
        self.depth_w = 1 << np.arange(depth)
        sc = js.get("scale_and_bias", [1.0, [0.0]])
        self.scale, self.bias = sc[0], (sc[1][0] if isinstance(sc[1], list) else sc[1])

    def raw(self, x):
        x = np.nan_to_num(
            np.asarray(x, np.float32).ravel(), nan=-np.inf
        )  # float32 like CatBoost; NaN as min
        bits = x[self.feat] > self.border
        idx = (bits * self.depth_w).sum(1)
        return self.scale * self.leaves[np.arange(len(idx)), idx].sum() + self.bias

    def predict_proba(self, x):
        p = 1.0 / (1.0 + np.exp(-self.raw(x)))
        return np.array([[1 - p, p]])


class FastBag:
    def __init__(self, bag):
        self.ms = [FastModel(m) for m in bag.ms]

    def predict_proba(self, x):
        return np.mean([m.predict_proba(x) for m in self.ms], axis=0)


def compile_gates(models):
    return [FastBag(b) for b in models]
