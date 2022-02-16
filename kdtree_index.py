import itertools
import xarray as xr
import numpy as np

from scipy.spatial import KDTree


class KDTreeIndex(xr.core.indexes.Index):
    def __init__(self, data, names, dims, **options):
        self.names = names
        self.dims = dims
        self.shape = data.shape
        self.kdtree = KDTree(data.reshape(-1, self.shape[-1]), **options)

    @classmethod
    def from_variables(cls, variables, **options):
        data = np.concatenate(
            [var.data[..., None] for var in variables.values()], axis=-1
        )
        dims = {var.dims for var in variables.values()}
        if len(dims) != 1:
            raise ValueError("variables need to have the same dimensions")
        dims, = dims
        names = list(variables.keys())
        return cls(data, names, dims, **options)

    def query(self, points):
        distances, indices = self.kdtree.query(points)
        return np.unravel_index(indices, self.shape[:-1])

    def sel(self, indexers):
        unknown_dimensions = set(indexers) - set(self.names)
        if unknown_dimensions:
            raise ValueError("unknown dimensions:", list(unknown_dimensions))

        points = np.concatenate(
            [indexers[name][..., None] for name in self.names],
            axis=-1,
        )

        indices = self.query(points)
        isel_indexers = {
            dim: xr.DataArray(data, dims="points")
            for dim, data in zip(self.dims, indices)
        }

        return isel_indexers
