import numpy as np
import pickle


class CorrelationList:
    def __init__(self, shape):
        self._meanx = np.zeros(shape, dtype=float)
        self._meany = np.zeros(shape, dtype=float)
        self._cxy = np.zeros(shape, dtype=float)
        self._m2x = np.zeros(shape, dtype=float)
        self._m2y = np.zeros(shape, dtype=float)
        self._n = np.zeros(shape, dtype=float)

    def __setstate__(self, state):
        # Earlier saved accumulators contain uncentered sums. Preserve their
        # usable statistics when loading without requiring the original traces.
        if '_meanx' not in state:
            n = state['_n']
            meanx = np.divide(state['_sumx'], n, out=np.zeros_like(n), where=n != 0)
            meany = np.divide(state['_sumy'], n, out=np.zeros_like(n), where=n != 0)
            state = {
                '_n': n, '_meanx': meanx, '_meany': meany,
                '_cxy': state['_sumxy'] - state['_sumx'] * meany,
                '_m2x': state['_sumxsq'] - state['_sumx'] * meanx,
                '_m2y': state['_sumysq'] - state['_sumy'] * meany,
            }
        self.__dict__.update(state)

    def __getitem__(self, key):
        if isinstance(key, (int, tuple, list)):
            sx = np.sqrt(np.maximum(self._m2x[key], 0))
            sy = np.sqrt(np.maximum(self._m2y[key], 0))
            corr = np.divide(self._cxy[key], sx, out=np.zeros_like(sx), where=sx > 0)
            corr = np.divide(corr, sy, out=np.zeros_like(sy), where=sy > 0)
            corr = np.clip(corr, -1., 1.)
            # An untouched accumulator has no correlation estimate.
            return np.where(self._n[key] == 0, np.nan, corr)[()]
        if isinstance(key, slice):
            raise NotImplementedError
        else:
            raise TypeError

    def update(self, key, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim != 1 or x.shape != y.shape:
            raise ValueError('Correlation observations must be matching one-dimensional arrays')
        if not x.size:
            return
        meanx, meany = np.mean(x), np.mean(y)
        dx, dy = x - meanx, y - meany
        n = self._n[key]
        total = n + x.size
        shiftx, shifty = meanx - self._meanx[key], meany - self._meany[key]
        weight = n * x.size / total
        self._cxy[key] += np.dot(dx, dy) + shiftx * shifty * weight
        self._m2x[key] += np.dot(dx, dx) + shiftx * shiftx * weight
        self._m2y[key] += np.dot(dy, dy) + shifty * shifty * weight
        self._meanx[key] += shiftx * x.size / total
        self._meany[key] += shifty * x.size / total
        self._n[key] = total

    def merge(self, correlation_array):
        other = correlation_array
        if not isinstance(other, CorrelationList):
            raise TypeError
        total = self._n + other._n
        fraction = np.divide(other._n, total, out=np.zeros_like(total), where=total != 0)
        shiftx, shifty = other._meanx - self._meanx, other._meany - self._meany
        weight = self._n * fraction
        self._cxy += other._cxy + shiftx * shifty * weight
        self._m2x += other._m2x + shiftx * shiftx * weight
        self._m2y += other._m2y + shifty * shifty * weight
        self._meanx += shiftx * fraction
        self._meany += shifty * fraction
        self._n = total

    def save(self):
        with open("/tmp/correlations.p", "wb") as stream:
            pickle.dump(self, stream)
