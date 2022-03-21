from struct import pack, unpack

import numpy as np


RANGE_SIZE = 64
PRECISION = 8
MAX_RANGE = (1 << RANGE_SIZE) - 1
MIN_RANGE = 1 << (RANGE_SIZE - PRECISION * 2)

# pack format strings (https://docs.python.org/ja/3/library/struct.html#format-strings)
data_format = {
    "uint8": ">B",
    "uint16": ">H",
    "uint32": ">I",
    "uint64": ">Q"
}

bytes_size = {
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
    "uint64": 8
}


class EncoderBase:
    def __init__(self, stream):
        self.stream = stream
        self.low = 0
        self.range = MAX_RANGE

    def encode(self, index, dist):
        self.range //= dist.sum_count
        self.low += self.range * dist.cum(index)
        self.range *= dist.count(index)
        self.update_range()

    def update_range(self):
        min_range_top = 1 << (RANGE_SIZE - PRECISION)
        while (self.low ^ (self.low + self.range)) < min_range_top:
            self.put()
            self.range = (self.range << PRECISION) & MAX_RANGE
            self.low = (self.low << PRECISION) & MAX_RANGE
        while self.range < MIN_RANGE:
            self.put()
            self.range = (~self.low << PRECISION) & MAX_RANGE
            self.low = (self.low << PRECISION) & MAX_RANGE

    def put(self):
        low_top = self.low >> (RANGE_SIZE - PRECISION)
        self.stream.write(pack('B', low_top))

    def finish_encode(self):
        for _ in range(0, RANGE_SIZE, PRECISION):
            self.put()
            self.low = (self.low << PRECISION) & MAX_RANGE


class Encoder(EncoderBase):
    def encode(self, index, dist="uniform", dtype="uint8"):
        """
        Parameters
        ----------
        index: int
        dist: "uniform" | Distribution
        dtype: "uint8" | "uint16" | "uint32" | "uint64"
        """
        if dist == "uniform":
            for i in pack(data_format[dtype], index):
                super().encode(i, UniformDistribution(256))
        else:
            super().encode(index, dist)


class DecoderBase:
    def __init__(self, stream):
        self.stream = stream
        self.low = 0
        self.range = MAX_RANGE
        self.code = 0
        for _ in range(0, RANGE_SIZE, PRECISION):
            self.get()

    def decode(self, dist):
        n = dist.sum_count
        self.range //= n
        target = (self.code - self.low) // self.range
        index = dist.index(target)
        self.low += self.range * dist.cum(index)
        self.range *= dist.count(index)
        self.update_range()

        return index

    def update_range(self):
        min_range_top = 1 << (RANGE_SIZE - PRECISION)
        while (self.low ^ (self.low + self.range)) < min_range_top:
            self.get()
            self.range = (self.range << PRECISION) & MAX_RANGE
            self.low = (self.low << PRECISION) & MAX_RANGE
        while self.range < MIN_RANGE:
            self.get()
            self.range = (~self.low << PRECISION) & MAX_RANGE
            self.low = (self.low << PRECISION) & MAX_RANGE

    def get(self):
        self.code = (self.code << PRECISION) & MAX_RANGE
        self.code += unpack('B', self.stream.read(1))[0]


class Decoder(DecoderBase):
    def decode(self, dist="uniform", dtype="uint8") -> int:
        """
        Parameters
        ----------
        dist: "uniform" | Distribution
        dtype: "uint8" | "uint16" | "uint32" | "uint64"
        """
        if dist == "uniform":
            index = 0
            for _ in range(bytes_size[dtype]):
                index <<= 8
                index += super().decode(UniformDistribution(256))
        else:
            index = super().decode(dist)

        return index


class Distribution:
    def __init__(self, counts):
        self._size = len(counts)
        self._counts = np.maximum(np.array(counts) * (1 << 20) / np.sum(counts), 1).astype(int)
        self._cums = np.append(0, np.cumsum(self._counts)[:-1])
        self.sum_count = int(np.sum(self._counts))

    def count(self, index):
        return int(self._counts[index])

    def cum(self, index):
        return int(self._cums[index])

    def index(self, target):
        left, right = 0, self._size - 1
        while left < right:
            mid = (left + right) // 2
            if self.cum(mid + 1) <= target:
                left = mid + 1
            else:
                right = mid

        return left


class UniformDistribution(Distribution):
    def __init__(self, size, *args, **kwargs):
        counts = [1] * size
        super().__init__(counts, *args, **kwargs)
