import numpy as np
import torch


np.set_printoptions(suppress=True, precision=3)


def quantize(param, quantize_range, mu=255, precision=8):
    min, range = quantize_range[0], quantize_range[1] - quantize_range[0]
    param = (param - min) / range * 2 - 1  # [-1, 1]に正規化
    if mu != 0:
        param = torch.sign(param) * torch.log(1 + mu * torch.abs(param)) / torch.log(torch.tensor(1 + mu))
    param = (param + 1) / 2  # [0, 1]に正規化

    return int(torch.clip(param * 2 ** precision, 0, 2 ** precision - 1))


def dequantize(index, quantize_range, mu=255, precision=8):
    min, range = quantize_range[0], quantize_range[1] - quantize_range[0]
    param = torch.tensor((index + 0.5) / 2 ** precision * 2 - 1, dtype=torch.float64)  # [-1, 1]に正規化
    if mu != 0:
        param = torch.sign(param) / mu * ((1 + mu) ** torch.abs(param) - 1)
    param = (param + 1) / 2  # [0, 1]に正規化

    return min + param * range
