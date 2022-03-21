import torch
from torch import Tensor


def clip_exp(x: Tensor):
    return torch.exp(torch.clip(x, -100.0, 100.0))


def mixture_logistic(x: Tensor, mixture_weights: Tensor, center_positions: Tensor, invert_scales: Tensor):
    x = (-1. + 2. / 255. * x)

    sigmoid_x_plus = torch.sigmoid((x + 1. / 255. - center_positions) * invert_scales)
    sigmoid_x_minus = torch.sigmoid((x - 1. / 255. - center_positions) * invert_scales)
    probs = sigmoid_x_plus - sigmoid_x_minus

    sigmoid_255_plus = torch.sigmoid((1 + 1. / 255. - center_positions) * invert_scales)
    sigmoid_0_minus = torch.sigmoid((-1 - 1. / 255. - center_positions) * invert_scales)
    sum_probs = sigmoid_255_plus - sigmoid_0_minus

    return (mixture_weights * probs / sum_probs).sum(axis=-1)


class ModelParamBase:
    def __init__(self, values=None):
        self.init_param = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
        self.range = torch.tensor([[-4.0, 6.0], [-4.0, 6.0], [-10.0, 10.0]])
        if values is not None:
            self.values = values
        else:
            self.values = self.init_param.reshape(-1, 1).repeat(1, 10)


class ModelParamPred:
    def __init__(self, values=None):
        self.init_param = torch.tensor([1.0, 5.0, 0.0], dtype=torch.float64)
        self.range = torch.tensor([[-4.0, 6.0], [0.0, 10.0], [-5.0, 5.0]])
        if values is not None:
            self.values = values
        else:
            self.values = self.init_param.reshape(-1, 1).repeat(1, 6)


class _ProbabilityModel:
    def batch(self, index, *model_param):
        return self.__class__(*[param[index] for param in self.params], *model_param)

    def _selective_softmax(self, x: Tensor):
        return clip_exp(x) * self.flag / (clip_exp(x) * self.flag).sum(axis=-1)[..., None]

    def probability(self, x: Tensor):
        mixture_weights = self.mixture_weights()
        center_positions = self.center_positions()
        invert_scales = self.invert_scales()

        return mixture_logistic(x, mixture_weights, center_positions, invert_scales)

    def distribution(self):
        x = torch.arange(256).reshape(256, 1)

        return self.probability(x)

    def log_prob(self, x: Tensor):
        return -torch.log2(self.probability(x[..., None]) + 1e-06)


class PmodelBase(_ProbabilityModel):
    def __init__(self, basic_param: Tensor, model_param: ModelParamBase,
                 flag=torch.tensor([True] * 10)):
        self.basic_param = basic_param
        self.model_param = model_param
        self.flag = flag
        self.params = [basic_param]
        self.model_params = [model_param]

    def mixture_weights(self):
        log_weights = self.basic_param[..., 0, :]
        alpha, _, _ = self.model_param.values

        return self._selective_softmax(log_weights * alpha)

    def center_positions(self):
        return torch.clip(self.basic_param[..., 1, :], -1.0, 1.0)

    def invert_scales(self):
        log_scales = torch.clip(self.basic_param[..., 2, :], -7.0)
        _, beta, gamma = self.model_param.values

        return torch.clip(clip_exp(-log_scales * beta + gamma), 1e-07)


class PmodelWithPred(_ProbabilityModel):
    def __init__(self, basic_param: Tensor, pred_param: Tensor,
                 model_param_base: ModelParamBase, model_param_pred: ModelParamPred,
                 flag=torch.tensor([True] * 11)):
        self.basic_param = basic_param
        self.pred_param = pred_param
        self.model_param_base = model_param_base
        self.model_param_pred = model_param_pred
        self.flag = flag
        self.params = [basic_param, pred_param]
        self.model_params = [model_param_base, model_param_pred]

    def mixture_weights(self):
        log_weights = self.basic_param[..., 0, :]
        d = self.pred_param[..., 1, :]
        alpha1, _, _ = self.model_param_base.values
        alpha2, _, _ = self.model_param_pred.values

        return self._selective_softmax(torch.cat([log_weights * alpha1, alpha2 * d], dim=-1))

    def center_positions(self):
        center_positions = torch.clip(self.basic_param[..., 1, :], -1.0, 1.0)
        pred = torch.clip((-1. + 2. / 255. * self.pred_param[..., 0, :]), -1.0, 1.0)

        return torch.cat([center_positions, pred], dim=-1)

    def invert_scales(self):
        log_scales = torch.clip(self.basic_param[..., 2, :], -7.0)
        d = self.pred_param[..., 1, :]
        _, beta1, gamma1 = self.model_param_base.values
        _, beta2, gamma2 = self.model_param_pred.values

        return torch.clip(clip_exp(torch.cat([-log_scales * beta1 + gamma1, beta2 + gamma2 * d], dim=-1)), 1e-07)
