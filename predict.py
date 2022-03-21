from sklearn.linear_model import Ridge
import torch
from torch import Tensor
from tqdm import trange


def calc_distance(target, reference):
    return torch.clip(((target - reference) ** 2).mean(axis=-1), 1e-20)


class TrainingRegion:
    def __init__(self, image: torch.Tensor, feature_map: torch.Tensor):
        self.image = image
        self.feature_map = feature_map

    def batch(self, x, y, size=10):
        min_x = max(x - size, 0)
        max_x = min(x + size, self.image.shape[1] - 1)
        min_y = max(y - size, 0)
        pixel_values = torch.cat([
            self.image[min_y:y, min_x:max_x + 1].flatten(),
            self.image[y, min_x:x]
        ])
        features = torch.cat([
            self.feature_map[min_y:y, min_x:max_x + 1].reshape(-1, 160),
            self.feature_map[y, min_x:x]
        ], dim=0)

        return pixel_values, features


class AdaptivePredictor(TrainingRegion):
    def __init__(self, image_shape, alpha=50.0):
        self.image = torch.zeros(image_shape, dtype=torch.uint8)
        self.feature_map = torch.zeros(*image_shape, 160, dtype=torch.float64)
        self.regression = Ridge(alpha)

    def dependability(self, X: Tensor, y: Tensor, weight: Tensor):
        pred = self.regression.predict(X)
        error = (weight * (y - pred) ** 2).sum()

        return (error / weight.sum()) ** (1 / 2)

    def predict(self, x, y, features: Tensor):
        pred_params = torch.tensor([[], []], dtype=torch.float64)
        for batch_size in (30, 50, 80):
            pixel_train, feature_train = self.batch(x, y, batch_size)
            distance = calc_distance(feature_train, features)
            for num_sample in (160, 1600):
                sample_indices = distance.sort()[1][:num_sample]
                pixel_train_ = pixel_train[sample_indices]
                feature_train_ = feature_train[sample_indices]
                distance_ = distance[sample_indices]

                self.regression.fit(feature_train_, pixel_train_, 1 / distance_)
                depend = self.dependability(feature_train_, pixel_train_, 1 / distance_)
                pred = self.regression.predict(features[None, ...])[0]
                pred_params = torch.cat([pred_params, torch.tensor([[pred], [depend]], dtype=torch.float64)], dim=-1)

        return pred_params

    def pred_all_pixel(self, image: Tensor, feature_map: Tensor):
        pred_param_map = torch.zeros(*image.shape, 2, 6, dtype=torch.float64)
        for y in trange(image.shape[0], desc="Predict image"):
            for x in range(image.shape[1]):
                features = feature_map[y, x]
                if (x, y) in [(0, 0), (1, 0)]:
                    pred_param_map[y, x] = torch.zeros(2, 1, dtype=torch.float64)
                else:
                    pred_param_map[y, x] = self.predict(x, y, features)
                pixel_value = image[y, x]
                self.update_pixel(x, y, pixel_value, features)

        return pred_param_map

    def update_pixel(self, x, y, pixel_value, features):
        self.image[y, x] = pixel_value
        self.feature_map[y, x] = features
