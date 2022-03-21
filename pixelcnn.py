import warnings
warnings.simplefilter("ignore", FutureWarning)

import numpy as np
import tensorflow as tf
import torch
from torch.nn.functional import elu
from tqdm import trange

from fast_pixelcnn.model import model_spec


class PixelCNN:
    def __init__(self, image_shape, checkpoint, num_mix=10):
        self.image = np.zeros(image_shape, dtype=np.int64)
        self.num_mix = num_mix

        image_height, image_width = image_shape
        shape = 1, image_height, image_width, 4

        _shift = np.ones(shape, np.float32)
        self._downshift = _shift.copy()
        self._downshift[:, 0, :, :] = 0
        self._rightshift = _shift.copy()
        self._rightshift[:, :, 0, :] = 0

        tf.reset_default_graph()

        self._ph_row_input = tf.placeholder(
            tf.float32, [shape[0], 1, shape[2], shape[3]],
            name='row_input'
        )
        self._ph_pix_input = tf.placeholder(
            tf.float32, [shape[0], 1, 1, shape[3]],
            name='pix_input'
        )
        self._ph_row_id = tf.placeholder(
            tf.int32, [],
            name='row_id'
        )
        self._ph_col_id = tf.placeholder(
            tf.int32, [],
            name='col_id'
        )
        self._fast_nn_out, self._v_stack = \
            tf.make_template('model', model_spec)(
                self._ph_row_input,
                self._ph_pix_input,
                self._ph_row_id,
                self._ph_col_id,
                shape
            )

        self.session = session = tf.Session()

        ema = tf.train.ExponentialMovingAverage(0.9995)
        cache_variables = [
            v for v in tf.global_variables() if 'cache' in v.name
        ]
        session.run(tf.variables_initializer(cache_variables))
        variables_to_restore = {
            k: v
            for k, v in ema.variables_to_restore().items()
            if 'cache' not in k
        }
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(session, checkpoint)

    def update_pixel(self, x, y, pixel_val):
        self.image[y, x] = pixel_val
        pixel_val = (np.cast[np.float32](pixel_val) - 127.5) / 127.5
        self._downshift[:, y + 1:y + 2, x, :3] = pixel_val
        self._rightshift[:, y, x + 1:x + 2, :3] = pixel_val

    def update_row(self, y):
        self.session.run(
            self._v_stack,
            {
                self._ph_row_id: y,
                self._ph_row_input: self._downshift[:, y:y + 1, :, :]
            }
        )

    def run_cnn(self, x, y):
        output, ul = self.session.run(
            self._fast_nn_out,
            {
                self._ph_row_id: y,
                self._ph_col_id: x,
                self._ph_pix_input: self._rightshift[:, y:y + 1, x:x + 1, :]
            }
        )

        output = torch.from_numpy(output[0, 0, 0, :].astype(np.float64))
        log_weights = output[:self.num_mix]
        output = output[self.num_mix:].reshape(3, 3 * self.num_mix)
        center_positions = output[0, :self.num_mix]
        log_scales = output[0, self.num_mix:2 * self.num_mix]

        features = elu(torch.from_numpy(ul[0, 0, 0, :].astype(np.float64)))

        return torch.stack([log_weights, center_positions, log_scales]), features

    def pred_all_pixel(self, image):
        basic_param_map = torch.empty(image.shape[0], image.shape[1], 3, 10, dtype=torch.float64)
        feature_map = torch.empty(image.shape[0], image.shape[1], 160, dtype=torch.float64)
        for y in trange(image.shape[0], desc='CNN part'):
            self.update_row(y)
            for x in range(image.shape[1]):
                basic_param_map[y, x], feature_map[y, x] = self.run_cnn(x, y)
                self.update_pixel(x, y, image[y, x])

        return basic_param_map, feature_map


class TestingPixelCNN:
    def __init__(self, basic_param_cache: str, features_cache: str):
        self.basic_param_map = torch.from_numpy(np.load(basic_param_cache))
        self.feature_map = torch.from_numpy(np.load(features_cache))

    def update_pixel(self, x, y, pixel_val):
        pass

    def update_row(self, y):
        pass

    def run_cnn(self, x, y):
        return self.basic_param_map[y, x], self.feature_map[y, x]

    def pred_all_pixel(self, image):
        return self.basic_param_map, self.feature_map
