from io import BytesIO

import fire
import numpy as np
from PIL import Image
import torch
from tqdm import trange

from common import *
from predict import AdaptivePredictor
from probability import *
import rangecoder
from rangecoder import Distribution


class Decoder(rangecoder.Decoder):
    def read_header(self):
        image_height = self.decode(dtype="uint16")
        image_width = self.decode(dtype="uint16")

        return (image_height, image_width)

    def decode_flag(self, size):
        byte_size = int(np.ceil(size / 8))
        flag = torch.tensor([], dtype=bool)
        for _ in range(byte_size):
            index = self.decode(dtype="uint8")
            flag = torch.cat([flag, index // torch.flip(2 ** torch.arange(8), [0]) % 2 == 1])

        return flag[-size:]

    def decode_param(self, param, flag=None, precision=8):
        if flag is None:
            flag = torch.tensor([True] * param.values.shape[1])
        for i in range(param.values.shape[0]):
            for j in torch.arange(param.values.shape[1])[flag]:
                param.values[i, j].copy_(self._decode_param(param.range[i], precision))

    def _decode_param(self, quantize_range, precision=8):
        index = self.decode(dtype=f"uint{precision}")

        return dequantize(index, quantize_range, precision)

    def decode_pixel(self, dist: np.ndarray):
        pixel_value = self.decode(Distribution(dist))

        return pixel_value


def decode(
    input_file="encoded",
    output_image="decoded.pgm",
    checkpoint="/DATABASE/PixelCNN/checkpoint/params_imagenet.ckpt",
    test=True,
    image_name="camera.pgm",
    cache_dir="/DATABASE/PixelCNN/"
):
    stream = BytesIO(open(input_file, "rb").read())
    decoder = Decoder(stream)

    image_shape = decoder.read_header()
    image = torch.zeros(image_shape, dtype=torch.uint8)
    print("Width:", image_shape[1])
    print("Height:", image_shape[0])

    if test:
        from pixelcnn import TestingPixelCNN
        basic_param_cache = f"{cache_dir}basic_params/{image_name[:-4]}.npy"
        features_cache = f"{cache_dir}features/{image_name[:-4]}.npy"
        pixelcnn = TestingPixelCNN(basic_param_cache, features_cache)
    else:
        from pixelcnn import PixelCNN
        pixelcnn = PixelCNN(image_shape, checkpoint)

    predictor = AdaptivePredictor(image.shape)

    print("Decode flag")
    flag = decoder.decode_flag(size=16)
    print(torch.arange(len(flag))[flag] + 1)

    print("Decode parameters")
    model_param_base = ModelParamBase()
    model_param_pred = ModelParamPred()
    decoder.decode_param(model_param_base, flag[:10])
    decoder.decode_param(model_param_pred, flag[10:])
    print(model_param_base.values[..., flag[:10]].detach().numpy())
    print(model_param_pred.values[..., flag[10:]].detach().numpy())

    for y in trange(image_shape[0], desc="Decoding image"):
        pixelcnn.update_row(y)
        for x in range(image_shape[1]):
            basic_param, features = pixelcnn.run_cnn(x, y)
            if (x, y) in [(0, 0), (1, 0)]:
                pred_param = torch.zeros(2, 6, dtype=torch.float64)
            else:
                pred_param = predictor.predict(x, y, features)
            pmodel = PmodelWithPred(basic_param, pred_param,
                                    model_param_base, model_param_pred, flag)
            dist = pmodel.distribution()
            pixel_value = torch.tensor(decoder.decode_pixel(dist.numpy()), dtype=torch.uint8)
            image[y, x] = pixel_value
            pixelcnn.update_pixel(x, y, pixel_value)
            predictor.update_pixel(x, y, pixel_value, features)

    Image.fromarray(image.detach().numpy()).save(output_image)


if __name__ == "__main__":
    fire.Fire(decode)
