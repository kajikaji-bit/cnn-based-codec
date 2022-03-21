from io import BytesIO

import fire
import numpy as np
from PIL import Image
import torch
from tqdm import trange

from common import *
from optimize import selective_optimize
from predict import AdaptivePredictor
from probability import *
import rangecoder
from rangecoder import Distribution


class Encoder(rangecoder.Encoder):
    def __init__(self, stream: BytesIO):
        super().__init__(stream)
        self.coding_info = {"Header": 0, "Flag": 0, "Params": 0, "Image": 0}

    def write_header(self, image_shape):
        self.image_height, self.image_width = image_shape
        self.encode(self.image_height, dtype="uint16")
        self.encode(self.image_width, dtype="uint16")
        self.coding_info["Header"] += 32

    def encode_flag(self, flag: Tensor):
        for f in torch.split(flag, [(len(flag) - 1) % 8 + 1] + [8] * ((len(flag) - 1) // 8)):
            index = sum(f * 2 ** torch.flip(torch.arange(len(f)), [0]))
            self.encode(index, dtype="uint8")
            self.coding_info["Flag"] += 8

    def encode_param(self, param, flag=None, precision=8):
        if flag is None:
            flag = torch.tensor([True] * param.values.shape[1])
        for i in range(param.values.shape[0]):
            for j in torch.arange(param.values.shape[1])[flag]:
                self._encode_param(param.values[i, j], param.range[i], precision)

    def _encode_param(self, param, quantize_range, precision=8):
        index = quantize(param, quantize_range, precision)
        self.encode(index, dtype=f"uint{precision}")
        self.coding_info["Params"] += 8
        with torch.no_grad():
            param.copy_(dequantize(index, quantize_range, precision))

    def encode_pixel(self, pixel_value, dist: np.ndarray):
        self.encode(pixel_value, Distribution(dist))
        self.coding_info["Image"] += -np.log2(dist[pixel_value] / dist.sum() + 1e-06)

    def print_coding_info(self):
        file_size = len(self.stream.getvalue()) * 8
        image_size = file_size - self.coding_info["Header"] - self.coding_info["Flag"] - self.coding_info["Params"]
        coding_rate = file_size / self.image_width / self.image_height
        print("-----------------------------")
        print("Coding info")
        print("-----------------------------")
        print("Header:", self.coding_info["Header"], "bits")
        print("Flag:", self.coding_info["Flag"], "bits")
        print("Params:", round(self.coding_info["Params"]), "bits")
        print("Image:", round(image_size), "bits")
        print("Total:", round(file_size), "bits")
        print("Coding rate:", round(coding_rate, 5), "bits/pel")
        print("-----------------------------")


def encode(
    image_dir="/DATABASE/TMW/",
    image_name="camera.pgm",
    output_file="encoded",
    checkpoint="/DATABASE/PixelCNN/checkpoint/params_imagenet.ckpt",
    test=True,
    cache_dir="/DATABASE/PixelCNN/"
):
    stream = BytesIO()
    encoder = Encoder(stream)
    image = torch.from_numpy(np.array(Image.open(image_dir + image_name)))
    encoder.write_header(image.shape)
    print(image_name)
    print("Width:", image.shape[1])
    print("Height:", image.shape[0])

    if test:
        basic_param_map = torch.from_numpy(np.load(f"{cache_dir}basic_params/{image_name[:-4]}.npy"))
        feature_map = torch.from_numpy(np.load(f"{cache_dir}features/{image_name[:-4]}.npy"))
    else:
        from pixelcnn import PixelCNN
        pixelcnn = PixelCNN(image.shape, checkpoint)
        basic_param_map, feature_map = pixelcnn.pred_all_pixel(image)
        np.save(f"{cache_dir}basic_params/{image_name[:-4]}.npy", basic_param_map.numpy())
        pixelcnn.session.close()

    predictor = AdaptivePredictor(image.shape)
    pred_param_map = predictor.pred_all_pixel(image, feature_map)

    model_param_base = ModelParamBase()
    model_param_pred = ModelParamPred()
    flag = torch.tensor([True] * 16)
    pmodel = PmodelWithPred(basic_param_map, pred_param_map, model_param_base, model_param_pred, flag)
    selective_optimize(pmodel, image)

    print("Encode Flag")
    encoder.encode_flag(flag)
    print(torch.arange(len(flag))[flag] + 1)

    print("Encode parameters")
    encoder.encode_param(model_param_base, flag[:10])
    encoder.encode_param(model_param_pred, flag[10:])
    print(model_param_base.values.detach().numpy())
    print(model_param_pred.values.detach().numpy())

    for y in trange(image.shape[0], desc="Encoding image"):
        for x in range(image.shape[1]):
            pixel_value = image[y, x]
            basic_param = basic_param_map[y, x]
            pred_param = pred_param_map[y, x]
            pmodel = PmodelWithPred(basic_param, pred_param,
                                    model_param_base, model_param_pred, flag)
            dist = pmodel.distribution()
            encoder.encode_pixel(pixel_value, dist.detach().numpy())

    encoder.finish_encode()
    encoder.print_coding_info()

    with open(output_file, "wb") as f:
        f.write(stream.getvalue())


if __name__ == "__main__":
    fire.Fire(encode)
