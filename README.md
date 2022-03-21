# PixelCNN++を用いた画像の可逆符号化

- CNNに基づく画像の可逆符号化方式における確率モデルの選択的パラメータ修正の検討(PCSJ2021)
- CNNと適応予測を併用した確率モデリングに基づく画像の可逆符号化(ITE2021)

## Base Method
- [Fast PixelCNN++: speedy image generation](https://github.com/PrajitR/fast-pixel-cnn)

## Setup
```sh
# create environment
$ conda create -n cnn-based-codec python=3.6.8
# activate
$ conda activate cnn-based-codec
# install
$ conda install tensorflow==1.12.0
$ conda install pytorch torchvision -c pytorch
$ conda install scikit-learn -c conda-forge
$ conda install fire -c conda-forge
$ conda install tqdm
```

## Using example
```sh
# Encode
$ python encode.py --image_name "lena.pgm" --output_file "encoded"
# Decode
$ python decode.py --input_file "encoded" --output_image "decoded.pgm"
```
