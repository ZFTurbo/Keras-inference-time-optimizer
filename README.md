# Keras inference time optimizer (KITO)

This code takes on input trained Keras model and optimize layer structure and weights in such a way that model became 
much faster (~10-30%), but works identically to initial model. It can be extremely useful in case you need to process large 
amount of images with trained model. Reduce operation was tested on all Keras models zoo. See 
comparison table below.

## Installation

```
pip install kito
```

## How it works?
 
In current version it only apply single type of optimization: It reduces Conv2D + BatchNormalization set of layers to 
single Conv2D layer. Since Conv2D + BatchNormalization is very common set of layers, optimization works well 
almost on all modern CNNs for image processing.

Also supported:
* DepthwiseConv2D + BatchNormalization => DepthwiseConv2D 
* SeparableConv2D + BatchNormalization => SeparableConv2D
* Conv2DTranspose + BatchNormalization => Conv2DTranspose
* Conv3D + BatchNormalization => Conv3D

## How to use

Typical code:

```
model.fit(...)
...
model.predict(...)
```

must be replaced with following block:

```
from kito import reduce_keras_model
model.fit(...)
...
model_reduced = reduce_keras_model(model)
model_reduced.predict(...)
```

So basically you need to insert 2 lines in your code to speed up operations. But note that it requires 
some time to convert model. You can see usage example in [test_bench.py](https://github.com/ZFTurbo/Keras-inference-time-optimizer/blob/master/kito/test_bench.py)

## Comparison table

| Neural net | Input shape | Number of layers (Init) | Number of layers (Reduced) | Number of params (Init) | Number of params (Reduced) | Time to process 10000 images (Init) |  Time to process 10000 images (Reduced) | Conversion Time (sec) | Maximum diff on final layer | Average difference on final layer |  
| --- | --- | --- |  --- |  --- |  --- |  --- |  --- |  --- |  --- |  --- |  
| MobileNet (1.0) | (224, 224, 3) | 102 | 75 | 4,253,864| 4,221,032| **32.38** | **22.13** | 12.45 | 2.80e-06 | 4.41e-09 |
| MobileNetV2 (1.4) | (224, 224, 3) | 152 | 100 | 6,156,712| 6,084,808| **52.53** | **37.71** | 87.00 | 3.99e-06 | 6.88e-09 |
| ResNet50 | (224, 224, 3) | 177 | 124 | 25,636,712 | 25,530,472 | **58.87** | **35.81** | 45.28 | 5.06e-07 | 1.24e-09 |
| Inception_v3 | (299, 299, 3) | 313 | 219 | 23,851,784 | 23,817,352 | **79.15** | **59.55** | 126.02 | 7.74e-07 | 1.26e-09 |
| Inception_Resnet_v2 | (299, 299, 3) | 782 | 578 | 55,873,736 | 55,813,192 | **131.16** | **102.38** | 766.14 | 8.04e-07 | 9.26e-10 |
| Xception | (299, 299, 3) | 134 | 94 | 22,910,480 | 22,828,688 | **115.56** | **76.17** | 28.15 | 3.65e-07 | 9.69e-10 |
| DenseNet121 | (224, 224, 3) | 428 | 369 | 8,062,504 | 8,040,040 | **68.25** | **57.57** | 392.24 | 4.61e-07 | 8.69e-09 |
| DenseNet169 | (224, 224, 3) | 596 | 513 | 14,307,880 | 14,276,200 | **80.56** | **68.74** | 772.54 | 2.14e-06 | 1.79e-09 |
| DenseNet201 | (224, 224, 3) | 708 | 609 | 20,242,984 | 20,205,160 | **98.99** | **87.04** | 1120.88 | 7.00e-07 | 1.27e-09 |
| NasNetMobile | (224, 224, 3) | 751 | 563 | 5,326,716 | 5,272,599 | **46.05** | **31.76** | 728.96 | 1.10e-06 | 1.60e-09 |
| NasNetLarge | (331, 331, 3) | 1021 | 761 | 88,949,818 | 88,658,596 | **445.58** | **328.16** | 1402.61 | 1.43e-07 | 5.88e-10 |
| [ZF_UNET_224](https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model) | (224, 224, 3) | 85 | 63 | 31,466,753 | 31,442,689 | **96.76** | **69.17** | 9.93 | 4.72e-05 | 7.54e-09 |
| [DeepLabV3+](https://github.com/bonlime/keras-deeplab-v3-plus) (mobile) | (512, 512, 3) | 162 | 108 | 2,146,645 | 2,097,013 | **583.63** | **432.71** | 48.00 | 4.72e-05 | 1.00e-05 |
| [DeepLabV3+](https://github.com/bonlime/keras-deeplab-v3-plus) (xception) | (512, 512, 3) | 409 | 263 | 41,258,213 | 40,954,013 | **1000.36** | **699.24** | 333.1 | 8.63e-05 | 5.22e-06 |
| [ResNet152](https://github.com/broadinstitute/keras-resnet) | (224, 224, 3) | 566 | 411 | 60,344,232 | 60,117,096 | **107.92** | **68.53** | 357.65 | 8.94e-07 | 1.27e-09 |

**Config**: Single NVIDIA GTX 1080 8 GB. Timing obtained on Tensorflow 1.4 (+ CUDA 8.0) backend

## Notes

* It feels like conversion works very slow for no reason, but it should be much faster since all 
manipulations with layers and weights are very fast. Probably I use some very slow Keras operations in process. 
Feel free to give advice on how to change code to make it faster.
* You can check that both models work the same with function: compare_two_models_results(model, model_reduced, 10000)
* Non-zero difference on final layer is accumulated because of large amount of floating point operations, which is not precise
* Some non-standard layer or parameters (which is not used in any keras.applications CNN) can produce wrong results. 
Most likely code will just fail in these conditions and you will see layer which cause it in python error message.
 
## Requirements

* Code was tested on Keras 2.1.6 (TensorFlow 1.4 backend) and on Keras 2.2.0 (TensorFlow 1.8.0 backend)

## Formulas

![Base formulas](https://raw.githubusercontent.com/ZFTurbo/Keras-inference-time-optimizer/master/img/conv_bn_fusion.png)

## Other implementations

[PyTorch BN Fusion](https://github.com/MIPT-Oulu/pytorch_bn_fusion) - with support for VGG, ResNet, SeNet.