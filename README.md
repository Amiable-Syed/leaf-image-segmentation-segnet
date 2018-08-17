# SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation


### Prerequisites

* Keras 2.0
* opencv for python
* Tensorflow


## Downloading the Pretrained VGG Weights

You need to download the pretrained VGG-16 weights trained on imagenet if you want to use VGG based models

```shell
mkdir data
cd data
wget "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5"
```



## Training the Model

To train the model run the following command:

```shell
python  train.py \
 --save_weights_path=weights/ex1 \
 --train_images="data/dataset1/images_prepped_train/" \
 --train_annotations="data/dataset1/annotations_prepped_train/" \
 --n_classes=2
```

Choose model_name from vgg_segnet  vgg_unet, vgg_unet2, fcn8, fcn32

## Getting the predictions

To get the predictions of a trained model

```shell
 python predict.py \
 --save_weights_path=weights/ex1 \
 --epoch_number=0 \
 --test_images="data/dataset1/images_prepped_test/" \
 --output_path="data/predictions/" \
 --n_classes=2
```

