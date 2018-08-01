# SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation


### Prerequisites

* Keras 2.0
* opencv for python
* Tensorflow


### Preparing the data for training

You need to make two folders

*  Images Folder - For all the training images 
* Annotations Folder - For the corresponding ground truth segmentation images

The filenames of the annotation images should be same as the filenames of the RGB images.

The size of the annotation image for the corresponding RGB image should be same. 

For each pixel in the RGB image, the class label of that pixel in the annotation image would be the value of the blue pixel.

Example code to generate annotation images :

```python
import cv2
import numpy as np

ann_img = np.zeros((30,30,3)).astype('uint8')
ann_img[ 3 , 4 ] = 1 # this would set the label of pixel 3,4 as 1

cv2.imwrite( "ann_1.png" ,ann_img )
```

Only use bmp or png format for the annotation images.


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
 --val_images="data/dataset1/" \
 --val_annotations="data/dataset1/annotations_prepped_test/" \
 --n_classes=2
 --model_name="vgg_segnet" 
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
 --model_name="vgg_segnet" 
```

