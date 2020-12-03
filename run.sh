#!/bin/bash

# check for gpu with cuda installed
nvidia-smi

# git clone https://github.com/fizyr/keras-retinanet.git
# pip install keras==2.4
# pip install tensorflow==2.3.0
# cd keras-retinanet/
# pip install .
# python setup.py build_ext --inplace
# pip install gdown
# gdown --id 1mOK2qtcY9oNiAhWbOgvj4v8e9FuqcGIB 
# unzip imgs.zip

python libSetup.py

# training

# keras_retinanet/bin/train.py \
# --steps 500 \
# --epoch 10 \
# csv train_annotations.csv classes.csv

# create inference model
keras_retinanet/bin/convert_model.py snapshots/resnet50_csv_10.h5 snapshots/inference.h5


# compile model and run inference on an image
python libSetup_2.py