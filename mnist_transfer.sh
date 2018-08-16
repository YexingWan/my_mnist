#!/usr/bin/env bash
python3 ./convert_mnist.py -i mnist/train-images-idx3-ubyte -l mnist/train-labels-idx1-ubyte
mv image.npy train_image.npy
mv label.npy train_label.npy
python3 ./convert_mnist.py -i mnist/t10k-images-idx3-ubyte -l mnist/t10k-labels-idx1-ubyte
mv image.npy test_image.npy
mv label.npy test_label.npy




