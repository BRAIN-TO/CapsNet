# Training

This directory contains the training scripts used to train various caspule networks

**generators.py** contains some generator objects that apply online pixel shifting
to the images during training as well as shuffles the samples between epochs

**train_capsnet.py** trains a capsule network based MNIST classifier

**train_encoder.py** trains a capsule network based image-fMRI encoder on the
Generic Object Decoding dataset which uses ImageNet images as stimuli

**train_simple_encoder** trains a capsule network based image-fMRI encoder on the
Handwritten Digits in fMRI dataset which uses MNIST images as stimuli