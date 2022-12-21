# CapsNet

This is a repository containing the work done in accordance with the thesis:

*Deep fMRI Encoding Models of Human Vision* by Shawn Carere

submitted in conformity with the requirements for the degree of Master of Science,
Department of Medical Biophysics, University of Toronto under the supervision of
Dr. Kamil Uludag.

## Overview

For an overview of how all results and figures (including supplementary figures) in the entire thesis were obtained, please refer to the [final_results tutoiral](analysis/final_results.ipynb). For anyone interested in this work, we strongly recommend that you start with this notebook tutorial. It contains everything except model inference and training. Please note that due to a restructuring, some of the current import and file paths in the other scripts may be incorrect.

Some of the data preprocessing code was borrowed from Beliy et al. [1] in order to be consistent with their approach.

> [1] R. Beliy, G. Gaziv, A. Hoogi, F. Strappini, T. Golan, and M. Irani, “From voxels to pixels and back: Self-supervision in natural-image reconstruction from fMRI,” in Advances in Neural Information Processing Systems, 2019, vol. 32. Accessed: May 06, 2022. [Online]. Available: https://proceedings.neurips.cc/paper/2019/hash/7d2be41b1bde6ff8fe45150c37488ebb-Abstract.html


### PyCaps

Contains a tensforflow implementation of Capsule networks proposed in the papers:

> [2] S. Sabour, N. Frosst, and G. E. Hinton, “Dynamic Routing Between Capsules,” in Advances in Neural Information Processing Systems 30, Long Beach, California, 2017, pp. 3856–3866. Accessed: Oct. 15, 2020. [Online]. Available: http://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf

> [3] G. E. Hinton, S. Sabour, and N. Frosst, “Matrix capsules with EM routing,” presented at the International Conference on Learning Representations, Vancouver, Canada, Feb. 2018. Accessed: Oct. 15, 2020. [Online]. Available: https://openreview.net/forum?id=HJWLfGWRb

In the future, this could be abstracted from this research project to provide an
API extension to tensorflow for training and experimenting with capsule networks.
Currently includes:

  - Layer subclasses for primary, convolution and class/dense capsule layers
  - Model subclasses for the models from the original capsule network papers
  - Loss functions used in the original papers
  - Dynamic and EM routing algorithms

### Training

Contains training files and scripts used to train capsule models. Includes

  - Training for capsule based image-fMRI encoding models
  - Training for capsule based MNIST classifiers
  - Generators used to augment data during training

Two different image-fMRI datasets from the following works were used to train the encoders

> [4] T. Horikawa and Y. Kamitani, “Generic decoding of seen and imagined objects using hierarchical visual features,” Nature Communications, vol. 8, no. 1, Art. no. 1, May 2017, doi: 10.1038/ncomms15037.

> [5] M. A. J. van Gerven, F. P. de Lange, and T. Heskes, “Neural Decoding with Hierarchical Generative Models,” Neural Computation, vol. 22, no. 12, pp. 3127–3142, Dec. 2010, doi: 10.1162/NECO_a_00047.

### Analysis

Contains various scripts and notebooks that were used to analyze the data and model
performance. Currently, there are plans to include a final analysis script showing how all the figures and metrics
in the thesis were calculated/obtained

### Misc

Other miscellaneous files. Includes

  - Data handler object for preprocessing the image-fMRI data from the Generic 
  Object Decoding dataset [4].
  - Functions used to load data, further preprocess data or analyze results