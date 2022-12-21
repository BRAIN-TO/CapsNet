# MISC

**helper_functions.py** contains various functions used to load and process data
and results

**kamitani_data_handler.py** is a script from [Beliy et al.](https://github.com/WeizmannVision/ssfmri2im)
used to load the Generic Object Decoding dataset. We made some slight modifications
to this class whilst experimenting with the dataset

**kamitani_image_prepare.py** is another script from [Beliy et al.](https://github.com/WeizmannVision/ssfmri2im).
It processes the raw ImageNet images and prepares them to be used in training.
The processed train and test images are placed in a dictionary and saved in pickle
format