from tensorflow.keras.utils import Sequence
import numpy as np
from scipy.ndimage import shift

class ShiftGen(Sequence):
    def __init__(self, X, Y, batch_size=32, max_shift=5):
        self.indexes = {}
        self.batch_size = batch_size
        self.Y = np.array(Y)
        self.X = np.array(X)
        self.indexes  = np.random.permutation(self.Y.shape[0])
        self.max_shift=max_shift

    def _rand_shift(self, img, max_shift = 0 ):
        """
        randomly shifted image

        :param img: image
        :param max_shift: image output size

        :return randomly shifted image
        """
        x_shift, y_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
        img_shifted = shift(img, [x_shift, y_shift, 0], prefilter=False, order=0, mode='nearest')
        return img_shifted
    def __len__(self):
        'Denotes the number of batches per epoch'
        return max(int(self.Y.shape[0] // self.batch_size), 1) #

    def __getitem__(self, batch_num):
        indexes = (self.indexes)[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]
        y = self.Y[indexes]
        x = self.X[indexes]
        x_shifted = np.zeros(x.shape)
        for i in range(x.shape[0]):
            x_shifted[i] = self._rand_shift(x[i], self.max_shift)
        return x_shifted,y

    def on_epoch_end(self):
        self.indexes = np.random.permutation(self.indexes)
