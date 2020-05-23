import numpy as np
from skimage.feature import local_binary_pattern


def getLBPimage(image):
    lbp = local_binary_pattern(image, 24, 8, 'uniform')
    hist = np.histogram(lbp, bins=np.arange(27))[0].astype('float')
    hist = hist / np.sum(hist)
    return hist


image = np.array([[0, 0, 1, 1],
                  [0, 0, 1, 1],
                  [0, 2, 2, 2],
                  [2, 2, 3, 3]], dtype=np.uint8)
getLBPimage(image)