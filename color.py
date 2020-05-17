import numpy as np
from skimage import io
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt
from scipy.stats import moment

def show_image(image):
    plt.imshow(image)
    plt.show()

def get_color_feature(image):
    # convert tu rgb sang hsv
    hsv_image = np.array(rgb2hsv(image))

    # lay tung kenh mau
    H = hsv_image[:, :, 0]
    print(H.shape)
    S = hsv_image[:, :, 1]
    print(S.shape)
    V = hsv_image[:, :, 2]
    print(V.shape)
    
    # reshape ve 1 vector
    H = H.reshape(1, -1)[0]
    S = S.reshape(1, -1)[0]
    V = V.reshape(1, -1)[0]

    # tinh mean cua vector
    meanH = np.mean(H)
    meanS = np.mean(S)
    meanV = np.mean(V)

    # tinh standard deviation
    stdH = moment(H, moment = 2) ** (1./2)
    stdS = moment(S, moment = 2) ** (1./2)
    stdV = moment(V, moment = 2) ** (1./2)

    # tinh skewness
    sknH = np.cbrt(moment(H, moment=3))
    sknS = np.cbrt(moment(S, moment=3)) 
    sknV = np.cbrt(moment(V, moment=3))

    colour_vector = np.array([meanH, meanS, meanV, stdH, stdS, stdV, sknH, sknS, sknV])
    return colour_vector





data = io.imread('../flower.jpg')
#show_image(data)
print(data.shape)
hsv = rgb2hsv(data)
colour_vector = get_color_feature(data)
print(colour_vector)