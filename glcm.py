import numpy as np
from skimage import io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
eps = 2.2204e-16

def greycomatrix(image, distance, angles, levels):
         # khoi tao ma tran ket qua kich thuoc levels * levels
        result = np.zeros(shape=(levels, levels), dtype= np.uint32)
        print(result.shape)

        dk = (int)(round(distance * np.sin(angles)))
        dl = (int)(round(distance * np.cos(angles)))
        # duyet tung pixel de tim ma tran dong hien
        for k in range(image.shape[0]):
            for l in range (image.shape[1]):
                m = k + dk 
                n = l + dl
                if((m >= image.shape[0]) or (n >= image.shape[1])): 
                    continue
                result[image[k][l]][image[m][n]] += 1
        result = result.astype('float')
        return result / result.sum()

        
def glcm_feature(glcm_matrix, levels):
    glcm = glcm_matrix.astype('float')
    glcm = glcm / glcm.sum() # chuan hoa ma tran

    glcm_mean = np.mean(glcm)
    px = glcm.sum(axis=1)
    py = glcm.sum(axis=0)
    meanX = np.dot(px, np.arange(0, levels))
    meanY = np.dot(py, np.arange(0, levels))
    varX  = np.dot(px, np.square(np.arange(0, levels) - meanX))
    varY  = np.dot(px, np.square(np.arange(0, levels) - meanY))
    p_XplusY = np.zeros(2 * levels - 1)
    p_XminusY = np.zeros(levels)


    energy = np.sum(glcm ** 2)                               # F1 Energy
    entropy = np.sum(np.multiply(glcm, np.log2(glcm + eps))) # F2 entropy

    dissi = 0                           # F3 dissimilarity 
    contr = 0                           # F4 Contrast 
    homom = 0                           # F5 Inverse difference
    
    homop = 0                           # F7 homogenetity
    autocorrelation = 0                 # F8 auto correlation
    cluster_shade   = 0                 # F9 cluster shade
    cluster_prominence = 0              # F10 cluster cluster prominence
    maximum_probability = np.amax(glcm) # F11 maximum probability
    sosqu = 0                           # F12 Sum of squares
    indnm = 0                           # F21 inverse difference normalized
    inmnc = 0                           # F22 inverse difference moment

    for i in range( glcm.shape[0] ):
        for j in range(glcm.shape[1]):

            dissi = dissi + abs(i-j) * glcm[i][j]
            contr = contr + (abs(i -j) ** 2) * glcm[i][j]
            homop = homop + glcm[i][j] / (1 + (i - j) ** 2)
            homom = homom + glcm[i][j] / (1 + abs(i - j))
            sosqu = sosqu + glcm[i][j] * ((i - glcm_mean) ** 2)
            indnm = indnm + glcm[i][j] / (1 + abs(i - j) / (glcm.shape[0] * 1.0))
            inmnc = inmnc + glcm[i][j] / (1 + ((i - j)/(glcm.shape[0] * 1.0)) ** 2)
            autocorrelation += glcm[i][j] * i * j
            cluster_shade   += ((i + j - meanX - meanY ) ** 3) * glcm[i][j]
            cluster_prominence += (( i + j - meanX - meanY) ** 4) * glcm[i][j]

            p_XplusY[i + j] += glcm[i][j]
            p_XminusY[abs(i - j)] += glcm[i][j]    



    sum_average  = np.dot(p_XplusY, np.arange(2 * levels - 1))                                   # F13 sum average
    sum_variance = np.dot(p_XplusY, np.square(np.arange(2 * levels - 1) - sum_average))          # F14 sum variance
    sum_entropy  = - np.dot(p_XplusY, np.log2(p_XplusY + eps))                                   # F15 sum entropy

    difference_average  = np.dot(p_XminusY, np.arange(levels))                                   #  difference average
    difference_variance = np.dot(p_XminusY, np.square(np.arange(levels) - difference_average))   # F16 difference variance
    difference_entropy  = np.dot(p_XminusY, np.log2(p_XminusY + eps))                            # F17 difference entropy

    PXY  = np.dot(px.reshape(-1, 1), py.reshape(1, -1))
    HXY  = - np.sum(np.dot(glcm, np.log2(glcm + eps)))
    HXY1 = - np.sum(np.dot(glcm, np.log2(PXY + eps)))
    HXY2 = - np.sum(np.dot(PXY, np.log2(PXY + eps)))
    HX   = - np.sum(np.dot(px, np.log2(px + eps)))
    HY   = - np.sum(np.dot(py, np.log2(py + eps)))

    IMC1 = (HXY - HXY1) / (max(HX, HY))                 # F18 Information measures of correlation (1)
    IMC2 = (1 - np.exp(-2.0 * (HXY2 - HXY))) ** (1.0/2) # F19. Information measures of correlation (2)



    return np.array([energy, dissi, contr, entropy, homop, homom, sosqu, indnm, inmnc])



image = np.array([[0, 0, 1, 1],
                  [0, 0, 1, 1],
                  [0, 2, 2, 2],
                  [2, 2, 3, 3]], dtype=np.uint8)

res = greycomatrix(image, 2, np.pi/4, 4)

vector = glcm_feature(res, 4)
print(vector)