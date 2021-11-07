import numpy as np
import matplotlib.pyplot as plt
import cv2
# my processing function - make sure that this file is imported 
import processingFunctions as pf


def channelToImage(channels):
    _, M, N = channels.shape
    Blue = (channels[0]).reshape((M, N, 1))
    Green = (channels[1]).reshape((M, N, 1))
    Red = (channels[2]).reshape((M, N, 1))
    img = np.concatenate([Blue, Green, Red], axis=-1)
    return img

# extracts channel and normalizes the channels
def channelExtractor(I):
    M, _ = I.shape
    N = int(np.floor(M/3))
    Blue_channel = (I[0 : N] / (2 ** 16 - 1)).astype(np.float64)
    Green_channel = (I[N : 2*N] / (2 ** 16 - 1)).astype(np.float64)
    Red_channel = (I[2*N : 3*N] / (2 ** 16 - 1)).astype(np.float64)
    channels = np.stack([Blue_channel, Green_channel, Red_channel], axis=0)
    return channels

# calculate error:
def errorCal(chn1, chn2, errMod):
    M, N = chn1.shape
    Error = 0
    if errMod == 'ABS':
        return np.sum( np.abs(chn1 - chn2) )
    else:        
        return np.sqrt(np.sum( (chn1 - chn2) ** 2 ))

# shift the channel along with 'one' axis at a time
def channelShifter(chn, shift):
    chnShifted = np.copy(chn)

    M, N = chnShifted.shape
    if shift[0] >= 0:
        for i in range(M):
            chnShifted[i, int(shift[0]):] = chnShifted[i, 0:N-int(shift[0])]
    else:
        for i in range(M):
            chnShifted[i, 0:N+int(shift[0])] = chnShifted[i, -int(shift[0]):]

    if shift[1] >= 0:
        for j in range(N):
            chnShifted[int(shift[1]):, j] = chnShifted[0:M-int(shift[1]), j]
    else:
        for j in range(N):
            chnShifted[0:M+int(shift[1]), j] = chnShifted[-int(shift[1]):, j]
        
    return chnShifted

# resolution change
def changeRes(chn, coeff):
    M, N = chn.shape
    Mo, No = ( int(np.ceil(M / coeff)), int(np.ceil(N / coeff)) )
    resizedChn = np.zeros((Mo, No), dtype=chn.dtype)
    for i in range(0,M,coeff):
        resizedChn[int(i / coeff), : ] = chn[i, np.arange(0, N, coeff)]

    return resizedChn


def findProperShift(fixedChn, movingChn, resFactor, fixedInterval, errMod='RMS'):
    properShift = np.zeros(2, dtype=int)
    for frac in range(int(np.log2(resFactor)) + 1):
        properShift *= 2        

        currentFactor = int(resFactor / ( 2 ** frac) ) # changing resultion factor
        fixedResized = changeRes(fixedChn, currentFactor)
        movingResized = changeRes(movingChn, currentFactor)

        M, N = fixedResized.shape

        if frac == 0:
            errors = np.zeros( (4 * fixedInterval + 1 , 4 * fixedInterval + 1) ) # in last level enlarge interval by two        
            for i in range(4 * fixedInterval + 1):
                for j in range(4 * fixedInterval + 1):
                    xshift = int((i - 2 * fixedInterval + properShift[0]))
                    yshift = int((j - 2 * fixedInterval + properShift[1]))
                    C = channelShifter(movingResized, (xshift, yshift) )
                    errors[i, j] = errorCal(fixedResized[2*fixedInterval:M-2*fixedInterval, 2*fixedInterval:N-2*fixedInterval], 
                                                       C[2*fixedInterval:M-2*fixedInterval, 2*fixedInterval:N-2*fixedInterval], errMod)

            ind = np.array(np.unravel_index(np.argmin(errors), errors.shape), dtype=properShift.dtype)
            properShift += ind - np.array([2 * fixedInterval, 2 * fixedInterval], dtype=properShift.dtype) 

        else:
            errors = np.zeros( (2*fixedInterval+1 , 2*fixedInterval+1) )
            for i in range(2 * fixedInterval + 1):
                for j in range(2 * fixedInterval + 1):
                    xshift = int((i - fixedInterval + properShift[0]))
                    yshift = int((j - fixedInterval + properShift[1]))
                    C = channelShifter(movingResized, (xshift, yshift) )
                    errors[i, j] = errorCal(fixedResized[fixedInterval:M-fixedInterval, fixedInterval:N-fixedInterval], 
                                                       C[fixedInterval:M-fixedInterval, fixedInterval:N-fixedInterval], errMod)

            ind = np.array(np.unravel_index(np.argmin(errors), errors.shape), dtype=properShift.dtype)
            properShift += ind - np.array([fixedInterval, fixedInterval], dtype=properShift.dtype) 

        print("Current stage: {}".format(frac))
        print("Current shifts: ({}, {})".format(properShift[0], properShift[1]) )

    return properShift


def findClip(chn1_shift, ch2_shift):
    positive_clip = 0
    negative_clip = -1
    if chn1_shift * ch2_shift > 0:
        if chn1_shift > 0:
            positive_clip = np.amax([chn1_shift, ch2_shift])
        else:
            negative_clip = np.amin([chn1_shift, ch2_shift])
    else:
        if chn1_shift > 0:
            positive_clip = chn1_shift
            negative_clip = ch2_shift
        else:
            positive_clip = ch2_shift
            negative_clip = chn1_shift
    
    return [positive_clip, negative_clip]

def whiteColorClipper(cols, chn_col_numbers):
    white_thr = np.sum(cols[:, 0:5]) / (3 * 5) - 0.02 
    _, y = np.where(cols >= white_thr)
    left_margin = np.amax(y[np.where(y < chn_col_numbers//32)])
    right_margin = np.amin(y[np.where(y > chn_col_numbers//32)])
    return [left_margin, right_margin]

def calColorRow(chns):
    _, _, N = chns.shape
    color_rows = np.sum(chns, axis=2) / N
    return color_rows

def calColorCol(chns):
    _, M, _ = chns.shape
    color_cols = np.sum(chns, axis=1) / M
    return color_cols

def gaussianKernel(sigma, N):
    gauss = lambda sig, x: 1 / (np.sqrt(2 * np.pi) * sig) * np.exp(-x**2 / (2 * (sig ** 2)))
    sigmaList = np.ones(N*N) * sigma
    xList = np.arange(N) - ((N-1) / 2)
    gaussVal = np.array(list(map(gauss, sigmaList, xList))).reshape(N, 1)
    # for two dimmensional kernel:
    # kernel = gaussVal @ gaussVal.T
    return gaussVal / (np.sum(gaussVal))


# if col is inserted => negative: left, positive: right
# if row is inserter => negative: up, positive: down
def findFracIndex(chn_color, len_val, sigma, ksize):
    # cal derivative:
    chn_color_diff = np.abs(np.sum(np.diff(chn_color), axis=0) / 3)
    
    # filtering
    kernel = gaussianKernel(sigma, ksize)
    kernel = kernel.reshape(len(kernel))
    chn_color_diff = np.convolve(chn_color_diff, kernel, 'same')

    # normalizing
    chn_color_diff /= np.amax(chn_color_diff)

    # used for report:
    # plt.plot(np.arange(len(chn_color_diff)), chn_color_diff)
    # plt.xlabel('Pixels')
    # plt.ylabel('value')
    # plt.title('Smoothed derivative (abs)')
    # plt.grid(True)
    # plt.show()

    thr = 10 * np.sum(chn_color_diff) / len_val 

    indices = np.array(np.where(chn_color_diff >= thr))
    indices = indices.reshape(indices.shape[1])
    negative_halves = indices[indices <= len_val // 2]
    positive_halves = indices[indices > len_val // 2]

    if len(negative_halves) == 0:
        negative_half = 0
    else:
        negative_half = negative_halves[-1]
    
    if len(positive_halves) == 0:
        positive_half = 0
    else:
        positive_half = positive_halves[0]
    
    return [negative_half, positive_half]    


# change the picture name to use other pictures
print("Enter the img's name (without .tif, e.g. Amir, Train, Mosque etc)")
picture_name = input()
I = cv2.imread(picture_name + '.tif', cv2.IMREAD_UNCHANGED)
if I is None:
    raise(AttributeError("Img's name is wrong. Please make sure to enter ONLY the picture name (without .tif). Then check that if the img and this script are togheter."))
else:
    channels = channelExtractor(I)
    print("Img loaded successfully!")
    print("Starting procedures ...")
    print("--------------------------------------------------")
# channel extraction:

################################################################################
# proper shift finder:

# green channel is fixed:
resFactor = 16 # pyramid depth
fixedInterval = 4 # search interval
print("Finding red channel proper shift:")
redShift = findProperShift(channels[1], channels[2], resFactor, fixedInterval, 'ABS')
print("Finding blue channel shifts:")
blueShift = findProperShift(channels[1], channels[0], resFactor, fixedInterval, 'ABS')

# shift channels with respect to blueShift and redShift
print("--------------------------------------------------")
print("Final red channel's shift:")
print(redShift)
print("Final blue channel's shift:")
print(blueShift)
channels[0] = channelShifter(channels[0], blueShift)
channels[2] = channelShifter(channels[2], redShift)

################################################################################
# clip extra color caused by shifting the image:

[left_clip, right_clip] = findClip(redShift[1], blueShift[1])
[top_clip, bottom_clip] = findClip(redShift[0], blueShift[0])
channels = channels[:, left_clip:right_clip, top_clip:bottom_clip]

###############################################################################
# white color clipping:
color_cols = calColorCol(channels)
[left_margin, right_margin] = whiteColorClipper(color_cols, channels.shape[1])
channels = channels[:, :, left_margin:right_margin]

###############################################################################
# dark color clipping:
color_cols = calColorCol(channels)
color_rows = calColorRow(channels)

_, M, N = channels.shape

# find proper fraction of picture to be clipped
sigma = 2
[left_frac, right_frac] = findFracIndex(color_cols, N, sigma, 6*sigma+1)
[top_frac, bottom_frac] = findFracIndex(color_rows, M, sigma, 6*sigma+1)

channels = channels[:, top_frac:bottom_frac, left_frac:right_frac]

channels *= (2 ** 8 - 1)
img = channelToImage(channels.astype(np.uint8))

cv2.imwrite('res03-{}.jpg'.format(picture_name), img)

pf.showImg(img, (960, 960), 'Color Image')
cv2.waitKey(0)
cv2.destroyAllWindows()


