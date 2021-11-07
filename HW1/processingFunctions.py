# show picture with specified resolution
import cv2
import numpy as np

def showImg(I, res, title='input image'):
    Ishow = cv2.resize(I, res)
    cv2.imshow(title, Ishow)
    return

# increse brightness by constant value
def increseBrightness(inTran, value):
    transform = np.zeros(256)
    for i in range(256):
        tmp = inTran[i] + value
        if tmp > 255:
            transform[i] = 255
        else:
            transform[i] = tmp
    
    return transform

# gamma function:
def gammaTransform(inTran, gamma):
    transform = np.zeros(256)
    for i in range(256):
        transform[i] = round(255 * ( ( inTran[i] / 255 ) ** gamma ))
    return transform

# points are in (r, s) format
def contrastStretching(inTran, point1, point2):
    transform = np.zeros(256)
    r1, s1 = point1
    r2, s2 = point2
    for i in range(256):
        if inTran[i] <= r1:
            transform[i] = round( inTran[i] * (s1 / r1) )
        elif inTran[i] > r1 and inTran[i] <= r2:
            transform[i] = round(s1 + ((s2 - s1) / (r2 - r1)) * (inTran[i] - r1))
        else:
            transform[i] = round(s2 + ((255 - s2) / (255 - r2)) * (inTran[i] - r2))
    
    return transform

# perform transformation using tran on the I image:
def calImg_RGB(I, tran):
    M, N, _ = I.shape
    for i in range(M):
        for j in range(N):
            for k in range(3):
               I[i][j][k] = tran[ I[i][j][k] ]
    I = I.astype('uint8')
    return I

def calImg_HSV(I, tran):
    M, N, _ = I.shape
    for i in range(M):
        for j in range(N):
            I[i][j][2] = tran[ I[i][j][2] ] # (H, S, V) : the last one is value
    I = I.astype('uint8')
    return I

def histogramCal_HSV(I):
    hist = np.zeros(256)
    M, N, _ = I.shape
    for i in range(M):
        for j in range(N):
            hist[ I[i][j][2] ] += 1   

    hist = np.true_divide(hist, M * N) # normalization
    return hist

def histogramEqulizer(hist):
    transform = np.zeros(256)
    for i in range(256):
        for j in range(i + 1):
            transform[i] += hist[j]
        transform[i] = round(transform[i] * 255)
    return transform
