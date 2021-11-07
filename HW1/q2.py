import numpy as np
import cv2
# my processing function - make sure that this file is imported 
import processingFunctions as pf


H = np.arange(256)
H = pf.gammaTransform(H, 0.35)
Z = pf.contrastStretching(H, (60, 10), (170, 240))


I = cv2.imread('Enhance2.jpg')

I_hsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)

I_hsv = pf.calImg_HSV(I_hsv, Z)



I_transformed = cv2.cvtColor(I_hsv, cv2.COLOR_HSV2BGR)

pf.showImg(I, (960, 540), 'original image')
pf.showImg(I_transformed, (960, 540), 'trnsformed image')
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('res02.jpg', I_transformed)