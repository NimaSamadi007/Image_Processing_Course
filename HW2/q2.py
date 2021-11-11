import numpy as np
import cv2

def showImg(img, res_factor, title='input image'):
    res = (int(img.shape[0]*res_factor), int(img.shape[1]*res_factor))
    img_show = cv2.resize(img, res)
    cv2.imshow(title, img_show)
    return




img = cv2.imread('Greek-ship.jpg', cv2.IMREAD_COLOR)
patch = cv2.imread('patch.png', cv2.IMREAD_COLOR)

# SSD:
result = cv2.matchTemplate(img, patch, cv2.TM_CCORR_NORMED)

cv2.normalize(result, result, norm_type=cv2.NORM_INF)
print(np.amin(result), np.amax(result))
result = (result / np.amax(result)) * 255

# result = (result / np.amax(result)) * 255

# locations = np.nonzero(result >= 255) 

# locs = np.array(locations)
# print(locs.shape)


# for i in range(locs.shape[1]):
#     pt1 = (locs[1, i], locs[0, i])
#     pt2 = (locs[1, i]+patch.shape[1], locs[0, i]+patch.shape[0])
#     img = cv2.rectangle(img, pt1, pt2, 
#                         color=(0, 0, 255), thickness=5, lineType=8, shift=0)

# print(patch.shape[0], patch.shape[1])

# cv2.rectangle(img, (3237, 1338), (3237+patch.shape[1], 1338+patch.shape[0]), 
#                     color=(0, 0, 255), thickness=5, lineType=8, shift=0)


cv2.imwrite('l2.jpg', result.astype(np.uint8))

# showImg(img, 0.2, 'found')
showImg(result.astype(np.uint8), 0.2, 'result')
cv2.waitKey(0)
cv2.destroyAllWindows()