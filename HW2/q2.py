import numpy as np
import cv2
from numpy.core.fromnumeric import nonzero

def showImg(img, res_factor, title='input image'):
    res = (int(img.shape[0]*res_factor), int(img.shape[1]*res_factor))
    img_show = cv2.resize(img, res)
    cv2.imshow(title, img_show)
    return

def calProperIndex(x, y, x_thr, y_thr, M, N):
    xi = x
    yi = y

    if x <= M - x_thr:
        xf = x + x_thr + 1
    else:
        xf = M

    if y <= N - y_thr:
        yf = y + y_thr + 1
    else:
        yf = N
    return (xi, xf), (yi, yf)




img = cv2.imread('Greek-ship.jpg', cv2.IMREAD_COLOR)
patch = cv2.imread('patch.png', cv2.IMREAD_COLOR)

# SSD:
result = cv2.matchTemplate(img, patch, cv2.TM_CCORR_NORMED)

cv2.normalize(result, result, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
# result = (result / np.amax(result)) * 255

locations = np.array(np.nonzero(result >= (1 - 7 * 10 ** (-3) )))

local_max = []

M, N = result.shape

for i in range(locations.shape[1]):
    x = locations[0, i]
    y = locations[1, i]
    (xi, xf), (yi, yf) = calProperIndex(x, y, patch.shape[0], patch.shape[1], M, N)
    cond = np.array(np.nonzero(result[x, y] > result[xi:xf, yi:yf]))
    if cond.shape[1] >= (patch.shape[0] * patch.shape[1] - 1):
        local_max.append([x, y])

local_max = np.array(local_max)


local_max = local_max.T
# xs = local_max[0]
# ys = local_max[1]
# print(result[xs, ys])


for i in range(local_max.shape[1]):
    pt1 = (local_max[1, i], local_max[0, i])
    pt2 = (local_max[1, i]+patch.shape[1], local_max[0, i]+patch.shape[0])
    img = cv2.rectangle(img, pt1, pt2, 
                        color=(0, 0, 255), thickness=5, lineType=8, shift=0)

# print(patch.shape[0], patch.shape[1])

# cv2.rectangle(img, (3237, 1338), (3237+patch.shape[1], 1338+patch.shape[0]), 
#                     color=(0, 0, 255), thickness=5, lineType=8, shift=0)


showImg(img, 0.2, 'found')
# showImg(result.astype(np.uint8), 0.2, 'result')
cv2.waitKey(0)
cv2.destroyAllWindows()