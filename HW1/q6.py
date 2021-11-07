import numpy as np
import matplotlib.pyplot as plt
import cv2

# my processing function - make sure that this file is imported 
import processingFunctions as pf

def calHist_RGB(img):
    M, N, K = img.shape
    # print(M, N, K)
    hist = np.zeros((3, 256), dtype=np.int64)
    for i in range(M):
        for j in range(N):
            for k in range(K):
                hist[k, img[i, j, k]] += 1

    return (hist / (M * N))

def plotter(inputHist, specHist, changHist, title):
    fig, axes = plt.subplots(1, 3)
    axes[0].stem(np.arange(0, 256), inputHist)
    axes[0].title.set_text("Input " + title + " hist")

    axes[1].stem(np.arange(0, 256), specHist)
    axes[1].title.set_text("Specified " + title + " hist")
    
    axes[2].stem(np.arange(0, 256), changHist)
    axes[2].title.set_text("Changed " + title + " hist")

    fig.tight_layout()


inputImage = cv2.imread('Dark.jpg', cv2.IMREAD_UNCHANGED)
specificImage = cv2.imread('Pink.jpg', cv2.IMREAD_UNCHANGED)

inputHist = calHist_RGB(inputImage)
specificHist = calHist_RGB(specificImage)


histTrans = np.zeros((3, 256))

for j in range(3):
    equlized_input_hist = pf.histogramEqulizer(inputHist[j])
    equlized_specific_hist = pf.histogramEqulizer(specificHist[j])
    for i in range(256):
        ind = np.unravel_index(np.argmin(np.abs(equlized_specific_hist - equlized_input_hist[i])), equlized_input_hist.shape)
        histTrans[j, i] = ind[0]

M, N, K = inputImage.shape
changedImg = np.zeros((M, N, K))
for i in range(M):
    for j in range(N):
        for k in range(K):
            newVal = histTrans[k, inputImage[i, j, k]]
            changedImg[i, j, k] = newVal



inputImage = inputImage.astype(np.uint8)
changedImg = changedImg.astype(np.uint8)

changedHist = calHist_RGB(changedImg.astype(np.uint8))

"""

plotter(inputHist[0], specificHist[0], changedHist[0], 'blue')
plotter(inputHist[1], specificHist[1], changedHist[1], 'green')
plotter(inputHist[2], specificHist[2], changedHist[2], 'red')

"""

# pf.showImg(changedImg, (960, 960), 'changed Image')
# cv2.waitKey(0)
# cv2.destroyAllWindows()
plt.rcParams.update({'font.size': 14})

fig, axes = plt.subplots(3, figsize=(12, 12))
axes[0].stem(np.arange(0, 256), changedHist[0])
axes[0].title.set_text("Changed blue hist")
axes[0].set_xlabel("pixel intensities")
axes[0].grid(True)

axes[1].stem(np.arange(0, 256), changedHist[1])
axes[1].title.set_text("Changed green hist")
axes[1].set_xlabel("pixel intensities")
axes[1].grid(True)

axes[2].stem(np.arange(0, 256), changedHist[2])
axes[2].title.set_text("Changed red hist")
axes[2].set_xlabel("pixel intensities")
axes[2].grid(True)

fig.tight_layout()
# plt.show()
plt.savefig('res10.jpg', dpi=300, bbox_inches='tight')

cv2.imwrite('res11.jpg', changedImg)
