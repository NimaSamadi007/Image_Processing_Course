import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils as utl

def calImageGradient(img):
    """
    Calculates image gradient, first converts image to grayscale
    and blures it to reduce noise
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #apply gaussian filter to reduce noise
    img_gray = cv2.GaussianBlur(img_gray, ksize=(3, 3), sigmaX=3, borderType=cv2.BORDER_CONSTANT)
    d_x = cv2.Sobel(src=img_gray, ddepth=-1, dx=1, dy=0, ksize=3, borderType=cv2.BORDER_CONSTANT)
    d_y = cv2.Sobel(src=img_gray, ddepth=-1, dx=0, dy=1, ksize=3, borderType=cv2.BORDER_CONSTANT)

    grad_mat = np.sqrt(d_x ** 2 + d_y ** 2)
    
    return grad_mat.astype(np.float64)

def makeListOfCenters(K):
    "Makes an empty dictionary which each key shows corresponding labels"
    centers_list = np.zeros((2, K), dtype=int)
    Sx = int(M / (K**(0.5)+1))
    Sy = int(N / (K**(0.5)+1))
    
    end_val = 0
    step = int(np.ceil(K**(0.5)))

    for i in range(step):
        for j in range(step):
            centers_list[0, end_val] = (i+1)*Sx
            centers_list[1, end_val] = (j+1)*Sy
            end_val += 1

    return centers_list

def drawPoints(img, centers):
    " Draw segments center point on image "
    img_copy = np.copy(img)
    for i in range(centers.shape[1]):
        cv2.circle(img_copy, (centers[1, i], centers[0, i]), 15, (0, 0, 255), -1)
    return img_copy

def purturbCenters(centers, img_grad):
    "Purturbs centers in a neighborhood of size (5, 5)"
    for i in range(centers_list.shape[1]):
        grad_sec = img_grad[centers_list[0, i]-2:centers_list[0, i]+3, 
                            centers_list[1, i]-2:centers_list[1, i]+3]    
        min_index = np.unravel_index(np.argmin(grad_sec), grad_sec.shape)
        #print(min_index)
        centers_list[0, i] += (min_index[0] - 2)
        centers_list[1, i] += (min_index[1] - 2)

def drawBoundries(img, segments, K):
    "Draws segments boundry in img"
    kernel = np.ones((3, 3), np.uint8)
    boundry_img = np.copy(img)    
    img_bin = np.zeros((M, N), np.uint8)
    for i in range(K):
        print("Drawing boundry for segment {}".format(i))
        img_bin[segments == i] = 255
        segment_boundry = cv2.morphologyEx(img_bin, cv2.MORPH_GRADIENT, kernel)
        boundry_img[segment_boundry == 255] = 0 
        img_bin[:, :] = 0    
    
    return boundry_img

#---------------------------------- MAIN ------------------------------#
img = cv2.imread('./slic.jpg', cv2.IMREAD_COLOR)
M, N, _ = img.shape

img_grad = calImageGradient(img)

# number of points
K = 1024
Sx = int(M / (K**(0.5)+1))
Sy = int(N / (K**(0.5)+1))
# step parameter - make grid of rectangle size
centers_list = makeListOfCenters(K)
# purturb center of segments in 5 * 5 square based on image gradients
purturbCenters(centers_list, img_grad)

img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab).astype(np.float64)

# relative importance factor
alpha = 5

# shows img segments in different color
img_segments = np.zeros((M, N), np.int32)
      
for i in range(M):
    print("At row {}".format(i))
    for j in range(N):        
        distance = np.abs(centers_list - np.array([i, j]).reshape(2, 1))
        indices = np.nonzero( (distance[0, :] <= Sx) & (distance[1, :] <= Sy) )
        if not len(indices[0]):
            center_index = np.argmin(distance[0, :]**2 + distance[1, :]**2)
            img[i, j, :] = img[centers_list[0, center_index], 
                               centers_list[1, center_index], :]
            #print(i, j)
            #print(indices)
        else:
            #print(indices)        
            x_centers = centers_list[0, indices[0]]
            y_centers = centers_list[1, indices[0]]    
            #print(x_centers)
            #print(y_centers)
            
            d_xy = (x_centers - i)**2 + (y_centers - j)**2
            #print(d_xy)
            d_lab = np.sum((img_lab[x_centers, y_centers, :] - img_lab[i, j, :])**2, axis=1)
            #print(d_lab)
            D = d_xy + alpha * d_lab
                
            corr_center_index = indices[0][np.argmin(D)]
            img_segments[i, j] = corr_center_index

fin_img = drawBoundries(img, img_segments, K)
cv2.imwrite('fin-img.jpg', fin_img)
    
    
    
    
    
    
    
    