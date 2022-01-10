import numpy as np
import cv2
import utils as utl

def makeListOfCenters(stepx, stepy, Sx, Sy, K):
    "Makes an empty dictionary which each key shows corresponding labels"
    centers_list = np.zeros((2, K), dtype=int)
    end_val = 0
    for i in range(stepx):
        for j in range(stepy):
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
    for i in range(centers.shape[1]):
        grad_sec = img_grad[centers[0, i]-2:centers[0, i]+3, 
                            centers[1, i]-2:centers[1, i]+3]    
        min_index = np.unravel_index(np.argmin(grad_sec), grad_sec.shape)
        centers[0, i] += (min_index[0] - 2)
        centers[1, i] += (min_index[1] - 2)

def drawBoundries(img, segments, K):
    "Draws segments boundry in img"
    M, N, _ = img.shape
    kernel = np.ones((3, 3), np.uint8)
    boundry_img = np.copy(img)    
    img_bin = np.zeros((M, N), np.uint8)
    for i in range(K):
        print("Drawing boundry for segment {}".format(i))
        img_bin[segments == i] = 255
        # find boundries using morphology
        segment_boundry = cv2.morphologyEx(img_bin, cv2.MORPH_GRADIENT, kernel)
        
        boundry_img[segment_boundry == 255] = 0 
        img_bin[:, :] = 0    
    
    return boundry_img


def runSLIC(img, img_lab, img_grad, alpha, K, img_name):
    M, N, _ = img.shape
    # step parameter - make grid of rectangle size
    if K == 2048:
        stepx = int(np.ceil((K/2)**(0.5)))
        stepy = 2*int(np.ceil((K/2)**(0.5)))
    else:
        stepx = int(np.ceil(K**(0.5)))
        stepy = int(np.ceil(K**(0.5)))

    Sx = int(M / (stepx+1))
    Sy = int(N / (stepy+1))

    # make initial centers
    centers_list = makeListOfCenters(stepx, stepy, Sx, Sy, K)
    # purturb center of segments in 5 * 5 square based on image gradients
    purturbCenters(centers_list, img_grad)

    # shows img segments in different color
    img_segments = np.zeros((M, N), np.int32)

    for i in range(M):
        print("At row {}".format(i))
        for j in range(N):
            distance = np.abs(centers_list - np.array([i, j]).reshape(2, 1))
            # find the nearest 4 centers
            indices = np.nonzero( (distance[0, :] <= Sx) & (distance[1, :] <= Sy) )
            if not len(indices[0]):
                # no nearest center found
                center_index = np.argmin(distance[0, :]**2 + distance[1, :]**2)
                img_segments[i, j] = center_index
            else:
                # find best out of 4 neighbors
                x_centers = centers_list[0, indices[0]]
                y_centers = centers_list[1, indices[0]]    
                
                d_xy = (x_centers - i)**2 + (y_centers - j)**2
                d_lab = np.sum((img_lab[x_centers, y_centers, :] - img_lab[i, j, :])**2, axis=1)
                D = alpha * d_xy + d_lab
                    
                corr_center_index = indices[0][np.argmin(D)]
                img_segments[i, j] = corr_center_index
    
    fin_img = drawBoundries(img, img_segments, K)
    cv2.imwrite('{}.jpg'.format(img_name), fin_img)
    print('{} has been written!'.format(img_name))
    
#--------------------------------- MAIN ------------------------------#
img = cv2.imread('./slic.jpg', cv2.IMREAD_COLOR)
# computing image gradient, used in purturbing initial center points:
img_grad = utl.calImageGradient(img, 1, 'Scharr')
# blur image to reduce noises
img_blured = cv2.GaussianBlur(img, ksize=(7, 7),
                              sigmaX=2, borderType=cv2.BORDER_CONSTANT)
img_lab = cv2.cvtColor(img_blured, cv2.COLOR_RGB2Lab).astype(np.float64)

# relative importance factor
# alpha = 0.02
# K = 64
# alpha = 0.05
# K = 256
# alpha = 0.1
# K = 1024
alpha = 0.5 
K = 2048
img_cop = np.copy(img)
runSLIC(img_cop, img_lab, img_grad, alpha, K, 'res09')
print("Done!")