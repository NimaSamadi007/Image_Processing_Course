import numpy as np
import cv2 as cv
import utils as utl
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

#------------------------------- FUNCTION ---------------------#
def makeMatrixD(m, n):
    # making D matrix:
    D = lil_matrix((m-2, m-2))
    I = lil_matrix((m-3, m-3))
    I.setdiag(-1)
    D.setdiag(4)
    D[:-1, 1:] += I
    D[1:, :-1] += I
    
    return D

def makeMatrixA(m, n):
    # first make matrix D
    D = makeMatrixD(m, n)
    # contruct A:
    A = lil_matrix(((m-2)*(n-2), (m-2)*(n-2)), dtype=float)
    I = lil_matrix((m-2, m-2))
    I.setdiag(1)
    
    # fill A with blocks of D on the diagonal
    for i in range(n-2):
        A[i*(m-2):(i+1)*(m-2), i*(m-2):(i+1)*(m-2)] = D
    
    # add I to A to fill off diagonals
    for i in range(n-3):
        A[i*(m-2):(i+1)*(m-2), (i+1)*(m-2):(i+2)*(m-2)] -= I
        A[(i+1)*(m-2):(i+2)*(m-2), i*(m-2):(i+1)*(m-2)] -= I

    return A
    
def solvePoissonPDE(laplacian_matrix, target_region, A):
    m, n = target_region.shape    
    target_region = target_region.astype(np.float32)
    
    # creating B coloumn
    # convert laplacian to 
    laplace_col = laplacian_matrix.T.reshape(-1, 1)
    B = lil_matrix(((m-2)*(n-2),1), dtype=float)
    B -= laplace_col
    for j in range(1, n-1):
        for i in range(1, m-1):
            # first or last row:
            if i == 1:
                # previous row
                val_row = target_region[i-1, j]
            elif i == m-2:
                # next row
                val_row = target_region[i+1, j]
            else:
                val_row = 0
            # first or last coloumn:
            if j == 1:
                # first coloumn
                val_col = target_region[i, j-1]
            elif j == n-2:
                val_col = target_region[i, j+1]
            else:
                val_col = 0            
            B[(i-1)+(j-1)*(m-2), 0] += (val_col + val_row)
    
    A = A.tocsr()
    u_col = spsolve(A, B)
    return u_col
    
#---------------------------- MAIN --------------------------------- #
source_img = cv.imread('./res05.jpg', cv.IMREAD_COLOR)
target_img = cv.imread('./res06.jpg', cv.IMREAD_COLOR)

mask = cv.imread('./mask.jpg', cv.IMREAD_GRAYSCALE)

# make binary mask
mask[mask > 0] = 1

Ms, Ns, _ = source_img.shape
Mt, Nt, _ = target_img.shape

# source (x, y) coordinates
x_s, y_s = 13, 73
# target (x, y) coordinates
x_t, y_t = 220, 916
# height and width of the region
height, width = 212, 204
M, N = height, width

# Possion blending for each channel
blended_target = np.zeros((M-2, N-2, 3), np.float32)
print("Constructing matrix A ...")
A = makeMatrixA(M, N)

print("Blending each channel ...")
for chn in range(3):
    source_region = source_img[x_s+1:x_s+height-1, y_s+1:y_s+width-1, chn]
    target_region = target_img[x_t:x_t+height, y_t:y_t+width, chn]
    
    # source image laplacian calculation:
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]], np.float32)
    source_laplacian = cv.filter2D(source_region.astype(np.float32), -1, laplacian_kernel, cv.BORDER_CONSTANT)
    
    target_region = target_region.astype(np.float32)
    print("Solving poisson equation for channel {}".format(chn+1))
    u_col = solvePoissonPDE(source_laplacian, target_region, A)
    utl.showRange(u_col, 'N')
    target_values = u_col.reshape(N-2, M-2).T
    blended_target[:, :, chn] = target_values


blended_target = utl.scaleIntensities(blended_target, 'C')
target_img_orig = np.copy(target_img)
target_img[x_t+1:x_t+height-1, y_t+1:y_t+width-1, :] = blended_target
target_img[mask == 0] = 0

erosion_kernel = np.ones((3, 3), dtype=np.uint8)
erosion_kernel = np.ones((3, 3), dtype=np.uint8)
eroded_img = cv.erode(mask, erosion_kernel, iterations=3)
mask = cv.dilate(eroded_img, erosion_kernel, iterations=3)
mask = np.stack([mask for _ in range(3)], axis=2)

final_img = target_img_orig * (1-mask) + target_img * mask

cv.imwrite('res07.jpg', final_img)
print("Done!")
