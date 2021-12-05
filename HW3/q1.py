import numpy as np
import cv2
import utils as utl

#/ ------------------------- FUNCTIONS -------------------- /#
def calAngleLenMat(M, N):
    "calculates angle-length matrix of M * M shape"
    
    angle_step = 2 * np.pi / (N-1)
    angle_row = np.arange(-np.pi, np.pi+angle_step, angle_step)
    angle_mat = utl.repeatRow(angle_row, M)

    len_step = (2 * np.sqrt(2) / (M-1))
    len_col = np.arange(-np.sqrt(2), np.sqrt(2)+len_step, len_step)
    len_mat = utl.repeatCol(len_col, N)

    return np.stack([len_mat, angle_mat], axis=2)

def updateVotingMat(voting_mat, len_angle_space, x, y):
    "update voting matrix for each (x, y) pair and for all (rho, theta) pairs"
    condition_matrix = np.abs((x * np.cos(len_angle_space[:, :, 1]) + y * np.sin(len_angle_space[:, :, 1]) - len_angle_space[:, :, 0]))
    voting_mat[condition_matrix < 1e-6] += 1

def houghTran(img_edges, step_shape):
    "Findes Hough transform of a specified image and returns voting matrix and len-angle space representation"

    M, N = img_edges.shape
    voting_mat = np.zeros((step_shape[0], step_shape[1]), dtype=np.int64)
    len_angle_space = calAngleLenMat(step_shape[0], step_shape[1])

    # check every edge candidate for a line
    for i in range(M):
        # print("In stage x={}".format(i))
        for j in range(N):
            if img_edges[i, j] == 255:
                x = (i - M//2) / (M//2)
                y = (j - N//2) / (N//2)
                updateVotingMat(voting_mat, len_angle_space, x, y)

    return voting_mat, len_angle_space

def convertToXY(rho, theta, M, N):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    if np.abs(sin_theta) < 1e-9:
        sin_theta = 1e-9
    if np.abs(cos_theta) < 1e-9:
        cos_theta = 1e-9
    
    pt1 = np.array([0, rho / sin_theta])
    pt2 = np.array([rho/cos_theta, 0])
    
    pt1 *= np.array([M//2, N//2])
    pt1 += np.array([M//2, N//2])
    pt2 *= np.array([M//2, N//2])
    pt2 += np.array([M//2, N//2])
    
    if pt1[1] > N:
        pt1[1] = N
    if pt2[0] < M:
        pt2[0] = N

    return (pt1.astype(int), pt2.astype(int))
#/ ------------------------- MAIN ------------------------ /#

img1 = cv2.imread('./im01.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('./im02.jpg', cv2.IMREAD_COLOR)

img1_edges = cv2.Canny(img1, 100, 250)
img2_edges = cv2.Canny(img2, 100, 250)

# threshold for converting edges to binary
thr = 10
img1_edges[img1_edges > thr] = 255
img1_edges[img1_edges <= thr] = 0
img2_edges[img2_edges > thr] = 255
img2_edges[img2_edges <= thr] = 0

M1, N1 = img1_edges.shape

#cv2.imwrite('res01.jpg', img1_edges)
#cv2.imwrite('res02.jpg', img2_edges)

voting_mat, angle_line_spc = houghTran(img1_edges, (5, 5))
print(voting_mat)

for i in range(5):
    for j in range(5):
        if voting_mat[i, j] >= 40:
            pt1, pt2 = convertToXY(angle_line_spc[i, j, 0], angle_line_spc[i, j, 1], M1, N1)
            print(pt1[0], pt1[1])
            print(pt2[0], pt2[1])
            cv2.line(img1, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 0, 255), 5)

# cv2.line(img1, (0, 750), (500, 0), (0, 0, 255), 10)
utl.showImg(img1, 0.5, 'found lines')