import numpy as np
import cv2
import utils as utl

#/ ------------------------- FUNCTIONS -------------------- /#
def calLenAngleMat(len_level, angle_level):
    "calculates angle-length matrix of given shape (len_level * angle_level)"
    "First parameter is len and second parameter is angle"

    angle_row = np.linspace(start=-np.pi, stop=np.pi, num=angle_level, endpoint=True, dtype=np.float64)
    angle_mat = utl.repeatRow(angle_row, len_level)
    
    len_col = np.linspace(start=-2**(0.5), stop=2**(0.5), num=len_level, endpoint=True, dtype=np.float64)
    len_mat = utl.repeatCol(len_col, angle_level)

    return np.stack([len_mat, angle_mat], axis=2)

def updateVotingMat(voting_mat, len_angle_space, x, y, thr):
    "update voting matrix for each (x, y) pair and for all (rho, theta) pairs"
    condition_matrix = np.abs((x * np.cos(len_angle_space[:, :, 1]) + 
                               y * np.sin(len_angle_space[:, :, 1]) - 
                               len_angle_space[:, :, 0]))
    voting_mat[condition_matrix < thr] += 1

def houghTran(img_edges, len_level, angle_level, thr):
    "Findes Hough transform of a specified image and returns voting matrix and len-angle space representation"
    M, N = img_edges.shape
    voting_mat = np.zeros((len_level, angle_level), dtype=np.int64)
    len_angle_space = calLenAngleMat(len_level, angle_level)

    # check every edge candidate for a line
    for i in range(M):
        print("In stage x={}".format(i))
        for j in range(N):
            if img_edges[i, j] == 255:
                x = (j - N//2) / (N//2)
                y = -(i - M//2) / (M//2)
                # swap y and x since matrix and cartesian indices are different
                updateVotingMat(voting_mat, len_angle_space, x, y, thr)

    return voting_mat, len_angle_space

def convertToXY(rho, theta, M, N):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # determine quarter
    pt1 = np.zeros(2, dtype=np.float64)
    pt2 = np.zeros(2, dtype=np.float64)
    
    # extreme cases:    
    if np.abs(theta) <= 10 ** (-9) or np.abs(theta - np.pi) <= 10 ** (-9) or np.abs(theta + np.pi) <= 10 ** (-9):
        # vertical lines
        pt1 = np.array([rho, 1])
        pt2 = np.array([rho, -1])
        
    elif np.abs(theta - np.pi/2) <= 10 ** (-9) or np.abs(theta + np.pi/2) <= 10 ** (-9):
        # horizontal lines
        pt1 = np.array([1, rho])
        pt2 = np.array([-1, rho])

    elif theta > (-np.pi) and theta < (-np.pi / 2):
        # third quarter
        pt1 = np.array([(rho+sin_theta) / cos_theta, -1])
        pt2 = np.array([-1, (rho+cos_theta) / sin_theta])

    elif theta > (-np.pi/2) and theta < 0:
        # fourth quarter
        pt1 = np.array([(rho+sin_theta) / cos_theta, -1])
        pt2 = np.array([1, (rho-cos_theta) / sin_theta])

    elif theta > 0 and theta < (np.pi / 2):
        # first quarter
        pt1 = np.array([(rho-sin_theta) / cos_theta, 1])
        pt2 = np.array([1, (rho-cos_theta) / sin_theta])
        
    elif theta > (np.pi / 2) and theta < np.pi:
        # second quarter
        pt1 = np.array([(rho-sin_theta) / cos_theta, 1])
        pt2 = np.array([-1, (rho+cos_theta) / sin_theta])
        
    else:
        print("Not found!")        

    pt1 *= np.array([M//2, -N//2])
    pt1 += np.array([M//2, N//2])
    pt2 *= np.array([M//2, -N//2])
    pt2 += np.array([M//2, N//2])
    
    return (pt1.astype(int), pt2.astype(int))
#/ ------------------------- MAIN ------------------------ /#

img = cv2.imread('./im02.jpg', cv2.IMREAD_COLOR)
"""
# utl.showImg(img, 1)
img_edges = cv2.Canny(img, 100, 250)

# threshold for converting edges to binary
thr = 10
img_edges[img_edges > thr] = 255
img_edges[img_edges <= thr] = 0
# utl.showImg(img_edges, 1)

M, N = img_edges.shape
angle_num = 400
len_num = 400
thr = 1e-2

print("Finding hough space representation ...")
voting_mat, len_angle_spc = houghTran(img_edges, len_num, angle_num, thr)
hough_space = utl.scaleIntensities(voting_mat)
"""
print(len(np.nonzero(voting_mat >= 0.4*np.amax(voting_mat))[0]))

max_indices = utl.findLocalMax(voting_mat, 0.4*np.amax(voting_mat), 0.1)
print(max_indices.shape)
#utl.show()
cv2.imwrite("hough_space.jpg", hough_space)

# utl.showRange(voting_mat, 'R')
print("Finding local maximums and drawing lines ...")
for i in range(max_indices.shape[1]):
    pt1, pt2 = convertToXY(len_angle_spc[max_indices[0, i], max_indices[1, i], 0], 
                           len_angle_spc[max_indices[0, i], max_indices[1, i], 1], N, M)
    cv2.line(img, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 0, 255), 1)

#utl.showImg(img, 1, 'found')
print("Done!")
cv2.imwrite('lines-found.jpg', img)
