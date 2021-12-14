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

def isExist(arr, val, thr):
    "checkes if val exists in arr with a threshold"
    for i in range(len(arr)):
        if abs(val - arr[i]) <= thr:
            return i
    # not found
    return -1

def isParallelWithThr(distinct_angles, angle, angle_appears, angle_thr, app_thr):
    "cheks if given angle is distinct_angles and its repeatition is greater than thr"
    for i in range(len(distinct_angles)):
        if abs(angle - angles[i]) < angle_thr:
            if angle_appears[i] >= app_thr:
                return True
    return False
#/ ------------------------- MAIN ------------------------ /#

img = cv2.imread('./im02.jpg', cv2.IMREAD_COLOR)
img_edges = cv2.Canny(img, 100, 250)
#img_edges = cv2.Laplacian(img, ddepth=-1, ksize=1, borderType=cv2.BORDER_CONSTANT)
# threshold for converting edges to binary

thr = 10
img_edges[img_edges > thr] = 255
img_edges[img_edges <= thr] = 0
#utl.showImg(img_edges, 1)
M, N = img_edges.shape
angle_num = 400
len_num = 400
thr = 1e-2

#utl.showImg(img_edges, 0.8)

"""

print("Finding hough space representation ...")
voting_mat, len_angle_spc = houghTran(img_edges, len_num, angle_num, thr)
hough_space = utl.scaleIntensities(voting_mat)

np.save('voting.npy', voting_mat)
np.save('space.npy', len_angle_spc)
"""

voting_mat = np.load('voting.npy')
len_angle_spc = np.load('space.npy')
print(len(np.nonzero(voting_mat >= 0.3*np.amax(voting_mat))[0]))

max_indices = utl.findLocalMax(voting_mat, 0.4*np.amax(voting_mat), 0.1)
print(max_indices.shape)

angles = max_indices[1, :]
rhos = max_indices[0, :]
print(angles)

distinct_angles = []
angles_appear = []

distinct_angles.append(angles[0])
angles_appear.append(1)


for i in range(1, len(angles)):
    index = isExist(distinct_angles, angles[i], 4)
    if index == -1:
        distinct_angles.append(angles[i])
        angles_appear.append(1)
    else: 
        angles_appear[index] += 1
        
#print(distinct_angles)
#print(angles_appear)

#print("Finding local maximums and drawing lines ...")
angle_thr_ind = np.unravel_index(np.argsort(angles_appear, axis=None)[-2:], 
                                 len(angles_appear))[0][0]
angle_thr = angles_appear[angle_thr_ind]

rhos = []
thetas = []
# consider parallel lines and delete others
for i in range(max_indices.shape[1]):
    if isParallelWithThr(distinct_angles, max_indices[1, i], angles_appear, 4, angle_thr):
        rho = len_angle_spc[max_indices[0, i], max_indices[1, i], 0]
        theta = len_angle_spc[max_indices[0, i], max_indices[1, i], 1]
        rhos.append(rho)
        thetas.append(theta)

# try to remove lines that are not in chess area, using their color
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
final_rhos = []
final_thetas = []
for j in range(len(thetas)):
    tmp_img = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
    flag = False
    pt1, pt2 = convertToXY(rhos[j], thetas[j], N, M)
    cv2.line(tmp_img, (pt1[0], pt1[1]), (pt2[0], pt2[1]), 255, 1)
    
    line_pixels = np.nonzero(tmp_img == 255)
    x_pixels = line_pixels[0]
    y_pixels = line_pixels[1]
    
    # check neighoubers for line_len and neigh_num each side
    line_len = 20
    neigh_num = 20
    chess_square_thr = 150
    
    for i in range(len(x_pixels)-line_len):
        xi = x_pixels[i]
        yi = y_pixels[i]    
        
        left_equivalent_pixel = 0
        right_equivalent_pixel = 0
        for k in range(line_len):
            left_equivalent_pixel += np.sum(img_gray[x_pixels[i+k], (yi-neigh_num) % N:yi])
            right_equivalent_pixel += np.sum(img_gray[x_pixels[i+k], yi:(yi+neigh_num) % N])
    
        if abs(left_equivalent_pixel - right_equivalent_pixel) / (neigh_num * line_len) >= chess_square_thr:
            print("True, line passes chess area!")
            flag = True
            break
    if flag:
        final_rhos.append(rhos[j])
        final_thetas.append(thetas[j])

#print(final_rhos)
#print(final_thetas)
#utl.showImg(tmp_img, 0.5)

for i in range(len(final_thetas)):
    pt1, pt2 = convertToXY(final_rhos[i], final_thetas[i], N, M)
    cv2.line(img, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 0, 255), 2)


print("Done!")
cv2.imwrite('lines-found.jpg', img)
