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

    pt1 *= np.array([N//2, -M//2])
    pt1 += np.array([N//2, M//2])
    pt2 *= np.array([N//2, -M//2])
    pt2 += np.array([N//2, M//2])
    
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
        if abs(angle - distinct_angles[i]) < angle_thr:
            if angle_appears[i] >= app_thr:
                return True
    return False

def lineDrawer(input_img, rhos, thetas, thick):
    "Drawes different lines according to respective angles and rhos"
    img_cop = np.copy(input_img)
    M = input_img.shape[0]
    N = input_img.shape[1]
    for i in range(len(thetas)):
        pt1, pt2 = convertToXY(rhos[i], thetas[i], M, N)
        cv2.line(img_cop, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 0, 255), thick)
    return img_cop

def img2bin(img, bin_thr):
    "Makes an img binary within a threshold"
    img[img > bin_thr] = 255
    img[img <= bin_thr] = 0

def findDistinctAngles(angles, common_lines):
    "Findes distinct angles which are unique with a threshold"
    distinct_angles = []
    angles_appear = []

    distinct_angles.append(angles1[0])
    angles_appear.append(1)

    for i in range(1, len(angles)):
        index = isExist(distinct_angles, angles[i], common_lines)
        if index == -1:
            distinct_angles.append(angles[i])
            angles_appear.append(1)
        else: 
            angles_appear[index] += 1
    
    return distinct_angles, angles_appear

def findParallelLines(max_indices, distinct_angles, angles_appear, common_lines, app_thr, len_angle_spc):
   " findes parallel lines and delete others "
   rhos = []
   thetas = []

   for i in range(max_indices.shape[1]):
       if isParallelWithThr(distinct_angles, max_indices[1, i], angles_appear, common_lines, app_thr):
           rho = len_angle_spc[max_indices[0, i], max_indices[1, i], 0]
           theta = len_angle_spc[max_indices[0, i], max_indices[1, i], 1]
           rhos.append(rho)
           thetas.append(theta)

   rhos = np.array(rhos)
   thetas = np.array(thetas)
    
   return rhos, thetas

def passesChessArea(img, rhos, thetas, line_len, neigh_num, chess_square_thr, voting_thr):
    "Finds lines that pass through the chess area"
    final_rhos = []
    final_thetas = []
    thetas_degree = (thetas / (np.pi) * 180)

    # the value of image is passed
    M, N = img.shape
    for i in range(len(thetas)):
        tmp_img = np.zeros((M, N), dtype=img.dtype)
        
        pt1, pt2 = convertToXY(rhos[i], thetas[i], M, N)
        cv2.line(tmp_img, (pt1[0], pt1[1]), (pt2[0], pt2[1]), 255, 1)

        shifting_angle = 0
        if thetas_degree[i] >= 0 and thetas_degree[i] <= 90:
            # first quarter:
            shifting_angle = -thetas_degree[i]
        elif thetas_degree[i] >= 90 and thetas_degree[i] <= 180:
            # second quarter
            shifting_angle = 180-thetas_degree[i]
        elif thetas_degree[i] >= -180 and thetas_degree[i] <= -90:
            # third quarter
            shifting_angle = -180-thetas_degree[i]
        else:
            # fourth quarter:
            shifting_angle = -thetas_degree[i]

        # required rotation for aligning lines vertical
        rotation_mat = cv2.getRotationMatrix2D((N//2, M//2), shifting_angle * 1.35, 1)
        
        tmp_img_rot = cv2.warpAffine(tmp_img, rotation_mat, (N, M), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        img_vel_rot = cv2.warpAffine(img, rotation_mat, (N, M), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        line_pixels = np.nonzero(tmp_img_rot == 255)
        x_pixels = line_pixels[0]
        y_pixels = line_pixels[1]
        
        # votes for checking neighoubers that satisfies the condition
        votes = 0
        
        for j in range(len(x_pixels)-line_len):
            yi = y_pixels[j]    
            
            left_equivalent_pixel = 0
            right_equivalent_pixel = 0
            total_num = neigh_num * line_len
            for k in range(line_len):
                if yi-neigh_num >= 0 and yi+neigh_num < N:
                    left_equivalent_pixel += np.sum(img_vel_rot[x_pixels[j+k], yi-neigh_num:yi])
                    right_equivalent_pixel += np.sum(img_vel_rot[x_pixels[j+k], yi:yi+neigh_num])
                else:
                    total_num -= neigh_num
            if total_num == neigh_num * line_len:
                if abs(left_equivalent_pixel - right_equivalent_pixel) / total_num >= chess_square_thr:
                    votes += 1

        if votes >= voting_thr:
            print(votes)
            print("True, line passes chess area!")
            #print(left_equivalent_pixel / total_num, 
            #      right_equivalent_pixel / total_num)
            final_rhos.append(rhos[i])
            final_thetas.append(thetas[i])
    
    return final_rhos, final_thetas

#/ ------------------------- MAIN ------------------------ /#

img1 = cv2.imread('./im01.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('./im02.jpg', cv2.IMREAD_COLOR)

img1_edges = cv2.Canny(img1, 400, 500)
img2_edges = cv2.Canny(img2, 400, 500)

# threshold for converting edges to binary
bin_thr = 10
img2bin(img1_edges, bin_thr)
img2bin(img2_edges, bin_thr)

M1, N1 = img1_edges.shape
M2, N2 = img2_edges.shape

angle_num = 400
len_num = 400
hough_thr = 1e-2

#print("Finding hough space representation ...")
#voting_mat1, len_angle_spc1 = houghTran(img1_edges, len_num, angle_num, hough_thr)
#voting_mat2, len_angle_spc2 = houghTran(img2_edges, len_num, angle_num, hough_thr)
#hough_space = utl.scaleIntensities(voting_mat)

#np.save('voting1.npy', voting_mat1)
#np.save('space1.npy', len_angle_spc1)

#np.save('voting2.npy', voting_mat2)
#np.save('space2.npy', len_angle_spc2)


voting_mat1 = np.load('voting1.npy')
len_angle_spc1 = np.load('space1.npy')

voting_mat2 = np.load('voting2.npy')
len_angle_spc2 = np.load('space2.npy')


max_indices1 = utl.findLocalMax(voting_mat1, 0.48 * np.amax(voting_mat1), 0.1)
max_indices2 = utl.findLocalMax(voting_mat2, 0.48 * np.amax(voting_mat2), 0.1)

#print(max_indices.shape)
rhos1 = max_indices1[0, :]
angles1 = max_indices1[1, :]

rhos2 = max_indices2[0, :]
angles2 = max_indices2[1, :]
#print(angles)

#img_m1 = lineDrawer(img1, len_angle_spc1[rhos1, angles1, 0], len_angle_spc1[rhos1, angles1, 1], 2)
#utl.showImg(img_m1, 0.5, 'all lines1', False)

#img_m2 = lineDrawer(img2, len_angle_spc2[rhos2, angles2, 0], len_angle_spc2[rhos2, angles2, 1], 2)
#utl.showImg(img_m2, 0.5, 'all lines2')

# threshold for angles to be considered as one line
common_lines = 20

distinct_angles1, angles_appear1 = findDistinctAngles(angles1, common_lines)
distinct_angles2, angles_appear2 = findDistinctAngles(angles2, common_lines)


# how many choices of threshold is available for repeatness of lines
possible_choices = 4

print("Raw rhos and thetas: ")        
print(distinct_angles1)
print(angles_appear1)

print(distinct_angles2)
print(angles_appear2)
print("--------------------------")

angle_thr_ind1 = np.unravel_index(np.argsort(angles_appear1, axis=None)[-possible_choices:], 
                                 len(angles_appear1))[0][0]
app_thr1 = angles_appear1[angle_thr_ind1]

angle_thr_ind2 = np.unravel_index(np.argsort(angles_appear2, axis=None)[-possible_choices:], 
                                 len(angles_appear2))[0][0]
app_thr2 = angles_appear2[angle_thr_ind2]

print("app_thr1: {}, app_thr2: {}".format(app_thr1, app_thr2))
print("--------------------------")

rhos1, thetas1 = findParallelLines(max_indices1, distinct_angles1, 
                                   angles_appear1, common_lines, app_thr1, len_angle_spc1)

rhos2, thetas2 = findParallelLines(max_indices2, distinct_angles2, 
                                   angles_appear2, common_lines, app_thr2, len_angle_spc2)

#print("Parallel lines: ")
#print(rhos1)
#print(thetas1)

#print(rhos2)
#print(thetas2)
#print("--------------------------")

#img_m1 = lineDrawer(img1, rhos1, thetas1, 2)
#img_m2 = lineDrawer(img2, rhos2, thetas2, 2)

#utl.showImg(img_m1, 0.5, 'reduced lines1', False)
#utl.showImg(img_m2, 0.5, 'reduced lines2')


img_hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
img_vel1 = img_hsv1[:, :, -1]

img_hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
img_vel2 = img_hsv2[:, :, -1]

# thresholds to check if a line passes through chess area
line_len = 20
neigh_num = 20
chess_square_thr = 120
voting_thr = 80

print("Img 1 lines")
final_rhos1, final_thetas1 = passesChessArea(img_vel1, rhos1, thetas1, line_len, 
                                             neigh_num, chess_square_thr, voting_thr)

print("Img 2 lines")
final_rhos2, final_thetas2 = passesChessArea(img_vel2, rhos2, thetas2, line_len, 
                                             neigh_num, chess_square_thr, voting_thr)

img_m1 = lineDrawer(img1, final_rhos1, final_thetas1, 2)
img_m2 = lineDrawer(img2, final_rhos2, final_thetas2, 2)

utl.showImg(img_m1, 0.5, 'final lines1', False)
utl.showImg(img_m2, 0.5, 'final lines2')


"""
cv2.imwrite('lines-found.jpg', img_cop)

print("<><><><><><><><><><><><><><><><>")
for i in range(len(final_thetas)):
    tmp_img1 = np.zeros((M, N), dtype=img.dtype)
    pt1, pt2 = convertToXY(final_rhos[i], final_thetas[i], M, N)
    cv2.line(tmp_img1, (pt1[0], pt1[1]), (pt2[0], pt2[1]), 255, 1)
    
    for j in range(len(final_thetas)):
        if j != i:
            tmp_img2 = np.zeros((M, N), dtype=img.dtype)
            pt3, pt4 = convertToXY(final_rhos[j], final_thetas[j], M, N)
            cv2.line(tmp_img2, (pt3[0], pt3[1]), (pt4[0], pt4[1]), 255, 1)
            
            # finding intersection:
            x_intr, y_intr = np.nonzero(tmp_img1 & tmp_img2)
            if x_intr.size <= 0:
                tmp_img2[:, 1:] = tmp_img2[:, 0:-1]
                x_intr, y_intr = np.nonzero(tmp_img1 & tmp_img2)
                if x_intr.size <= 0:
                    # no intersection can be found
                    continue

            cv2.circle(img, (y_intr[0], x_intr[0]), 5, (0, 0, 255), -1)


cv2.imwrite('dots-found.jpg', img)

"""
print("Done!")     
