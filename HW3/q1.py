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
        if (i % 10) == 0:
            print("In row x={}".format(i))
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

    distinct_angles.append(angles[0])
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

def passesChessArea(img, rhos, thetas, line_len, neigh_num, chess_square_thr):
    "Finds lines that pass through the chess area"
    final_rhos = []
    final_thetas = []
    thetas_degree = (thetas / (np.pi) * 180)

    # the value of image is passed
    M, N = img.shape
    votes_arr = []
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
        rotation_mat = cv2.getRotationMatrix2D((N//2, M//2), shifting_angle * (1.35), 1)
        
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

        votes_arr.append(votes)        
        
    # a fraction of average is used as voting threshold
    voting_thr = 0.9 * np.sum(votes_arr) / len(votes_arr)
    for i in range(len(votes_arr)):
        if votes_arr[i] >= voting_thr:
            final_rhos.append(rhos[i])
            final_thetas.append(thetas[i])
        
    return final_rhos, final_thetas

def calMaxColorThreshold(img_vel, step):
    "Calculates maximum difference between black and white colors in predefined squares"
    M, N = img_vel.shape
    max_diff = 0
    for i in range(0, M, step):
        for j in range(0, N, step):
            square = img_vel[i:i+step, j:j+step] 
            max_diff += (np.amax(square) - np.amin(square))
    
    return max_diff / (M / step * N / step)

def drawIntersectionPoints(in_img, rhos, thetas):
    "Draws intersection points of every possible two lines"
    img = np.copy(in_img)
    M, N, _ = img.shape
    for i in range(len(thetas)):
        tmp_img1 = np.zeros((M, N), dtype=img.dtype)
        pt1, pt2 = convertToXY(rhos[i], thetas[i], M, N)
        cv2.line(tmp_img1, (pt1[0], pt1[1]), (pt2[0], pt2[1]), 255, 1)
        
        for j in range(len(thetas)):
            if j != i:
                tmp_img2 = np.zeros((M, N), dtype=img.dtype)
                pt3, pt4 = convertToXY(rhos[j], thetas[j], M, N)
                cv2.line(tmp_img2, (pt3[0], pt3[1]), (pt4[0], pt4[1]), 255, 1)
                
                # finding intersection:
                x_intr, y_intr = np.nonzero(tmp_img1 & tmp_img2)
                if x_intr.size <= 0:
                    tmp_img2[:, 1:] = tmp_img2[:, 0:-1]
                    x_intr, y_intr = np.nonzero(tmp_img1 & tmp_img2)
                    if x_intr.size <= 0:
                        # no intersection can be found, go next round
                        continue
                # draw the intersection point
                cv2.circle(img, (y_intr[0], x_intr[0]), 5, (0, 0, 255), -1)
    
    return img

def isLineSimmilar(rho1, theta1, rho2, theta2, rho_thr, theta_thr):
    "Checks if two lines are simillar in img i.e. they are same line"
    if (abs(theta1 - theta2) <= theta_thr) and (abs(rho1 - rho2) <= rho_thr):
        return True
    elif (abs(theta1 - theta2 - np.pi) <= theta_thr) and (abs(rho1 + rho2) <= rho_thr):
        return True
    elif (abs(theta1 - theta2 + np.pi) <= theta_thr) and (abs(rho1 + rho2) <= rho_thr):
        return True
    else:
        return False

def deleteSimillarLines(rhos, thetas, rho_thr, theta_thr):
    "Deletes same line, simillar to NMS algorithm"
    deleted = np.zeros(len(thetas), dtype=int)
    for i in range(len(thetas)):
        if not deleted[i]:
            for j in range(i+1, len(thetas)):
            # check if the lines are simillar within a threshold:
                if isLineSimmilar(rhos[i], thetas[i], rhos[j], thetas[j], rho_thr, theta_thr):
                    deleted[j] = 1
    # Now using 'deleted' array, delete redundant elements
    final_thetas = []
    final_rhos = []
    for i in range(len(deleted)):
        if not deleted[i]:
            final_thetas.append(thetas[i])
            final_rhos.append(rhos[i])
    return final_rhos, final_thetas
#/ ---------------------------------------- MAIN ---------------------------------------------- /#

img1 = cv2.imread('./im01.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('./im02.jpg', cv2.IMREAD_COLOR)

img1_edges = cv2.Canny(img1, 400, 500)
img2_edges = cv2.Canny(img2, 400, 500)

# threshold for converting edges to binary
bin_thr = 10
img2bin(img1_edges, bin_thr)
img2bin(img2_edges, bin_thr)

cv2.imwrite('res01.jpg', img1_edges)
cv2.imwrite('res02.jpg', img2_edges)

M1, N1 = img1_edges.shape
M2, N2 = img2_edges.shape

angle_num = 400
len_num = 400
hough_thr = 1e-2

print("Finding hough space representation ...")
print("Img1: ")
voting_mat1, len_angle_spc1 = houghTran(img1_edges, len_num, angle_num, hough_thr)
print("Img2: ")
voting_mat2, len_angle_spc2 = houghTran(img2_edges, len_num, angle_num, hough_thr)
print("Hough transform finding is finished")

print("Saving hough space img ...")
hough_space1 = utl.scaleIntensities(voting_mat1)
cv2.imwrite('res03-hough-space.jpg', hough_space1)
hough_space2 = utl.scaleIntensities(voting_mat2)
cv2.imwrite('res04-hough-space.jpg', hough_space2)

print("Drawing found lines in images and saving ...")
max_indices1 = utl.findLocalMax(voting_mat1, 0.47 * np.amax(voting_mat1), 0.1)
max_indices2 = utl.findLocalMax(voting_mat2, 0.47 * np.amax(voting_mat2), 0.1)

rhos1 = max_indices1[0, :]
angles1 = max_indices1[1, :]

rhos2 = max_indices2[0, :]
angles2 = max_indices2[1, :]

img1_raw_lines = lineDrawer(img1, len_angle_spc1[rhos1, angles1, 0], len_angle_spc1[rhos1, angles1, 1], 2)
cv2.imwrite('res05-lines.jpg', img1_raw_lines)
img2_raw_lines = lineDrawer(img2, len_angle_spc2[rhos2, angles2, 0], len_angle_spc2[rhos2, angles2, 1], 2)
cv2.imwrite('res06-lines.jpg', img2_raw_lines)

print("Finding distinct lines ...")
# threshold for angles to be considered as one line
common_lines = angle_num / 20

distinct_angles1, angles_appear1 = findDistinctAngles(angles1, common_lines)
distinct_angles2, angles_appear2 = findDistinctAngles(angles2, common_lines)

# how many choices of threshold is available for repeatness of lines
possible_choices = 4

angle_thr_ind1 = np.unravel_index(np.argsort(angles_appear1, axis=None)[-possible_choices:], 
                                 len(angles_appear1))[0][0]
app_thr1 = angles_appear1[angle_thr_ind1]

angle_thr_ind2 = np.unravel_index(np.argsort(angles_appear2, axis=None)[-possible_choices:], 
                                 len(angles_appear2))[0][0]
app_thr2 = angles_appear2[angle_thr_ind2]


rhos1, thetas1 = findParallelLines(max_indices1, distinct_angles1, 
                                   angles_appear1, common_lines, app_thr1, len_angle_spc1)

rhos2, thetas2 = findParallelLines(max_indices2, distinct_angles2, 
                                   angles_appear2, common_lines, app_thr2, len_angle_spc2)

print("Deleting lines that don't pass through chess area ...")
img_hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
img_vel1 = img_hsv1[:, :, -1]

img_hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
img_vel2 = img_hsv2[:, :, -1]

# thresholds to check if a line passes through chess area
line_len = 15
neigh_num = 15

chess_square_thr1 = calMaxColorThreshold(img_vel1, int(1.7*(line_len + neigh_num)))
chess_square_thr2 = calMaxColorThreshold(img_vel2, int(1.7*(line_len + neigh_num)))

rhos1, thetas1 = passesChessArea(img_vel1, rhos1, thetas1, line_len, 
                                 neigh_num, chess_square_thr1)
rhos2, thetas2 = passesChessArea(img_vel2, rhos2, thetas2, line_len, 
                                 neigh_num, chess_square_thr2)


final_rhos1, final_thetas1 = deleteSimillarLines(rhos1, thetas1, 0.05, 0.05)
final_rhos2, final_thetas2 = deleteSimillarLines(rhos2, thetas2, 0.05, 0.05)

print("Drawing final lines in images and saving ...")
img_m1 = lineDrawer(img1, final_rhos1, final_thetas1, 2)
img_m2 = lineDrawer(img2, final_rhos2, final_thetas2, 2)

cv2.imwrite('res07-chess.jpg', img_m1)
cv2.imwrite('res08-chess.jpg', img_m2)

print("Finding intersections and drawing them ...")
img_dots1 = drawIntersectionPoints(img1, final_rhos1, final_thetas1)
img_dots2 = drawIntersectionPoints(img2, final_rhos2, final_thetas2)

cv2.imwrite('res09-corners.jpg', img_dots1)
cv2.imwrite('res10-corners.jpg', img_dots2)
print("Done!")     
