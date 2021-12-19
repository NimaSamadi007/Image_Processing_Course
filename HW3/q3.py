import numpy as np
import cv2 
import utils as utl


#------------------------ FUNCTIONS -------------------------- #
def removeHolesFromResult(match_result, hole_indices, extra_indices):
    "instead of holes in matching result, fills with 'inf' value to avoid selecting them"
    "hole_indices: dictionary of holes, each in the format of [left, right, top, bottom]"
    "extra_indices: other places that I ignore to enhance performance"
    M, N = match_result.shape
    for i in range(len(hole_indices)):
        if hole_indices[i][0] - M_p >= 0:
            x1 = hole_indices[i][0] - M_p
        else:
            x1 = 0
        
        if hole_indices[i][1] + M_p < M:
            x2 = hole_indices[i][1] + M_p
        else:
            x2 = M-1

        if hole_indices[i][2] - N_p >= 0:
            y1 = hole_indices[i][2] - N_p
        else:
            y1 = 0

        if hole_indices[i][3] + N_p < N:
            y2 = hole_indices[i][3] + N_p
        else:
            y2 = 0
        match_result[x1:x2, y1:y2] = np.inf
    for i in range(len(extra_indices)):
        if extra_indices[i][0] != -1:
            match_result[extra_indices[i][0]:extra_indices[i][1],
                         extra_indices[i][2]:extra_indices[i][3]] = np.inf    

def updateTexture(syn_tex, tex, M_p, N_p, x_s, y_s, x_thr, y_thr, 
                  rand_sel, hole_indices, extra_indices):
    "updates texture at each step"
    patch = np.copy(syn_tex[x_s:x_s+M_p, y_s:y_s+N_p, :])

    mask_template = np.ones(patch.shape, dtype=np.uint8)
    mask_template[x_thr:, y_thr:, :] = np.zeros((M_p-x_thr, N_p-y_thr, 3), dtype=np.uint8)
    
    matching_result = cv2.matchTemplate(tex, patch, cv2.TM_SQDIFF_NORMED, mask=mask_template)
    
    # remove results of holes:
    removeHolesFromResult(matching_result, hole_indices, extra_indices)

    matching_indices = np.unravel_index(np.argsort(matching_result, axis=None)[0:rand_sel], 
                                                    matching_result.shape)
    
    random_index = np.random.randint(low=0, high=rand_sel)
    matching_indices = np.array([matching_indices[0][random_index], matching_indices[1][random_index]])
    
    found_patch = tex[matching_indices[0]:matching_indices[0]+M_p, 
                        matching_indices[1]:matching_indices[1]+N_p]

    ssd_result_x = (patch[0:x_thr, :, :].astype(np.float64) - found_patch[0:x_thr, :, :].astype(np.float64))**2
    ssd_result_x = np.sum(ssd_result_x, axis=2)

    ssd_result_y = (patch[:, 0:y_thr, :].astype(np.float64) - found_patch[:, 0:y_thr, :].astype(np.float64))**2
    ssd_result_y = np.sum(ssd_result_y, axis=2)

    path_x = utl.findMinCut(ssd_result_x, mode="COL")
    path_y = utl.findMinCut(ssd_result_y, mode="ROW")

    # replace found patch (next it will be corrected)
    syn_tex[x_s:x_s+M_p, y_s:y_s+N_p, :] = found_patch

    mat_path_x = np.zeros((M_p, N_p), dtype=int)
    mat_path_y = np.zeros((M_p, N_p), dtype=int)

    for i in range(N_p):
        mat_path_x[path_x[0, i], i] = 1
    
    for i in range(M_p):
        mat_path_y[i, path_y[i, 0]] = 1

    # mat_final showes boundries of patches
    tmp_mat = mat_path_x[0:x_thr, 0:y_thr] & mat_path_y[0:x_thr, 0:y_thr]
    common_points = np.nonzero(tmp_mat == 1)
    if common_points[0].shape[0]:
        common_point = (common_points[0][-1], common_points[1][-1])

        mat_path_x[:, 0:common_point[1]] = 0
        mat_path_y[0:common_point[0], :] = 0   
    else:
        # no collision found - shift one to the right and check again
        mat_shifted = np.concatenate([np.zeros((x_thr, 1), dtype=int),
                                        mat_path_x[0:x_thr, 0:y_thr-1]], axis=1)
        
        tmp_mat = mat_shifted & mat_path_y[0:x_thr, 0:y_thr]
        common_points = np.nonzero(tmp_mat == 1)
        common_point = (common_points[0][-1], common_points[1][-1])

        mat_path_x[:, 0:common_point[1]] = 0
        mat_path_y[0:common_point[0], :] = 0   
        
    mat_final = mat_path_x | mat_path_y
        
    # traverse for each row
    for i in range(M_p):
        flag = False
        for j in range(N_p):
            # we have reached the boundry
            if mat_final[i, j] == 1:
                flag = True
                syn_tex[x_s+i, y_s:y_s+j] = patch[i, 0:j]
                break
        if not flag:
            # no boundry is found:
            syn_tex[x_s+i, y_s:y_s+N_p] = patch[i, :]

    for j in range(N_p):
        flag = False    
        for i in range(M_p):
            if mat_final[i, j] == 1:
                flag = True
                syn_tex[x_s:x_s+i, y_s+j] = patch[0:i, j]
                break
        if not flag:
            syn_tex[x_s:x_s+M_p, y_s+j] = patch[:, j]

def runQ3(img_name, holes, extra_indices, x_thr, y_thr, M_p, N_p, random_select):
    "runs complete algorithm for each image"
    img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    print("Running algorithm for {}".format(img_name))
    for k in range(len(holes)):
    # hole to be filled
        img_hole = np.copy(img[holes[k][0]-M_p:holes[k][1]+M_p, holes[k][2]-N_p: holes[k][3]+N_p, :])
        M_i, N_i, _ = img_hole.shape
        print("Filling hole {}".format(k))
        for i in range(1, (M_i - M_p) // (M_p - x_thr) + 1):
            for j in range(1, (N_i - N_p) // (N_p - y_thr) + 1):
                x_end = i*M_p-(i-1)*(x_thr)
                x_start = x_end - x_thr
                y_end = j*N_p-(j-1)*(y_thr)
                y_start = y_end - y_thr

                updateTexture(img_hole, img, M_p, N_p,
                            x_start, y_start, x_thr, y_thr, 
                            random_select, holes, extra_indices)

        img[holes[k][0]-M_p:holes[k][1]+M_p, holes[k][2]-N_p:holes[k][3]+N_p, :] = img_hole

    # filter to reduce borders
    img = cv2.GaussianBlur(img, (3, 3), 1, borderType=cv2.BORDER_CONSTANT)

    return img
# ------------------------ MAIN ------------------------------- #

################### parameters #########################
# hole borders dictionary in the [x_top, x_bottom, y_left, y_right] format

holes = {0:[54, 167, 327, 543], 1:[614, 710, 1135, 1252], 2:[743, 931, 824, 976]}
#extra places to ignore chosing from them
extra_indices = {0:[0, 300, 650, 1400]}
x_thr = 10
y_thr = 10
M_p, N_p = 40, 40
random_select = 1

img_filled = runQ3('im03-hole.jpg', holes, extra_indices, x_thr, y_thr, M_p, N_p, random_select)
cv2.imwrite('res15.jpg', img_filled)

#im04 holes border:
holes = {0:[684, 1170, 728, 970]}
# consider all possible choices
extra_indices = {0:[-1, -1, -1, -1]}
x_thr = 25
y_thr = 25
M_p, N_p = 100, 100
random_select = 5

img_filled = runQ3('im04-hole.jpg', holes, extra_indices, x_thr, y_thr, M_p, N_p, random_select)
cv2.imwrite('res16.jpg', img_filled)
