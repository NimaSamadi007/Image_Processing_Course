import numpy as np
import cv2 
import utils as utl


#------------------------ FUNCTIONS -------------------------- #
def updateTexture(syn_tex, tex, M_p, N_p, M_i, N_i, x_left, x_right, y_top, y_bottom,
                  x_s, x_f, y_s, y_f, x_thr, y_thr, rand_sel):
    "updates texture at each step"
    "syn_text: synthesized texture to be updated at each step"
    "tex: original texture"
    "(M_p, N_p): patches shape"
    "M_i, N_i: synthesized image size"
    "x_left, x_right, y_top, y_bottom: hole borders"
    "x_s & x_f: start and final x indices"
    "y_s & y_f: start and final y indices"
    "x_thr & y_thr: x and y threshold used in extracting patches"
    "rand_sel: number of random choices for choosing patch at each step"

    patch = np.copy(syn_tex[x_s:x_s+M_p, y_s:y_s+N_p, :])

    mask_template = np.ones(patch.shape, dtype=np.uint8)
    mask_template[x_thr:, y_thr:, :] = np.zeros((M_p-x_thr, N_p-y_thr, 3), dtype=np.uint8)
    
    matching_result = cv2.matchTemplate(tex, patch, cv2.TM_SQDIFF_NORMED, mask=mask_template)
    # print(tex.shape)
    # print(patch.shape)
    # print(matching_result.shape)
    
    matching_result[x_left-2*x_thr:x_right+2*x_thr, y_top-2*y_thr:y_bottom+2*y_thr] = np.inf
    
    matching_indices = np.unravel_index(np.argsort(matching_result, axis=None)[0:rand_sel], 
                                                    matching_result.shape)
    
    
    random_index = np.random.randint(low=0, high=rand_sel)
    matching_indices = np.array([matching_indices[0][random_index], matching_indices[1][random_index]])
    print(matching_indices)

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

    repr = np.zeros((M_p, N_p, 3), dtype=int)
    repr[:, :, 0] = mat_path_x
    repr[:, :, 2] = mat_path_y
    repr *= 255
    
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

# ------------------------ MAIN ------------------------------- #
img = cv2.imread("im04-hole.jpg", cv2.IMREAD_COLOR)

################### parameters #########################
# hole borders
x_left = 684
x_right = 1170
y_top = 728
y_bottom = 970
# overlapping threshold
x_thr = 25
y_thr = 25
# patch size 
M_p, N_p = 100, 100

random_select = 1

img_hole = img[x_left-M_p:x_right+M_p, y_top-N_p: y_bottom+N_p, :]

M_i, N_i, _ = img_hole.shape
#########################################################

# just senthesizing L shape patches
for i in range(1, (M_i - M_p) // (M_p - x_thr) + 1):
    for j in range(1, (N_i - N_p) // (N_p - y_thr) + 1):
        x_end = i*M_p-(i-1)*(x_thr)
        x_start = x_end - x_thr
        y_end = j*N_p-(j-1)*(y_thr)
        y_start = y_end - y_thr

        # if i == 1 and j == 1:
        updateTexture(img_hole, img, M_p, N_p,
                        M_i, N_i, x_left, x_right, y_top, y_bottom, 
                        x_start, x_end, y_start, y_end,
                        x_thr, y_thr, random_select)

utl.showImg(img_hole, 1)