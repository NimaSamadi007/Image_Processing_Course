import numpy as np
import cv2 
import utils as utl

#----------------------- FUNCTIONS ------------------------------- #

def updateTexture(syn_tex, tex, M_p, N_p, M_i, N_i, 
                  x_s, x_f, y_s, y_f, x_thr, y_thr, rand_sel):
    "updates texture at each step"
    "syn_text: synthesized texture to be updated at each step"
    "tex: original texture"
    "(M_p, N_p): patches shape"
    "M_i, N_i, synthesizing image size"
    "x_s & x_f: start and final x indices"
    "y_s & y_f: start and final y indices"
    "x_thr & y_thr: x and y threshold used in extracting patches"
    "rand_sel: number of random choices for choosing patch at each step"

    # only row synthesizing
    if y_s and y_f and (not x_s) and (not x_f) :
        patch_y = syn_tex[x_s:x_s+M_p, y_s:y_f, :]
        matching_result_y = cv2.matchTemplate(tex, patch_y, cv2.TM_SQDIFF_NORMED)
        # clip matching result to ensure size consistency
        matching_result_y = matching_result_y[:, :-(N_p-y_thr)]

        matching_indices = np.unravel_index(np.argsort(matching_result_y, axis=None)[:rand_sel], 
                                            matching_result_y.shape)

        random_index = np.random.randint(low=0, high=rand_sel)
        matching_indices = np.array([matching_indices[0][random_index], matching_indices[1][random_index]])

        found_patch_y = tex[matching_indices[0]:matching_indices[0]+M_p, matching_indices[1]:matching_indices[1]+N_p]

        ssd_result_y = (patch_y.astype(np.float64) - found_patch_y[:, 0:y_thr, :].astype(np.float64))**2 
        ssd_result_y = np.sum(ssd_result_y, axis=2)

        path_y = utl.findMinCut(ssd_result_y, mode="ROW")
        
        for k in range(M_p):
            part1 = syn_tex[k, y_s : y_s+path_y[k, 0], :]
            part2 = found_patch_y[k, path_y[k, 0]:y_thr, :]
            syn_tex[k, y_s:y_f] = np.concatenate([part1, part2], axis=0)
        if y_f + N_p - y_thr > N_i:
            syn_tex[x_s:x_s+M_p, y_f:, :] = found_patch_y[:, y_thr:y_thr+N_i-y_f, :]  
        else:
            syn_tex[x_s:x_s+M_p, y_f:y_f+N_p-y_thr, :] = found_patch_y[:, y_thr:, :]

    # only column synthesizing
    elif x_s and x_f and (not y_s) and (not y_f):
        patch_x = syn_tex[x_s:x_f, y_s:y_s+N_p, :]
        matching_result_x = cv2.matchTemplate(tex, patch_x, cv2.TM_SQDIFF_NORMED)
        matching_result_x = matching_result_x[:-(M_p-x_thr), :]

        matching_indices = np.unravel_index(np.argsort(matching_result_x, axis=None)[:rand_sel], 
                                            matching_result_x.shape)

        random_index = np.random.randint(low=0, high=rand_sel)
        matching_indices = np.array([matching_indices[0][random_index], matching_indices[1][random_index]])
        
        found_patch_x = tex[matching_indices[0]:matching_indices[0]+M_p, matching_indices[1]:matching_indices[1]+N_p]
        ssd_result_x = (patch_x.astype(np.float64) - found_patch_x[0:x_thr, :, :].astype(np.float64))**2 
        ssd_result_x = np.sum(ssd_result_x, axis=2)

        path_x = utl.findMinCut(ssd_result_x, mode="COL")

        for k in range(N_p):
            part1 = syn_tex[x_s : x_s+path_x[0, k], k, :]
            part2 = found_patch_x[path_x[0, k]:x_thr, k, :]
            syn_tex[x_s:x_f, k] = np.concatenate([part1, part2], axis=0)
        if x_f + M_p - x_thr > M_i:  
            syn_tex[x_f:, y_s:y_s+N_p, :] = found_patch_x[x_thr:x_thr+M_i-x_f, :, :]  
        else:
            syn_tex[x_f:x_f+M_p-x_thr, y_s:y_s+N_p, :] = found_patch_x[x_thr:, :, :]

    # L shape senthsizing:
    else:
        patch = np.copy(syn_tex[x_s:x_s+M_p, y_s:y_s+N_p, :])

        mask_template = np.ones(patch.shape, dtype=np.uint8)
        mask_template[x_thr:, y_thr:, :] = np.zeros((M_p-x_thr, N_p-y_thr, 3), dtype=np.uint8)
        
        matching_result = cv2.matchTemplate(tex, patch, cv2.TM_SQDIFF_NORMED, mask=mask_template)
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

#/ ------------------- MAIN --------------- /#

img_index = 13
texture = cv2.imread('./Textures/texture{}.jpg'.format(img_index), cv2.IMREAD_COLOR)
print("Image loaded successfully, starting algorithm ...")
# PARAMETERS:

# texture size:
M_t, N_t, _ = texture.shape
# synthesized texture size:
M_i, N_i = 2500, 2500
syn_texture = np.zeros((M_i, N_i, 3), dtype=np.uint8)
# patch size
M_p, N_p = 100, 100

random_select = 5
y_thr = 25
x_thr = 25

# generalizing implementation:
for i in range((M_i - M_p) // (M_p - x_thr) + 1):
    print("Synthesizing row {} ...".format(i))
    for j in range((N_i - N_p) // (N_p - y_thr) + 1):
        # for general case:
        x_end = i*M_p-(i-1)*(x_thr)
        x_start = x_end - x_thr
        y_end = j*N_p-(j-1)*(y_thr)
        y_start = y_end - y_thr

        # first tile:
        if i == 0 and j == 0:
            # randomly select one patch from texture:
            x0 = np.random.randint(low=0, high=M_t-M_p)
            y0 = np.random.randint(low=0, high=N_t-N_p)
            syn_texture[0:M_p, 0:N_p, :] = texture[x0:x0+M_p, y0:y0+N_p, :]

        # first row:
        elif i == 0 and j != 0:
            x_end = 0
            x_start = 0
            updateTexture(syn_texture, texture, M_p, N_p,
                          M_i, N_i, x_start, x_end, y_start, y_end,
                          x_thr, y_thr, random_select)
        # first col:
        elif i != 0 and j == 0:
            y_end = 0
            y_start = 0
            updateTexture(syn_texture, texture, M_p, N_p,
                          M_i, N_i, x_start, x_end, y_start, y_end,
                          x_thr, y_thr, random_select)
        # L shape tiles:
        else:
            updateTexture(syn_texture, texture, M_p, N_p,
                        M_i, N_i, x_start, x_end, y_start, y_end,
                        x_thr, y_thr, random_select)
            
cv2.imwrite('texture{}.jpg'.format(img_index), syn_texture)
print("Done!")