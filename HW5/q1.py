import numpy as np
import cv2 as cv
import utils as utl

#---------------------------------- FUNCTIONS --------------------------------#
def drawTriangles(img, triangles_list):
    "Draws triangles based on triangle_list (output of opencv Subdiv2D)"
    img = np.copy(img)
    for i in range(triangles_list.shape[0]):
        vertices = triangles_list[i].reshape(3, 2).astype(np.int64)
        for j in range(3):
            cv.line(img, vertices[j], vertices[(j+1)%3], (0, 0, 255), 1)
    return img

#--------------------------------- MAIN --------------------------------------#

img1 = cv.imread('res01.jpg', cv.IMREAD_COLOR)
img2 = cv.imread('res02.jpg', cv.IMREAD_COLOR)

M, N, _ = img1.shape

# landmark points
img1_landmarks = np.array([[191, 171], [262, 167], [201, 245], [260, 243],
                           [226, 178], [228, 207], [170, 245], [192, 278],
                           [228, 291], [264, 277], [290, 245], [147, 199],
                           [309, 190], [0, 0], [N-1, 0], [0, M-1], [N-1, M-1],
                           [N//2-1, 0], [0, M//2-1], [N//2-1, M-1], [N-1, M//2-1]])
img2_landmarks = np.array([[167, 170], [240, 169], [175, 243], [239, 242],
                           [200, 178], [201, 209], [144, 248], [176, 279],
                           [208, 295], [254, 272], [283, 246], [121, 196],
                           [303, 193], [0, 0], [N-1, 0], [0, M-1], [N-1, M-1],
                           [N//2-1, 0], [0, M//2-1], [N//2-1, M-1], [N-1, M//2-1]])

# making triangles in the first image
triangle_obj = cv.Subdiv2D([0, 0, N, M])
# add landmark points
for point in img1_landmarks:
    triangle_obj.insert(point.astype(float))

triangles_list1 = triangle_obj.getTriangleList()
num_tris = triangles_list1.shape[0]
# img1_tri = drawTriangles(img1, triangles_list1)
# utl.showImg(img1_tri, 1)

tri_vertices1 = triangles_list1.reshape(-1, 3, 2).astype(int)
tri_vertices2 = np.zeros(tri_vertices1.shape, int)

for i in range(num_tris):
    for j in range(3):
        index = np.nonzero((img1_landmarks[:, 0] == tri_vertices1[i, j, 0]) & 
                           (img1_landmarks[:, 1] == tri_vertices1[i, j, 1])) 
        tri_vertices2[i, j] = np.copy(img2_landmarks[index])

#drawing triangles
triangles_list2 = tri_vertices2.reshape(num_tris, -1)

num_step = 45
final_warped_img1 = np.zeros((M, N, 3), np.uint8)  
final_warped_img2 = np.zeros((M, N, 3), np.uint8)  
mask = np.zeros((M, N), np.uint8)

steps = np.linspace(0, 1, num_step)
for t, step in enumerate(steps):
    print("At frame {}".format(t))
    inter_tri_vertices = ((1-step)*tri_vertices1.astype(float) + 
                          step*tri_vertices2.astype(float)).astype(int)
    
    # warping both images toward average triangles
    final_warped_img1[:, :, :] = 0
    final_warped_img2[:, :, :] = 0
    for i in range(num_tris):  
        # warping img1 -> img2
        affine_tran1 = cv.getAffineTransform(tri_vertices1[i].astype(np.float32), 
                                             inter_tri_vertices[i].astype(np.float32))
        warped_img = cv.warpAffine(img1, affine_tran1, (N, M))
        mask[:, :] = 0
        cv.drawContours(mask, [inter_tri_vertices[i]], 0, 255, -1)
        final_warped_img1[mask == 255] = warped_img[mask == 255]
        
        # warping img2 -> img1
        affine_tran2 = cv.getAffineTransform(tri_vertices2[i].astype(np.float32), 
                                             inter_tri_vertices[i].astype(np.float32))
        warped_img = cv.warpAffine(img2, affine_tran2, (N, M))
        mask[:, :] = 0
        cv.drawContours(mask, [inter_tri_vertices[i]], 0, 255, -1)
        final_warped_img2[mask == 255] = warped_img[mask == 255]
        
    morphed_img = (1-step) * final_warped_img1.astype(float) +\
                    (step) * final_warped_img2.astype(float)
    cv.imwrite('test/img-{}.jpg'.format(t), morphed_img.astype(np.uint8))
    cv.imwrite('test/img-{}.jpg'.format(2*num_step-1-t), morphed_img.astype(np.uint8))
  
    
    
