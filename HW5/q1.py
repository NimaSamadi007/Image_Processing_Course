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
                           [309, 190]])
img2_landmarks = np.array([[167, 170], [240, 169], [175, 243], [239, 242],
                           [200, 178], [201, 209], [144, 248], [176, 279],
                           [208, 295], [254, 272], [283, 246], [121, 196],
                           [303, 193]])


# utl.showImg(img1, 1, 'img1', False)
# utl.showImg(img2, 1, 'img2')

# making triangles in the first image
triangle_obj = cv.Subdiv2D([0, 0, N, M])
# add landmark points
for point in img1_landmarks:
    triangle_obj.insert(point.astype(float))

triangles_list1 = triangle_obj.getTriangleList()

tri_vertices1 = triangles_list1.reshape(-1, 3, 2).astype(int)
tri_vertices2 = np.zeros(tri_vertices1.shape, int)

# print(triangles_list.reshape(15, 3, 2))
for i in range(tri_vertices1.shape[0]):
    for j in range(3):
        index = np.nonzero((img1_landmarks[:, 0] == tri_vertices1[i, j, 0]) & 
                           (img1_landmarks[:, 1] == tri_vertices1[i, j, 1])) 
        tri_vertices2[i, j] = np.copy(img2_landmarks[index])

#drawing triangles
triangles_list2 = tri_vertices2.reshape(15, -1)
img1_tri = drawTriangles(img1, triangles_list1)
img2_tri = drawTriangles(img2, triangles_list2)

utl.showImg(img1_tri, 1, 'tris', False)
utl.showImg(img2_tri, 1, 'tris 2')
