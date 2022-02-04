import numpy as np
import cv2 as cv
import utils as utl

#---------------------------------- FUNCTIONS --------------------------------#
def fileReader(path):
    "Reads the points file - each line contains a point on the image"
    data = []
    with open(path) as f:
        lines = f.readlines()
        points_str = [line.strip() for line in lines]
        for point_str in points_str:
            point = point_str.split(' ')
            point = list(map(int, point))
            data.append(point)
    
    return np.array(data, int)

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

# reading landmark points
img1_landmarks = fileReader('./res01-points.txt')
img2_landmarks = fileReader('./res02-points.txt')

# drawing points
img1_points = utl.drawPoints(img1, img1_landmarks, 3)
img2_points = utl.drawPoints(img2, img2_landmarks, 3)

# making triangles in the first image
triangle_obj = cv.Subdiv2D([0, 0, N, M])
# add landmark points
for point in img1_landmarks:
    triangle_obj.insert(point.astype(float))

triangles_list1 = triangle_obj.getTriangleList()
num_tris = triangles_list1.shape[0]

tri_vertices1 = triangles_list1.reshape(-1, 3, 2).astype(int)
tri_vertices2 = np.zeros(tri_vertices1.shape, int)

for i in range(num_tris):
    for j in range(3):
        index = np.nonzero((img1_landmarks[:, 0] == tri_vertices1[i, j, 0]) & 
                           (img1_landmarks[:, 1] == tri_vertices1[i, j, 1])) 
        tri_vertices2[i, j] = np.copy(img2_landmarks[index])

triangles_list2 = tri_vertices2.reshape(num_tris, -1)

#drawing triangles
img1_tris = drawTriangles(img1, triangles_list1)
img2_tris = drawTriangles(img2, triangles_list2)

num_step = 45
final_warped_img1 = np.zeros((M, N, 3), np.uint8)  
final_warped_img2 = np.zeros((M, N, 3), np.uint8)  
mask = np.zeros((M, N), np.uint8)

steps = np.linspace(0, 1, num_step)
# making frames
for t, step in enumerate(steps):
    print("Creating frame {}".format(t))
    inter_tri_vertices = ((1-step)*tri_vertices1.astype(float) + 
                          step*tri_vertices2.astype(float)).astype(int)
    
    # warping both images toward average triangles
    final_warped_img1[:, :, :] = 0
    final_warped_img2[:, :, :] = 0
    # warping corresponded triangles in each images:
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
    
    # morhping two image:        
    morphed_img = (1-step) * final_warped_img1.astype(float) +\
                    (step) * final_warped_img2.astype(float)
    # saving intermidate images for making gif file - Uncomment the following
    # to save intermediate images
    # cv.imwrite('test/img-{}.jpg'.format(t), morphed_img.astype(np.uint8))
    # cv.imwrite('test/img-{}.jpg'.format(2*num_step-1-t), morphed_img.astype(np.uint8))
    
    if t == 14:
        cv.imwrite('res03.jpg', morphed_img.astype(np.uint8))
    elif t == 29:
        cv.imwrite('res04.jpg', morphed_img.astype(np.uint8))
    
print("Done!")
    
