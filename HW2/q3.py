import numpy as np
import cv2

## -------------------------- FUNCIONS ----------------------- ##
def showImg(img, res_factor, title='input image'):
    res = (int(img.shape[1]*res_factor), int(img.shape[0]*res_factor))
    img_show = cv2.resize(img, res)
    cv2.imshow(title, img_show)
    return

def myWarpFunction(img, trans_matrix, dsize):
    # opencv format
    # N: width, M: height of the image
    M, N = dsize
    warped_img = np.zeros((M, N, 3), dtype=np.uint8)
    inverse_M = np.linalg.inv(trans_matrix)
    for i in range(M): # i: height (y in opencv, x in numpy)
        for j in range(N): # j: width (x in opencv, y in numpy)
            corr_pixel = inverse_M @ np.array([j, i, 1], dtype=np.float64).reshape(3, 1)
            corr_pixel = np.array([corr_pixel[0], corr_pixel[1]]) / corr_pixel[2]
            # in numpy format, assign each pixel in warped image
            assignPixels(img, warped_img, corr_pixel, i, j)
    
    return warped_img

def assignPixels(img, warped_img, corr_pixel, i, j):
    # assigns warped image pixels for each channel from 
    # original image
    # x, y are in opencv format
    y = int(corr_pixel[0])
    x = int(corr_pixel[1])
    a = corr_pixel[0] - y
    b = corr_pixel[1] - x
    A = np.array([1-a, a], dtype=np.float64).reshape(1, 2)
    B = np.array([1-b, b], dtype=np.float64).reshape(2, 1)
    for k in range(3):
        img_mat = np.array([[img[x, y, k], img[x+1, y, k]], 
                            [img[x, y+1, k], img[x+1, y+1, k]]])     
        warped_img[i, j, k] = (A @ img_mat @ B).astype(np.uint8)
    return

## -------------------------- MAIN ----------------------- ##


print("Starting program ...")
books_img = cv2.imread('./books.jpg', cv2.IMREAD_COLOR)
print("Image was loaded successfully")

# books position format: (upper left, upper right, lower right, lower left)
book1_pos = np.array([[667, 207], [601, 397], [319, 288], [384, 104]])
book2_pos = np.array([[813, 970], [611, 1100], [423, 798], [624, 668]])
book3_pos = np.array([[351, 740], [155, 707], [206, 428], [403, 466]])
books_pos = [book1_pos, book2_pos, book3_pos]

# fixed ratio to scale warped image
FIXED_RATIO = 2
NUM_OF_BOOKS = 3

for i in range(NUM_OF_BOOKS):
    print("Warping book {} ...".format(i))
    width1 = FIXED_RATIO * np.round(np.sqrt(np.sum((books_pos[i][1] - books_pos[i][0]) ** 2)))
    width2 = FIXED_RATIO * np.round(np.sqrt(np.sum((books_pos[i][3] - books_pos[i][2]) ** 2)))
    height1 = FIXED_RATIO * np.round(np.sqrt(np.sum((books_pos[i][2] - books_pos[i][1]) ** 2)))
    height2 = FIXED_RATIO * np.round(np.sqrt(np.sum((books_pos[i][3] - books_pos[i][0]) ** 2)))

    height = (height1 + height2) // 2
    width = (width1 + width2) // 2

    book_i_img = np.zeros((int(height), int(width), 3), dtype=books_img.dtype)
    M, N,_ = book_i_img.shape
    # ith book corresponded position
    book_i_corr_pos = np.array([[0, 0], [N-1, 0], [N-1, M-1], [0, M-1]])

    transformation_matrix = cv2.getPerspectiveTransform(books_pos[i].astype(np.float32), 
                                                        book_i_corr_pos.astype(np.float32))

    book_i_img = myWarpFunction(books_img, transformation_matrix, (M, N))

    cv2.imwrite("res{}.jpg".format(i+16), book_i_img)

    if i != (NUM_OF_BOOKS-1):
        print("Book {} warped successfully, going for book {} ...".format(i, i+1))
    else:
        print("All books were warped. Done!")
