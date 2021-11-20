import numpy as np
import cv2
import utils as utl

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

    book_i_img = utl.myWarpFunction(books_img, transformation_matrix, (M, N))

    cv2.imwrite("res{}.jpg".format(i+16), book_i_img)

    if i != (NUM_OF_BOOKS-1):
        print("Book {} warped successfully, going for book {} ...".format(i, i+1))
    else:
        print("All books were warped. Done!")
