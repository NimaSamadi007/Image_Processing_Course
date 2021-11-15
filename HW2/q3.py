import numpy as np
import cv2

def showImg(img, res_factor, title='input image'):
    res = (int(img.shape[1]*res_factor), int(img.shape[0]*res_factor))
    img_show = cv2.resize(img, res)
    cv2.imshow(title, img_show)
    return

def myWarpFunction(img, trans_matrix, dsize):
    M, N = dsize
    warped_img = np.zeros((M, N, 3), dtype=np.uint8)
    inverse_M = np.linalg.inv(trans_matrix)
    
    for i in range(M):
        for j in range(N):
            corr_pixel = inverse_M @ np.array([i, j, 1], dtype=np.float64).reshape(3, 1)
            corr_pixel = np.array([corr_pixel[0], corr_pixel[1]]) / corr_pixel[2]
            x = int(corr_pixel[0])
            y = int(corr_pixel[1])
            a = corr_pixel[0] - x
            b = corr_pixel[1] - y
            A = np.array([1-a, a], dtype=np.float64).reshape(1, 2)
            B = np.array([1-b, b], dtype=np.float64).reshape(2, 1)
            for k in range(3):
                img_mat = np.array([[img[x, y, k], img[x, y+1, k]], [img[x+1, y, k], img[x+1, y+1, k]]])     
                warped_img[i, j, k] = A @ img_mat @ B
    
    return warped_img

def repeatRow(row, num):
    row_format = row.reshape(1, row.shape[0])
    repeater = np.ones((num, 1), dtype=row.dtype)
    return repeater @ row_format

def repeatCol(col, num):
    col_format = col.reshape(col.shape[0], 1)
    repeater = np.ones((1, num), dtype=col.dtype)
    return col_format @ repeater




books_img = cv2.imread('./books.jpg', cv2.IMREAD_COLOR)

print(books_img.shape)

# books position format: (upper left, upper right, lower right, lower left)

# book1_pos = np.array([[331, 105], [298, 200], [158, 144], [190, 52]]) * 2
book1_pos = np.array([[667, 207], [601, 397], [319, 288], [384, 104]])
book2_pos = np.array([[813, 970], [611, 1100], [423, 798], [624, 668]])
book3_pos = np.array([[351, 740], [155, 707], [206, 428], [403, 466]])


fixed_ratio = 2
width1 = fixed_ratio * np.round(np.sqrt(np.sum((book1_pos[1] - book1_pos[0]) ** 2)))
width2 = fixed_ratio * np.round(np.sqrt(np.sum((book1_pos[3] - book1_pos[2]) ** 2)))
height1 = fixed_ratio * np.round(np.sqrt(np.sum((book1_pos[2] - book1_pos[1]) ** 2)))
height2 = fixed_ratio * np.round(np.sqrt(np.sum((book1_pos[3] - book1_pos[0]) ** 2)))

height = (height1 + height2) // 2
width = (width1 + width2) // 2

book1_img = np.zeros((int(height), int(width), 3), dtype=books_img.dtype)

M, N,_ = book1_img.shape
print(M, N)
book1_corr_pos = np.array([[0, 0], [N-1, 0], [N-1, M-1], [0, M-1]])

transformation_matrix = cv2.getPerspectiveTransform(book1_pos.astype(np.float32), 
                                                    book1_corr_pos.astype(np.float32))


print(book1_img.shape)
print(M, N)

book1_img = cv2.warpPerspective(books_img, transformation_matrix, dsize=(N, M))

my_book1_img = myWarpFunction(books_img, transformation_matrix, (M, N))

print(book1_img.shape)


showImg(my_book1_img, 0.5, 'my book 1 image')
showImg(book1_img, 0.5, 'book 1 image')
cv2.waitKey(0)
cv2.destroyAllWindows()
