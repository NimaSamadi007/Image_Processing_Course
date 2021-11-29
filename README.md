# Image_Processing_Course
Image Processing Course Assignments


## HW1
### Q1 and Q2
These questions are about enhancing two dark photo's qualities. The origianl images are quite dark and are not pleasent for human's eyes. So, I enchanced them using gamma transformation, contrast stretching thecniques. These are origianl images:

| ![](./HW1/Enhance1.JPG "original img 1")  | ![](./HW1/Enhance2.jpg "original img 2") |
| ------------- | ------------- |

And the enhanced images are:

| ![](./HW1/res01.jpg "enhanced img 1")  | ![](./HW1/res02.jpg "enhanced img 2") |
| ------------- | ------------- |

### Q3
In Q3 I wrote a code to convert [Prokudin-Gorskii](https://www.loc.gov/pictures/collection/prok/ "Prokudin-Gorskii images") black and white images to colory jpg images. The Prokudin-Gorskii' images are in .tif format with separated blue, red, and green channels. q3.py can convert them properly. I've choose Amir, Mosque and Train images to test program. These program converts a 16 bit image to 8 bit jpg colory image. I'v used gaussian pyramid to improve speed while preserving the accuracy. After finding best matches for each channel, we should clip each side with a proper value (since the sides of the original images are in a bad shape). This procedure is done automatically. 

The results are as below:
| ![](./HW1/res03-Amir.jpg "amir image")  | ![](./HW1/res03-Mosque.jpg "mosque image") | ![](./HW1/res03-Train.jpg "train image") |
| ------------- | ------------- | ------------- |


### Q4
This program changes the flowers color to pink and blures the background:

| ![](./HW1/Flowers.jpg "flowers img")  | ![](./HW1/res06.jpg "color changed") |
| ------------- | ------------- |

### Q5
Filtering an image by using OpenCV filter2D function, using naive double-for-loop implementation, and matrix addition method (which is faster). The time for each method is written under each image:

| Opencv filter2D method | Double-for Method | Matrix Addition Method |
| ------------- | ------------- | ------------- |
| ![](./HW1/res07.jpg "opencv")  | ![](./HW1/res08.jpg "naive approach") | ![](./HW1/res09.jpg "Matrix addition") |
| ‫‪0.01636419900000008‬‬s | ‫‪137.183767327‬‬s | ‫‪0.2271269970000276‬‬s |

### Q6
Histogram specification code to enhance an image's quality

| ![](./HW1/Dark.jpg "original dark image")  | ![](./HW1/Pink.jpg "specific image") | ![](./HW1/res11.jpg "histogram specified image") | 
| ------------- | ------------- | ------------- |

## HW2

### Q1, Image sharpening techniques:
Image sharpaening using spatial and frequency domain tools. Original image is blured and we wish to sharp it using unsharp mask. This is the original, non sharp image:

![blured image](./HW2/flowers.blur.png "original image")

The sharped images are as below:
| ![](./HW2/res04.jpg "sharped img 1")  | ![](./HW2/res07.jpg "sharped img 2") |
| ------------- | ------------- |
| ![](./HW2/res11.jpg "sharped img 3")  | ![](./HW2/res14.jpg "sharped img 4")  |

### Q2, Simple template matching problem:
In this problem I used zero mean cross correlation method to match a given template with an image. The patch is a pipe which we want to find it in image. The result is as below:
![](./HW2/res15.jpg "found pipes")

### Q3:
In this problem we want to extract three books form an image. The books are rotated and there is a little perspective in the picture which makes it a bit hard to derive best results. I choosed every four corner of each book and fitted a homography transformation using opencv. Finally the image is warped using myWarpFunction - which I'v implemented it. The origianl image is:
![](./HW2/books.jpg "books image")
And three extracted books are:
| ![](./HW2/res16.jpg "book1")  | ![](./HW2/res17.jpg "book2") | ![](./HW2/res18.jpg "book3") |
| ------------- | ------------- | --------------------- |

### Q4, Hybrid images:
Hybrid images are kind of delusional. From near, you can see an image. As you go back and get away from the image, it seems you are observing another image. It happens as we interpret details when we are close enough to the image while from distance we can only see the overall shape. The details are high frequent component in image and the overall shape is composed by low frequency components. So, I used it to generate hybrid images. You can find the article from [Here](https://stanford.edu/class/ee367/reading/OlivaTorralb_Hybrid_Siggraph06.pdf "Hybrid images article"). I choosed these images:

| ![](./HW2/res19-near.jpg "near img")  | ![](./HW2/res20-far.jpg "far img") |
| ------------- | ------------- |

The motorcycle image will be seen from far while the bycicle image will be seen from near. The resulting hybrid image is:
| ![](./HW2/res30-hybrid-near.jpg "near interpretation")  | ![](./HW2/res31-hybrid-far.jpg "far interpretation") |
| ------------- | ------------- |

The hybrid image is smalled so you can see what it will look like when you see it from a distance.
