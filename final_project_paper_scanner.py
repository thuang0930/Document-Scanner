import cv2
import numpy as np

#get image contours
def getContours(img):
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    max_contour = []
    max_area = 0
    for cnt in contours:
        #get contour area
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_contour = cnt
    perimeter = cv2.arcLength(max_contour,True)
    #print(perimeter)
    print(max_area)
    return max_contour, perimeter

#get contour corner points
def getCornerPoints(contour, peri):
    approx = cv2.approxPolyDP(contour,0.02 * peri,True)
    return approx

path = r'/Users/erichuang/Desktop/Python3/Digital Image Processing/IMG_0097.JPEG'
img = cv2.imread(path)
img_Resized = cv2.resize(img,(768,1024))
imgContour = img_Resized.copy()

#convert the image to a gray scale image
imgGray = cv2.cvtColor(img_Resized,cv2.COLOR_BGR2GRAY)
#imgHist = cv2.equalizeHist(imgGray)
#apply the Gaussian Blur
imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
#apply Canny filter to get the edges
imgCanny = cv2.Canny(imgBlur,50,50)

# kernel_ones = np.ones((3,3))
# edge_dilate = cv2.dilate(imgCanny, kernel_ones, iterations=1)
# edge_erode = cv2.erode(edge_dilate,kernel_ones, iterations=1)

contour, perimeter = getContours(imgCanny)
corner_points = getCornerPoints(contour, perimeter)
print(corner_points)

#new image size
width,height = img_Resized.shape[1], img_Resized.shape[0]

#create an image with only the paper (stretched out image of the paper)
original_points = np.float32([corner_points[1],corner_points[0],corner_points[2],corner_points[3]])
mapped_new_corners = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(original_points,mapped_new_corners)
stretched_paper = cv2.warpPerspective(imgGray,matrix,(width,height))

#Sharpening the image
kernel = np.array([[-1,-1,-1],
                [-1,9,-1],
                [-1,-1,-1]])
kernel2 = np.array([[1/9,1/9,1/9],
                [1/9,1/9,1/9],
                [1/9,1/9,1/9]])
stretched_paper_sharpen = cv2.filter2D(stretched_paper, -1, kernel)

#Image Dilation
stretched_paper_erode = cv2.erode(stretched_paper_sharpen,(3,3),iterations=1)

#Image Erosion
#stretched_paper_dilate = cv2.dilate(stretched_paper_erode,(3,3),iterations=1)

#stretched_paper_final = cv2.medianBlur(stretched_paper_dilate,1)

#Image Thresholding
_, stretched_paper_threshold = cv2.threshold(stretched_paper_erode,127,255,cv2.THRESH_BINARY)

cv2.imshow("Image",img_Resized)
# cv2.imshow("Gray Image",imgGray)
# cv2.imshow("Blurred Image",imgBlur)
cv2.imshow("Edge Dectection",imgCanny)
# cv2.imshow("Scanned Image",stretched_paper)
# cv2.imshow("Edge", edge_dilate)
cv2.imshow("Sharpened Image",stretched_paper_erode)
#cv2.imshow("Dilated Sharpened Image",stretched_paper_dilate)
cv2.imshow("Threshold Image",stretched_paper_threshold)
cv2.waitKey(0)