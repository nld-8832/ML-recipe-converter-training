from cv2 import cv2
import numpy as np
import os

img_dir = 'C:\\Users\\linhn\\Desktop\\ML_final\\app\\testdata'
imgFiles = os.listdir(img_dir)
for (i, f) in enumerate(imgFiles):
    
    # Read pictures
    print('Segmentating %s'%f)
    img = cv2.imread('D:\\UET\\prepare_Dataset\\testDataset\\output_test_01\\%s'%f)

    img_output_path = 'D:\\UET\\prepare_Dataset\\testDataset\\output_test_02\\%s'%f
    if not os.path.exists(img_output_path):
        os.mkdir(img_output_path)

    # Converting to Gray 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Using Sobel Edge Detection to Generate Binary Graph
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    # Two valued
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # Expansion and Corrosion
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

    # Expansion once to make the outline stand out
    dilation = cv2.dilate(binary, element2, iterations=1)

    # Corrode once, remove details
    erosion = cv2.erode(dilation, element1, iterations=1)

    # Expansion again to make the outline more visible
    dilation2 = cv2.dilate(erosion, element2, iterations=2)

    # Finding Outlines and Screening Text Areas
    region = []
    contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        cnt = contours[i]

        # Calculate contour area and screen out small areas
        area = cv2.contourArea(cnt)
        if (area < 1000):
            continue

        # Find the smallest rectangle
        rect = cv2.minAreaRect(cnt)
        #print("Rect: ", rect)

        # Box is the coordinate of four points
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Computing height and width
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # According to the characteristics of the text, select those too thin rectangles, leaving flat ones.
        if (height > width * 1.3):
            continue

        region.append(box)

    
    # Segmentate into images
    i = 0
    for box in region:
        x, y, w, h = cv2.boundingRect(box)
        ROI = img[y:y+h, x:x+w]

        path = os.path.join(img_output_path, '{}.jpg'.format(i))
        cv2.imwrite(path, ROI)
        i += 1

    cv2.drawContours(img, region, -1, (0, 255, 0), 3)
    cv2.imwrite('D:\\UET\\prepare_Dataset\\testDataset\\output_test_01\\%s\\result.jpg'%f, img)

'''
    cv2.imshow('Line detected', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''