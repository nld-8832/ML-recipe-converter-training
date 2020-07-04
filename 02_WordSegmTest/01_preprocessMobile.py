from skimage.filters import threshold_local
from cv2 import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

imgFiles = os.listdir('D:\\UET\\prepare_Dataset\\testDataset\\input_test_3')

for (i, f) in enumerate(imgFiles):
    img = cv2.imread('D:\\UET\\prepare_Dataset\\testDataset\\input_test_3\\%s'%f)
    cv2.imshow('meh', img)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 100, 200)

    cv2.imshow('Line detected', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()