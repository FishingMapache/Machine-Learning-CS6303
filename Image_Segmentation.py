import numpy as np
import cv2wrap as cv2
import os

if __name__ == "__main__":
    img = cv2.imread('/Users/DianaZi/Desktop/images/image5.jpg') # Change the image directory to run
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    reshapped = res.reshape(img.shape)

    cv2.imshow('Clustered_Image', reshapped)
    cv2.imwrite(os.path.join('/Users/DianaZi/Desktop/Clustered_Images', 'Clustered_Image_5.jpg'), reshapped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
