import sys
import cv2
import numpy as np
import math
from sift import SIFT


def main(argv):
    if len(argv) < 2:
        print "Usage Error: python" ,argv[0] + "image_name [sigma]" 
        return
    
    sigma = 1.6
    k = math.sqrt(2)
    if len(argv) > 2:
        sigma = float(argv[2])
    if len(argv) > 3:
        k = float(argv[3])
    img = cv2.imread(argv[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = SIFT(sigma, k)
    sift.extract_features(img)

if __name__ == "__main__" :
    main(sys.argv[:])

