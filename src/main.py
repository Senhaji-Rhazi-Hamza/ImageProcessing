import sys
import cv2
import numpy as np
from sift import SIFT


def main(argv):
    if len(argv) < 3:
        print "Usage Error: python" ,argv[0] + "image_name sigma" 
        return
    img = cv2.imread(argv[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = SIFT(float(argv[2]))
    sift.extractFeatures(img)

if __name__ == "__main__" :
    main(sys.argv[:])

