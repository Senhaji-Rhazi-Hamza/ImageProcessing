import sys
import cv2
import numpy as np
from sift import SIFT


def main(argv):
    if len(argv) == 1:
        print "Usage Error: python" ,argv[0] + "image_name"
        return
    img = cv2.imread(argv[1])
    sift = SIFT()
    sift.extractFeatures(img)
    sift.showOcatves()    
    sift.showDoG()

if __name__ == "__main__" :
    main(sys.argv[:])

