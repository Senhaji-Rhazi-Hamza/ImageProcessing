import sys
import cv2
import numpy as np
import math
from sift import SIFT


def parse_arguments(argv):
    n = len(argv)
    sigma, k = 1.6, math.sqrt(2)
    if n < 2:
        print("Usage: python {} image_path \
                [-s value] [-k value]".format(argv[0]))
        exit(1)
    if n > 6:
        if argv[2] == '-s' and argv[4] == '-k':
            sigma = float(argv[3])
            k = float(argv[5])
        if argv[2] == '-k' and argv[4] == 's':
            k = float(argv[3])
            sigma = float(argv[5])
    if n > 4:
        if argv[2] == '-s':
            sigma = float(argv[3])
        if argv[2] == '-k':
            k = float(argv[3])
   return sigma, k 

def main(argv):
    sigma, k = parse_arguments(argv)
    sift = SIFT(sigma, k)
    img = cv2.imread(argv[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift.extract_features(img)

if __name__ == "__main__" :
    main(sys.argv[:])

