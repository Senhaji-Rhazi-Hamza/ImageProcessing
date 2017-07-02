import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from src.sift import SIFT

import sys
import cv2
import numpy as np
import math


def parse_arguments(argv):
    n = len(argv)
    sigma, k = 1.6, math.sqrt(2)
    if n < 2:
        print("Usage: python {} image_path [-s value] [-k value]".format(argv[0]))
        sys.exit(1)
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


def test(argv):
    sigma, k = parse_arguments(argv)
    sift = SIFT(sigma, k)
    img = cv2.imread(argv[1], 0)
    pyramid = sift.build_pyramid(img)
    octaves = sift.build_octaves(pyramid)
    DoG = sift.build_DoG(octaves)
    extrema = sift.compute_extrema(DoG)
    extrema1 = sift.remove_low_contrast(DoG, extrema)
    extrema2 = sift.remove_curvature(DoG, extrema1)
    for i in range (sift.octaveLvl):
        for j in range (sift.DoGLvl):
            print('pixels in image: {}'.format(DoG[i][j].shape[0] *
                DoG[i][j].shape[1]))
            print('number of extrema in [{}][{}] = {}'
                    .format(i, j, len(extrema[i][j]) // 2))
            print('number of extrema1 in [{}][{}] = {}'
                    .format(i, j, len(extrema1[i][j]) // 2))
            print('number of extrema2 in [{}][{}] = {}'
                    .format(i, j, len(extrema2[i][j]) // 2))


if __name__ == "__main__" :
    test(sys.argv[:])


