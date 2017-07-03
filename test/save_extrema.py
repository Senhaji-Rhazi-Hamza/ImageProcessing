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
    imgs = []
    imgs1 = []
    imgs2 = []
    for i in range(sift.octaveLvl):
        imgs.append([])
        imgs1.append([])
        imgs2.append([])
        for j in range(sift.DoGLvl):
            imgs[i].append([])
            imgs1[i].append([])
            imgs2[i].append([])
            imgs[i][j] = np.zeros(shape = DoG[i][j].shape)
            imgs1[i][j] = np.zeros(shape = DoG[i][j].shape)
            imgs2[i][j] = np.zeros(shape = DoG[i][j].shape)
            for x, y in extrema[i][j]:
                imgs[i][j][x, y] = 255
            for x, y in extrema1[i][j]:
                imgs1[i][j][x, y] = 255
            for x, y in extrema2[i][j]:
                imgs2[i][j][x, y] = 255
    sift.save_images(imgs, name = 'extrema') 
    sift.save_images(imgs1, name = 'extrema1')
    sift.save_images(imgs2, name = 'extrema2')


if __name__ == "__main__" :
    test(sys.argv[:])


