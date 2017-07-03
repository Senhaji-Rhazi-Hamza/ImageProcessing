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


def save_extremas(sift, DoG, extremas, n, m, title = 'extrema'):
    imgs = []
    for i in range(n):
        imgs.append([])
        for j in range(m):
            imgs[i].append([])
            imgs[i][j] = np.zeros(shape = DoG[i][j].shape)
            for x, y in extremas[i][j]:
                imgs[i][j][x, y] = 255
    sift.save_images(imgs, title = title) 



def test(argv):
    sigma, k = parse_arguments(argv)
    sift = SIFT(sigma, k)
    img = cv2.imread(argv[1], 0)
    pyramid = sift.build_pyramid(img)
    print('pyramid built')
    octaves = sift.build_octaves(pyramid)
    print('octaves built')
    DoG = sift.build_DoG(octaves)
    print('DoG built')
    extrema = sift.compute_extrema(DoG)
    print('extrema computed')
    extrema1 = sift.remove_low_contrast(DoG, extrema)
    print('extremas with low contrast removed')
    extrema2 = sift.remove_curvature(DoG, extrema1)
    print('extremas with high curvatures removed')
    save_extremas(sift, DoG, extrema, 1, 1, 'extremums/ext')
    save_extremas(sift, DoG, extrema1, 1, 1, 'extremums/ext1')
    save_extremas(sift, DoG, extrema2, 1, 1, 'extremums/ext2')

if __name__ == "__main__" :
    test(sys.argv[:])


