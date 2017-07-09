import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from src.sift import SIFT

from time import time
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

def save_extrema(sift, DoGs, extremas, n, title = 'extrema'):
    imgs = []
    for i in range(n):
        imgs.append([])
        s = (sift.scaleLvl, DoGs[i][0].shape[0], DoGs[i][0].shape[1])
        imgs[i] = np.zeros(shape = s)
        for s, y, x in extremas[i]:
            imgs[i][s, y, x] = 255
    sift.save_images(imgs, title = title) 


def test(argv):
    sigma, k = parse_arguments(argv)
    sift = SIFT(sigma, k)
    img = cv2.imread(argv[1], 0)
    t = time()
    pyramid = sift.build_pyramid(img)
    print('pyramid built: {:.2f}'.format(time() - t))
    t = time()
    octaves = sift.build_octaves(pyramid)
    print('octaves built: {:.2f}'.format(time() - t))
    t = time()
    DoGs = sift.build_DoGs(octaves)
    print('DoGs built: {:.2f}'.format(time() - t))
    t = time()
    sift.precompute_params(DoGs)
    print('precompute: {:.2f}'.format(time() - t))
    t = time()
    extrema = sift.compute_extrema(DoGs)
    print('extrema computed: {:.2f}'.format(time() - t))
    t = time()
    extrema1 = sift.remove_low_contrast(DoGs, extrema)
    print('extremas with low contrast removed: {:.2f}'.format(time() - t))
    t = time()
    extrema2 = sift.remove_curvature(DoGs, extrema1)
    print('extremas with high curvatures removed: {:.2f}'.format(time() - t)) 
    
    save_extrema(sift, DoGs, extrema, 1, 'extremums/ext')
    save_extrema(sift, DoGs, extrema1, 1, 'extremums/ext1')
    save_extrema(sift, DoGs, extrema2, 1, 'extremums/ext2')

if __name__ == "__main__" :
    test(sys.argv[:])


