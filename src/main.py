import sys
import cv2
import numpy as np
import math
import tester 
from sift import SIFT


def parse_arguments(argv):
    n = len(argv)
    sigma, k = 1.6, math.sqrt(2)
    if n > 5:
        if argv[1] == '-s' and argv[3] == '-k':
            sigma = float(argv[2])
            k = float(argv[4])
        if argv[1] == '-k' and argv[3] == 's':
            k = float(argv[2])
            sigma = float(argv[4])
    if n > 3:
        if argv[1] == '-s':
            sigma = float(argv[2])
        if argv[1] == '-k':
            k = float(argv[2])
    return sigma, k 

def main(argv):
    sigma, k = parse_arguments(argv)
    sift = SIFT(sigma, k)
    print('Welcome to the testing of SIFT method!')
    q = 'r'
    while q == 'r':
        q = input('Would you like to run sift in silent mode? \'y or n\': ')
        while q != 'y' and q != 'n':
            q = input('Answer by y or n please: ')
        silent = (q == 'y')
        path_train = input('enter path to train image: ')
        img = cv2.imread(path_train, 0)
        print('might take some time... Don\' Worry!')
        kp_train, desc_train = sift.extract_features(img, silent)
        q = 't'
        while q == 't':
            path_test = input('enter path to test image: ')
            img = cv2.imread(path_test, 0)
            print('still computing...')
            kp_test, desc_test = sift.extract_features(img, silent)
            tester.match(kp_train, desc_train, kp_test, desc_test)
            q = input("enter:\n't' to test another image\n'r' to restart\nany other key to quit: ")

if __name__ == "__main__" :
    main(sys.argv[:])

