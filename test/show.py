import cv2
import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from src.sift import SIFT
name = "../ressources/cat.jpg"

sift = SIFT()
img = cv2.imread(name, 1)
pyramids = sift.build_pyramid(img)
#sift.show_images(pyramids,5,0)
octaves = sift.build_octaves(pyramids)
sift.show_images(octaves,4,0)
DoGs = sift.build_DoGs(octaves)
sift.show_images(DoGs,3,0)
