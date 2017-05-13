import numpy as np
import cv2
import image as im
# Load an color image in grayscale

img = cv2.imread("ressources/cat.jpg",1)
name = "ressources/cat.jpg"
height, width = img.shape[:2]
size = min(height, width)
img = cv2.resize(img,(int(size / 2), int( size / 2)), interpolation = cv2.INTER_CUBIC)

imo = im.Image(name)
imo.show_dog(2,2)
