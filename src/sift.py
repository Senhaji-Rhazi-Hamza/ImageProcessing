import numpy as np
import cv2
import math

class SIFT:
    
    def __init__(self):
        self.k = math.sqrt(2)
        # self.sigma = 1.6
        s = math.sqrt(2)
        self.sigma0 = [s/2, s, 2 * s, 4 * s]
        self.scaleLvl = 5
        self.octaveLvl = 4
        self.DoGLvl = 4

    def extractFeatures(self, image):
        height, width = image.shape[:2]
        size = min(height, width)
        self.generateOctaves(image)
        self.generateDoG()
 
    
    def generateOctaves(self, image):
        # generate pyramide (octaveLvl different sizes):
        pyramide = [cv2.resize(image, None, fx = 2 ** -(i), 
            fy = 2 ** -(i), interpolation = cv2.INTER_CUBIC) 
            for i in range(self.octaveLvl)]
        # apply gaussian filter on pyramide to generate different octave/scales 
        self.octaves = [[cv2.GaussianBlur(pyramide[j], ksize = (0, 0),
            sigmaX = self.sigma0[j] * self.k ** i, sigmaY = 0) 
            for i in range(self.scaleLvl)] 
            for j in range(self.octaveLvl)]
 
    def showOcatves(self):
        for i in range (self.octaveLvl):
            for j in range (self.scaleLvl):
                img = 'image [' + str(i) + '][' + str(j) + ']'
                cv2.imshow(img, self.octaves[i][j])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
 
    def generateDoG(self):
        self.DoG = [[self.octaves[i][j + 1] - self.octaves[i][j] 
            for j in range(self.DoGLvl)]
            for i in range(self.octaveLvl)]
        

 
    def showDoG(self):
        for i in range (self.octaveLvl):
            for j in range (self.DoGLvl):
                img = 'DoG [' + str(i) + '][' + str(j) + ']'
                cv2.imshow(img, self.DoG[i][j])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
