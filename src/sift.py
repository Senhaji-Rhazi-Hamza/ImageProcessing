import numpy as np
np.set_printoptions(threshold=np.nan)
import cv2
import math


class SIFT:
    
    def __init__(self, sigma):
        self.k = math.sqrt(2)
        self.sigma = sigma
        s = math.sqrt(2)
        self.sigma0 = [s/2, s, 2 * s, 4 * s]
        self.scaleLvl = 5
        self.octaveLvl = 4
        self.DoGLvl = self.scaleLvl - 1

    def extractFeatures(self, image):
        height, width = image.shape[:2]
        size = min(height, width)
        octaves = self.generateOctaves(image)
#        self.showOcatves(octaves)
        DoG = self.generateDoG(octaves)
        self.showDoG(DoG)
        extremum = self.getExtremum(DoG)
        self.showExtremum(extremum)
    
    def generateOctaves(self, image):
        # generate pyramide (octaveLvl different sizes):
        pyramide = [cv2.resize(image, None, fx = 2 ** -(i), 
            fy = 2 ** -(i), interpolation = cv2.INTER_CUBIC) 
            for i in range(self.octaveLvl)]
        # apply gaussian filter on pyramide to generate different octave/scales 
        octaves = [[cv2.GaussianBlur(pyramide[i], ksize = (0, 0),
            sigmaX = self.sigma * self.k ** j, sigmaY = 0) 
            for j in range(self.scaleLvl)] 
            for i in range(self.octaveLvl)]
        return octaves

    def generateDoG(self, octaves):
        DoG = [[octaves[i][j + 1] - octaves[i][j] 
            for j in range(self.DoGLvl)]
            for i in range(self.octaveLvl)]
        return DoG

    def getExtremum(self, DoG):
        extremum = [DoG[i][0].copy() for i in range(self.octaveLvl)]
        for i in range (self.octaveLvl):
            for j in range (1, self.DoGLvl - 1):
                img = DoG[i][j]
                imgh = DoG[i][j + 1]
                imgb = DoG[i][j - 1]
                for k in range (1, img.shape[0] - 1):
                    for l in range (1, img.shape[1] - 1):
                        m = np.concatenate((imgh[k - 1 : k + 2, l - 1 : l + 2],
                                imgb[k - 1 : k + 2, l - 1 : l + 2]))
                        m = np.concatenate((m, img[k - 1: k + 2, l - 1 : l + 2]))
                        if img[k, l] == m.min() or img[k, l] == m.max():
                            extremum[i][k, l] = 255
        return extremum

    def showOcatves(self, octaves):
        for i in range (self.octaveLvl):
            for j in range (self.scaleLvl):
                img = 'image [' + str(i) + '][' + str(j) + ']'
                cv2.imshow(img, octaves[i][j])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
 
    def showDoG(self, DoG):
        for i in range (self.octaveLvl):
            for j in range (self.DoGLvl):
                img = 'DoG [' + str(i) + '][' + str(j) + ']'
                cv2.imshow(img, DoG[i][j])
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def showExtremum(self, extremum):
        for i in range (self.octaveLvl):
            img = 'Extremum [' + str(i) + ']'
            cv2.imshow(img, extremum[i])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

