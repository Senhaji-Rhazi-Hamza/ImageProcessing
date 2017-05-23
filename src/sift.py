import numpy as np
np.set_printoptions(threshold=np.nan)
import cv2
import math


class SIFT:
    
    def __init__(self, sigma, k):
        self.k = k
        self.sigma = sigma
        # s = math.sqrt(2)
        # self.sigma0 = [s/2, s, 2 * s, 4 * s]
        self.scaleLvl = 5
        self.octaveLvl = 4
        self.DoGLvl = self.scaleLvl - 1

    def extract_features(self, image):
        pyramide = self.build_pyramid(image)
        octaves = self.build_octaves(pyramide)
        self.show_images(octaves, 1, self.scaleLvl)
        DoG = self.build_DoG(octaves)
        self.show_images(DoG, self.octaveLvl, self.DoGLvl)
#        extremum = self.compute_extrema(DoG)
#        self.show_images(extremum, self.octaveLvl, 1)
    

    # generate pyramide (octaveLvl different sizes):
    def build_pyramid(self, image):
        pyramide = [cv2.resize(image, None, fx = 2 ** -(i), 
            fy = 2 ** -(i), interpolation = cv2.INTER_CUBIC) 
            for i in range(self.octaveLvl)]
        return pyramide

    # apply gaussian filter on pyramide to generate different octave/scales 
    def build_octaves(self, pyramide):
        octaves = [[cv2.GaussianBlur(pyramide[i], ksize = (0, 0),
            sigmaX = self.sigma * self.k ** j) 
            for j in range(self.scaleLvl)] 
            for i in range(self.octaveLvl)]
        return octaves

    def build_DoG(self, octaves):
        DoG = [[octaves[i][j + 1] - octaves[i][j] 
            for j in range(self.DoGLvl)]
            for i in range(self.octaveLvl)]
        return DoG

    def compute_extrema(self, DoG):
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

    def show_images(self, images, n, m):
        print("Showing a group of images. \
                \nPress any key to show next image.\
                \nPress 'q' to exit.")
        for i in range (n):
            for j in range (m):
                if m == 1:
                    im = images[i]
                else:
                    im = images[i][j]
                img = 'image [' + str(i) + '][' + str(j) + ']'
                cv2.imshow(img, im)
                if cv2.waitKey(0) == 113:
                    cv2.destroyAllWindows()
                    return
                cv2.destroyAllWindows()
 
