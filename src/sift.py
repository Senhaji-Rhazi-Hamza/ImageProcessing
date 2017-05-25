import numpy as np
np.set_printoptions(threshold=np.nan)
import cv2
import math


class SIFT:
    
    def __init__(self, sigma = 1.6, k = math.sqrt(2)):
        self.k = k
        self.sigma = sigma
        self.scaleLvl = 5
        self.octaveLvl = 4
        self.DoGLvl = self.scaleLvl - 1

    def extract_features(self, image):
        pyramide = self.build_pyramid(image)
        octaves = self.build_octaves(pyramide)
        DoG = self.build_DoG(octaves)
        # self.show_images(DoG)
        # extremum = self.compute_extrema(DoG)
        # self.save_images(extremum, self.octaveLvl, self.DoGLvl, "Extremum")
        # self.show_images(extremum) 
    

    # build pyramide (octaveLvl different sizes):
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

    # build differenc of gaussians
    def build_DoG(self, octaves):
        DoG = [[cv2.subtract(octaves[i][j + 1], octaves[i][j])
                for j in range(self.DoGLvl)]
                for i in range(self.octaveLvl)]
        return DoG

    def compute_extrema(self, DoG):
        extremum = [[np.zeros(shape=DoG[i][j].shape)
                for j in range(self.DoGLvl)]
                for i in range(self.octaveLvl)]
        self.show_images(extremum, 1, 1)
        for i in range (self.octaveLvl):
            for j in range (self.DoGLvl):
                img = DoG[i][j]
                imgh = None
                imgb = None
                if j < self.DoGLvl - 1:
                    imgh = DoG[i][j + 1]
                if j > 0:
                    imgb = DoG[i][j - 1]
                for k in range (1, img.shape[0] - 1):
                    for l in range (1, img.shape[1] - 1):
                        m = img[k - 1: k + 2, l - 1 : l + 2]
                        if imgh is not None:
                            m = np.concatenate((m, imgh[k - 1 : k + 2, l - 1 : l + 2]))
                        if imgb is not None:
                            m = np.concatenate((m, imgb[k - 1 : k + 2, l - 1 : l + 2]))
                        if img[k, l] == m.min() or img[k, l] == m.max():
                            extremum[i][j][k, l] = 255
        return extremum

    def show_images(self, images, n = 0, m = 0):
        print("Showing a group of images. \
                \nPress any key to show next image.\
                \nPress 'q' to exit.")
        if n == 0:
            n = len(images)
        if m == 0:
            m = len(images[0])
        for i in range (n):
            for j in range (m):
                img = 'image [' + str(i) + '][' + str(j) + ']'
                cv2.imshow(img, images[i][j])
                if cv2.waitKey(0) == 113:
                    cv2.destroyAllWindows()
                    return
                cv2.destroyAllWindows()
 
    def save_images(self, images, n = 0, m = 0, name = "image"):
        if n == 0:
            n = len(images)
        if m == 0:
            m = len(images[0])
        for i in range (n):
            for j in range (m):
                img = "ressources/" + name + '[' + \
                        str(i) + '][' + str(j) + '].jpg'
                cv2.imwrite(img, images[i][j])
        print(n * m, "images saved successfully")

