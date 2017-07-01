import numpy as np
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
        """Extract SIFT features from image.

        :param image: np.array,
        :rtype: np array
        """
        pyramid = self.build_pyramid(image)
        self.show_images(pyramid, 'pyramid')
        # octaves = self.build_octaves(pyramid)
        # DoG = self.build_DoG(octaves)
        # self.show_images(DoG)
        # extremum = self.compute_extrema(DoG)
        # self.save_images(extremum, self.octaveLvl, self.DoGLvl, "Extremum")
        # self.show_images(extremum) 
    

    def build_pyramid(self, image):
        """Builds the pyramid of an image. The pyramid has octaveLvl levels. \
                The first level is build by doubeling the size of the original \
                image, then each change (in the top direction) of level \
                divides the size by 2. size (level + 1) = size (level) / 2
                    
        :param image: np.array
        :rtype: [np.array] 
        """
        pyramid = [cv2.resize(image, None, fx = 2 ** i, 
            fy = 2 ** i, interpolation = cv2.INTER_LINEAR) 
            for i in range(1, -self.octaveLvl, -1)]
        print("pyramid length: {}".format(len(pyramid)))
        return pyramid

    def build_octaves(self, pyramid):
        """Apply Gaussian function with different scales to all levels of \
                the pyramid, this will generate the octaves. Each octave \
                consistes of different scales.
        
        :param pyramid: [np.array]
        :rtype: [[np.array]]
        """
        octaves = [[cv2.GaussianBlur(pyramid[i], ksize = (0, 0),
            sigmaX = self.sigma * self.k ** j) 
            for j in range(self.scaleLvl)] 
            for i in range(self.octaveLvl)]
        return octaves

    # build differenc of gaussians
    def build_DoG(self, octaves):
        """Build Difference of Gaussians (DoG) from octaves. There are \
                different scales for a specific octave, The DoG of level i is \
                the absolute difference between scale i and i + 1
        :param octaves: [[np.array]]
        """
        DoG = [[cv2.subtract(octaves[i][j + 1], octaves[i][j])
                for j in range(self.DoGLvl)]
                for i in range(self.octaveLvl)]
        return DoG

    def compute_extrema(self, DoG):
        """Computes extrema (minima and maxima) between the 27, 18 or 9 \
                neighbours depending on the scale level. 
        
        :param DoG: [[np.array]]
        :rtype: np.array(x, y, sigma, octave)
        """
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

    def __show_images(self, images, title, n = 0):
        """Helper method, shows a series of images
        
        :param images: [np.images]
        :rtype: Int, 1 if stopped, 0 if not"""
        if n == 0:
            n = len(images)
        for i in range (n):
            img_title = title + '[{}]'.format(i)
            cv2.imshow(img_title, images[i])
            if cv2.waitKey(0) == 113:
                cv2.destroyAllWindows()
                return 1
            cv2.destroyAllWindows()
        return 0


    def show_images(self, images, title = 'Image', n = 0, m = 0):
        """Show n * m images. If a length is not specified, it will take the \
                maximum value possible. 

        :param images: [[np.array]]
        :param n: length of first list
        :param m: length of second list
        :rtype: None
        """
        print("Showing a group of images.\nPress any key to show next image.\
                \nPress 'q' to exit.")
        if n == 0:
            n = len(images)
        if m == 0:
            self.__show_images(images, title)
            return
        for i in range (n):
            img_title = title + '[{}]'.format(i)
            if self.__show_images(images[i]) == 1:
                return 
            
    def save_images(self, images, n = 0, m = 0, name = "image"):
        """Save n * m images. If a length is not specified, it will take the \
                maximum value possible. 

        :param images: [[np.array]]
        :param n: length of first list
        :param m: length of second list
        """
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

