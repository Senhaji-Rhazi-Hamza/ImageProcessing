import numpy as np
import cv2
import math
from time import time
from numpy.lib.stride_tricks import as_strided as ast

import src.helpers as hp #uncomment for testing
# import helpers as hp  #comment for testing

class SIFT:
    
    def __init__(self, sigma = math.sqrt(2) / 2, k = math.sqrt(2)):
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
        octaves = self.build_octaves(pyramid)
        DoGs = self.build_DoGs(octaves)
        self.precompute_params(DoGs)
        extrema = self.compute_extrema(DoGs)
        extrema1 = self.remove_low_contrast(DoGs, extrema)
        extrema2 = self.remove_curvature(DoGs, extrema1)
        # self.save_images(extremum, self.octaveLvl, self.DoGLvl, "Extremum")
        self.show_images(DoGs) 
    

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
        return pyramid

    def build_octaves(self, pyramid):
        """Apply Gaussian function with different scales to all levels of \
                the pyramid, this will generate the octaves. Each octave \
                consistes of different scales.
        
        :param pyramid: [np.array]
        :rtype: [[np.array]]
        """
        octaves = [[cv2.GaussianBlur(pyramid[i], ksize = (0, 0),
            sigmaX = self.sigma * self.k ** (2 * i + j)) 
            for j in range(self.scaleLvl)] 
            for i in range(self.octaveLvl)]
        return octaves

    def build_DoGs(self, octaves):
        """Build Difference of Gaussians (DoGs) from octaves. There are \
                different scales for a specific octave, The DoGs of level i is \
                the absolute difference between scale i and i + 1
        :param octaves: [[np.array]]
        """
        DoGs = [[cv2.subtract(octaves[i][j + 1], octaves[i][j])
                for j in range(self.DoGLvl)]
                for i in range(self.octaveLvl)]
        return DoGs
    
    def compute_extrema(self, DoGs):
        """Computes extrema (minima and maxima) between the 27, 18 or 9 \
                neighbours depending on the scale level for all octaves. \
                for each (octave, scale) it computes a list of positions.
        
        :param DoGs: [[np.array]]
        :rtype: [[[(Int, Int)]]], ex: [octave][scale] = [(x, y)]
        """
        extrema = []
        for i in range (self.octaveLvl):
            extrema.append([])
            for j in range (self.DoGLvl):
                img = DoGs[i][j]
                img_top = DoGs[i][j + 1] if j < self.DoGLvl - 1 else None
                img_bot = DoGs[i][j - 1] if j > 0 else None
                for k in range (1, img.shape[0] - 1):
                    for l in range (1, img.shape[1] - 1):
                        m = img[k - 1: k + 2, l - 1 : l + 2]
                        if img_top is not None:
                            m = np.concatenate((m, 
                                img_top[k - 1 : k + 2, l - 1 : l + 2]))
                        if img_bot is not None:
                            m = np.concatenate((m,
                                img_bot[k - 1 : k + 2, l - 1 : l + 2]))
                        if img[k, l] == m.min() or img[k, l] == m.max():
                            extrema[i].append((j, k, l))
            extrema[i] = np.array(extrema[i])
        return extrema
 
    def remove_low_contrast_opt(self, DoGs, extrema):
        """Removes low contrast in extrema points.
        
        :param DoGs: [[np.array]]
        :param extrema: [[[(Int, Int)]]], ex: [octave][scale] = [(x, y)] 
        :rtype: [[[(Int, Int)]]], ex: [octave][scale] = [(x, y)]
        """
        ext = []
        for i in range(self.octaveLvl):
            ext.append([])
            D = np.array(DoGs[i])
            gradient = np.array(np.gradient(D)) / 255
            hessian = hp.hessian(D) / 255
            gradient = np.transpose(gradient, (1, 2, 3, 0))
            hessian = np.transpose(hessian, (2, 3, 4, 0, 1))
            det = np.linalg.det(hessian)
            j, y, x = np.array(np.where(det != 0))
            grad = gradient[j, y, x]
            hess = hessian[j, y, x]
            D = D[j, y, x]
            e = np.linalg.solve(hess, grad)
            i1 = np.where(np.all(np.abs(e) < 0.5, axis = 1))
            print(i1)
            d = np.empty(e.shape[0])
            for k in range(e.shape[0]):
                d[k] = 0.5 * grad[k].dot(e[k])
            d += D 
            i2 = np.where(d > 0.03)[0]
            i3 = np.intersect1d(i1, i2)
            ext[i] = extrema[i][i3]
        return ext
 
    def remove_low_contrast(self, DoGs, extrema):
        """Removes low contrast in extrema points.


        :param DoGs: [[np.array]]
        :param extrema: [[[(Int, Int)]]], ex: [octave][scale] = [(x, y)] 
        :rtype: [[[(Int, Int)]]], ex: [octave][scale] = [(x, y)]
        """
        ext = []
        for i in range(self.octaveLvl):
            ext.append([])
            D = np.array(DoGs[i])
            grad = np.array(np.gradient(D))
            hess = hp.hessian(D)
            for j, x, y in extrema[i]:
                g = grad[:, j, x, y]
                h = hess[:, :, j, x, y]
                e = np.linalg.lstsq(h, g)[0]
                if (np.abs(e) > 0.5).any():
                    continue
                d = D[j, x, y] + 0.5 * g.T.dot(e)
                if np.abs(d) > 0.03:
                    ext[i].append((j, x, y))
            ext[i] = np.array(ext[i])
        return ext

    def remove_curvature(self, DoGs, extrema = None):
        """Removes low contrast in extrema points.
        
        :param DoGs: [[np.array]]
        :param extrema: [[[(Int, Int)]]], ex: [octave][scale] = [(x, y)] 
        :rtype: [[[(Int, Int)]]], ex: [octave][scale] = [(x, y)]
        """
        ext = []
        tresh = (11 * 11) / 10
        for i in range(self.octaveLvl):
            ext.append([])
            D = np.array(DoGs[i])
            hess = hp.hessian(D)
            det = np.linalg.det(hess.T).T
            tr = np.trace(hess)
            det[det == 0] = 1e-7
            idx = np.array(np.where(tr ** 2 / det < tresh))
            ext[i] = np.array(idx.T)
            if extrema is not None:
                ext[i] = hp.intersect(ext[i], extrema[i])
        return ext

    def __gaussian_windows(self, desc = False):
        windows = np.empty((self.octaveLvl, self.DoGLvl, 16, 16))
        for i in range(self.octaveLvl):
            for j in range(self.DoGLvl):
                sigma = 8 if desc else self.sigma * self.k ** (2 * i + j) * 1.5
                for x in range(16):
                    for y in range(16):
                        windows[i, j, x, y] = np.exp(((x - 8) ** 2 \
                                + (y - 8) ** 2) / -2 * sigma ** 2) / \
                                2 * math.pi * sigma ** 2
        return windows


    def precompute_params(self, DoGs):
        """Precomputes all useful parameters
        
        :param DoGs: [[np.array]]
        :rtype: None
        """
        self.gradients = [None] * self.octaveLvl
        self.hessians = [None] * self.octaveLvl
        self.hessians_det = [None] * self.octaveLvl
        self.magnitudes = [None] * self.octaveLvl
        self.orientations = [None] * self.octaveLvl
        self.gaussian_widows = self.__gaussian_windows()
        self.gaussian_widows_desc = self.__gaussian_windows(True)
        for i in range(self.octaveLvl):
            D = np.array(DoGs[i]) / 255
            grad = np.array(np.gradient(D))
            self.gradients[i] = grad
            self.hessians[i] = hp.hessian(D)
            self.hessians_det[i] = np.linalg.det(self.hessians[i].T).T
            self.magnitudes[i] = np.linalg.norm(grad[1:], axis = 0)
            theta = np.arctan2(grad[1], grad[2]) * 180 / math.pi
            j, y, x = np.where(theta < 0)
            theta[j, y, x] = 360 - theta[j, y, x]
            self.orientations[i] = theta
    
    
    def __get_histogram(self, extremum, size):
        j, y, x = extremum
        n, m = self.magnitudes[0].shape
        y1 = max(y - 8, 0)
        y2 = min(y + 8, n)
        x1 = max(x - 8, 0)
        x2 = min(x + 8, m)
        window = self.magnitudes[j][y1 : y2, x1 : x2]
        # apply gaussian on window
        return window

    def __get_descroptors(self, DoG, extrema):
        descriptors = np.empty(shape = (extrema.shape + (128,)))
        for i in range(self.octaveLvl):
            D = np.array(DoG[i])
            grad = np.gradient(D)
            magn = np.linalg.norm(grad, axis = 0)
            theta = grad[0] / grad[1]
            for extremum in extrema[i]:
                continue
                #descriptor 
        return descriptors


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


    def show_images(self, images, n = 0, m = 0, title = 'Image'):
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
        elif n == -1:
            self.__show_images(images, title)
            return
        for i in range (n):
            img_title = title + '[{}]'.format(i)
            if self.__show_images(images[i], img_title) == 1:
                return 
            
    def save_images(self, images, n = 0, m = 0, title = "Image"):
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
                img_path = 'ressources/{}[{}][{}].jpg'.format(title, i, j)
                print(img_path)
                cv2.imwrite(img_path, images[i][j])
        print(n * m, "images saved successfully")

