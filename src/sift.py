import numpy as np
import src.differentiate as nd #uncomment for testing
# import differentiate as nd  #comment for testing
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
        octaves = self.build_octaves(pyramid)
        DoG = self.build_DoG(octaves)
        extrema = self.compute_extrema(DoG)
        extrema1 = self.remove_low_contrast(DoG, extrema)
        extrema2 = self.remove_curvature(DoG, extrema1)
        # self.save_images(extremum, self.octaveLvl, self.DoGLvl, "Extremum")
        self.show_images(DoG) 
    

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
            sigmaX = self.sigma * self.k ** j) 
            for j in range(self.scaleLvl)] 
            for i in range(self.octaveLvl)]
        return octaves

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
                neighbours depending on the scale level for all octaves. \
                for each (octave, scale) it computes a list of positions.
        
        :param DoG: [[np.array]]
        :rtype: [[[(Int, Int)]]], ex: [octave][scale] = [(x, y)]
        """
        extrema = []
        for i in range (self.octaveLvl):
            extrema.append([])
            for j in range (self.DoGLvl):
                extrema[i].append([])
                img = DoG[i][j]
                img_top = DoG[i][j + 1] if j < self.DoGLvl - 1 else None
                img_bot = DoG[i][j - 1] if j > 0 else None
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
                            extrema[i][j].append((k, l))
        return extrema


    def remove_low_contrast(self, DoG, extrema):
        """Removes low contrast in extrema points.
        
        :param DoG: [[np.array]]
        :param extrema: [[[(Int, Int)]]], ex: [octave][scale] = [(x, y)] 
        :rtype: [[[(Int, Int)]]], ex: [octave][scale] = [(x, y)]
        """
        ext = []
        for i in range(self.octaveLvl):
            ext.append([])
            D = np.array(DoG[i])
            grad = np.array(np.gradient(D)) / 255
            hess = nd.hessian(D) / 255
            for j in range(self.DoGLvl):
                ext[i].append([])
                for x, y in extrema[i][j]:
                    g = grad[:, j, x, y]
                    h = hess[:, :, j, x, y]
                    e = np.linalg.lstsq(h, g)[0]
                    e_j, e_x, e_y = j, x, y
                    if (np.abs(e) > 0.5).any():
                        continue
                    d = D[j, x, y] + 0.5 * g.T.dot(e)
                    if np.abs(d) > 0.03:
                        ext[i][j].append((e_x, e_y))
        return ext
 
    def remove_curvature(self, DoG, extrema):
        """Removes low contrast in extrema points.
        
        :param DoG: [[np.array]]
        :param extrema: [[[(Int, Int)]]], ex: [octave][scale] = [(x, y)] 
        :rtype: [[[(Int, Int)]]], ex: [octave][scale] = [(x, y)]
        """
        ext = []
        tresh = (11 * 11) / 10
        for i in range(self.octaveLvl):
            ext.append([])
            D = np.array(DoG[i])
            hess = nd.hessian(D)
            for j in range(self.DoGLvl):
                ext[i].append([])
                for x, y in extrema[i][j]:
                    h = hess[:, :, j, x, y]
                    det = np.linalg.det(h) 
                    if det == 0:
                    #    ext[i][j].append((x, y))
                        continue
                    if (np.trace(h) ** 2) / det <= tresh:
                        ext[i][j].append((x, y))
        return ext
 
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
