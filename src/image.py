import numpy as np
import cv2

class Image:
  #self.s = None 
  def __init__(self, name):
      self.img = cv2.imread(name,1)
      height, width = self.img.shape[:2]
      self.kparam = 2
      self.scales = None
      #self.octaves = None
      self.dogs = None
      self.size = min(height, width)
      self.img = cv2.resize(self.img,(self.size, self.size), interpolation = cv2.INTER_CUBIC)
  def show(self):
    cv2.imshow('image',self.img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  def set_scales(self):
    if self.scales is None:
      self.scales = [cv2.resize(self.img,None,fx= 2**-(i), fy=2**-(i), interpolation = cv2.INTER_CUBIC)  for i in range(int(np.log2(self.size)))]
  def show_scale(self, scale):
      if self.scales is None:
        self.set_scales()
      cv2.imshow('image',self.scales[scale])
      cv2.waitKey(0)
      cv2.destroyAllWindows()
  def show_last(self):
    if self.scales is None:
      self.set_scales()
    self.show_scale(len(self.scales) - 1)
  def set_dogs(self):
    ngrad = 5
    if self.scales is None:
      self.set_scales()
    if self.dogs is None:
       octaves = [[cv2.GaussianBlur(self.scales[j],(5,5),sigmaX = 1.6 * self.kparam**i) for i in range(ngrad) ] for j in range(len(self.scales))]
    self.dogs = [[octaves[j][i] - octaves[j][i + 1] for i in range(ngrad - 1)] for j in range(len(self.scales))]
  def show_dog(self, i, j):
    if self.dogs is None:
      self.set_dogs()
    cv2.imshow('image',self.dogs[i][j])
    cv2.waitKey(0)
