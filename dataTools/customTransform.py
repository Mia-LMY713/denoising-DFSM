import torch
import cv2
from torchvision.utils import save_image
import random

class AddGaussianNoise(object):
    def __init__(self, noise):
        self.noise = noise
        #self.mean = mean
        #self.pov = pov
    def __call__(self, tensor):
        #sigma = random.uniform(0, self.var ** self.pov)
        #noiseModel = torch.randn(tensor.size()).uniform_(0, 1.) * sigma  + self.mean
        noisyTensor = tensor + self.noise#noiseModel
        return noisyTensor#,noise 
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.var)

class NoiseModeling(object):
    def __init__(self, mean=0., var=.1, pov = 0.6):
        self.var = var
        self.mean = mean
        self.pov = pov
    def __call__(self, tensor):
        sigma = random.uniform(0, self.var ** self.pov)
        noiseModel = torch.randn(tensor.size()).uniform_(0, 1.) * sigma  + self.mean
        #noisyTensor = tensor + noiseModel
        return noise 
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.var)

