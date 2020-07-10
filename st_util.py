import os
import sys
import torch
from collections import defaultdict
import PIL
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from pathlib import Path
from random import shuffle
import torch.nn as nn
import torch.optim as optim
import io
import logging
import json
import base64

import base64
import io
import matplotlib.pyplot as plt
from PIL import Image

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
size = 300
padding = 30
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'


def compose(x, funcs, *args, order_key='_order', **kwargs):
    key = lambda o: getattr(o, order_key, 0)
    for f in sorted(list(funcs), key=key): x = f(x, **kwargs)
    return x

class Transform(): _order=0
        
class MakeRGB(Transform):
    def __call__(self, item): return {k: v.convert('RGB') for k, v in item.items()}

class ResizeFixed(Transform):
    _order=10
    def __init__(self, size):
        if isinstance(size,int): size=(size,size)
        self.size = size
        
    def __call__(self, item): return {k: v.resize(self.size, PIL.Image.BILINEAR) for k, v in item.items()}

class ToByteTensor(Transform):
    _order=20
    def to_byte_tensor(self, item):
        res = torch.ByteTensor(torch.ByteStorage.from_buffer(item.tobytes()))
        w,h = item.size
        return res.view(h,w,-1).permute(2,0,1)
    
    def __call__(self, item): return {k: self.to_byte_tensor(v) for k, v in item.items()}


class ToFloatTensor(Transform):
    _order=30
    def to_float_tensor(self, item): return item.float().div_(255.)
    
    def __call__(self, item): return {k: self.to_float_tensor(v) for k, v in item.items()}
    
class Normalize(Transform):
    _order=40
    def __init__(self, stats, p=None):
        self.mean = torch.as_tensor(stats[0] , dtype=torch.float32)
        self.std = torch.as_tensor(stats[1] , dtype=torch.float32)
        self.p = p
    
    def normalize(self, item): return item.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
    def pad(self, item): return nn.functional.pad(item[None], pad=(self.p,self.p,self.p,self.p), mode='replicate').squeeze(0)
    
    def __call__(self, item): 
        if self.p is not None: return {k: self.pad(self.normalize(v)) for k, v in item.items()}
        else: return {k: self.normalize(v) for k, v in item.items()}

class DeProcess(Transform):
    _order=50
    def __init__(self, stats, size=None, p=None, ori=None):
        self.mean = torch.as_tensor(stats[0] , dtype=torch.float32)
        self.std = torch.as_tensor(stats[1] , dtype=torch.float32)
        self.size = size
        self.p = p
        self.ori = ori
    
    def de_normalize(self, item): return ((item*self.std[:, None, None]+self.mean[:, None, None])*255.).clamp(0, 255)
    def rearrange_axis(self, item): return np.moveaxis(item, 0, -1)
    def to_np(self, item): return np.uint8(np.array(item))
    def crop(self, item): return item[self.p:self.p+self.size,self.p:self.p+self.size,:]
    def de_process(self, item): 
        return PIL.Image.fromarray(self.crop(self.rearrange_axis(self.to_np(self.de_normalize(item))))).resize(self.ori, PIL.Image.BICUBIC)
                
    def __call__(self, item): 
        if isinstance(item, torch.Tensor): return self.de_process(item) 
        if isinstance(item, tuple): return tuple([self.de_process(v) for v in item])
        if isinstance(item, dict): return {k: self.de_process(v) for k, v in item.items()}

        
def image_to_base64(image):
    # Make the image the correct format
    fd = io.BytesIO()
    # Save the image as PNG
    image.save(fd, format="PNG")
    return base64.b64encode(fd.getvalue())

def base64_to_image(data: bytes) -> np.ndarray:
    """Convert an image in base64 to a numpy array"""
    b64_image = base64.b64decode(data)
    fd = io.BytesIO(b64_image)
    img = PIL.Image.open(fd)
    #img_data = np.array(img).astype("float32")

    #if img_data.shape[-1] == 4:
    #    # We only support rgb
    #    img_data = img_data[:, :, :3]

    return img