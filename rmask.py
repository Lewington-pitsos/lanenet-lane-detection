import imageio
import numpy as np

import matplotlib.pyplot as plt


def interpolate_gaps(indices):
    prev = 0
    last_proper_index = 0
    currently_bad = False
    new_indices = []
    
    for index, val in enumerate(indices):
        if val > 0:
            if currently_bad:
                mean_value = int((prev + val) / 2) if prev > 0 else val
                for i in range(index - last_proper_index - 1):
                    new_indices.append(mean_value)
            
            currently_bad = False
            last_proper_index = index
            prev = val
            new_indices.append(val)
        else:
            currently_bad = True
            
    return new_indices


def highest_pixels(img, threshold=0):
    highest = []
    for i in range(img.shape[1]):
        column = img[:, i, 0]
        wanted = np.where(column > threshold)[0]
        if len(wanted) == 0:
            highest.append(0)
        else:
            highest.append(wanted[0])
            
    return highest

def create_mask(indices, image_height):
    arrays = []
    
    for i in indices:
        column_list = [0] * i
        column_list.extend([1] * (image_height - i))

        arrays.append(column_list)
    
    return np.transpose(np.array(arrays))

def cut_to_uppermost(img, cutoff):
    uppermost = np.min(cutoff)
  
    return img[uppermost:, :, :], np.array(cutoff) - uppermost

class RoadCropProc():
    def __init__(self, model_load_fn):
        self.model_load_fn = model_load_fn
        self.loaded = False
        self.model = None
        
    def __call__(self, img, labels):
        if not self.loaded:
            self.model = self.model_load_fn()
            self.loaded = True
        
        binary = self.model(img)
        
        cutoff = interpolate_gaps(highest_pixels(binary))
        
        img, cutoff = cut_to_uppermost(img, cutoff)
        mask = create_mask(cutoff, img.shape[0])        
        cropped_img = img * mask[:, :, None]
        
        return cropped_img, labels
        
    