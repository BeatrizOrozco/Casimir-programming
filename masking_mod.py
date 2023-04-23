import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from skimage import data, io, filters, viewer, measure
#from scipy import constants
#from PIL import Image
from skimage.segmentation import clear_border as clear_border
from skimage.measure import label,regionprops
from matplotlib.colors import LinearSegmentedColormap


def masking (image, threshold, num_labels):
    mask=image > threshold
    mask = np.vectorize(clear_border, signature='(n,m)->(n,m)')(mask)
    ##label the masked sections of code
    mask_labeled = np.vectorize(label, signature='(n,m)->(n,m)')(mask)
    #seperate the sections of channel 3
    sl=mask_labeled
    rps=regionprops(sl)
    areas=[r.area for r in rps]
    labels = label(mask, background=0) #remove background
    #sort the sections of the channel 3 from largest to smallest sections. This is to keep the largest sections.
    idxs=np.argsort(areas)[::-1]
    #print(idxs)
    zero_sl=np.zeros_like(sl)
    for i in idxs[:num_labels]:  #add counting algorithm for # of sections!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        zero_sl[tuple(rps[i].coords.T)] = i+1
        
        
    zero_sl[zero_sl>0]=255 #remove background noise to intensity set to 255
    
    return (zero_sl)