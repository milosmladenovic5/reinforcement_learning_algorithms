import random               
import time                  
from skimage import transform
from collections import deque# Ordered collection with ends

""" 

Take a frame, crop it, normalize it and then resize it.

"""

def preprocess_frame(frame):
    #grayscale frame is already done in the vizdoom config
    # x = np.mean(frame, -1)
    
    cropped_frame = frame[30:-10, 30:-30] # -10 in x, -30 in y axis

    normalized_frame = cropped_frame / 255.0

    #resize
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])

    return preprocessed_frame
    