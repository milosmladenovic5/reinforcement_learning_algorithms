"""
    preprocess_frame:
    Take a frame.
    Resize it.
        __________________
        |                 |
        |                 |
        |                 |
        |                 |
        |_________________|
        
        to
        _____________
        |            |
        |            |
        |            |
        |____________|
    Normalize it.
    
    return preprocessed_frame
    
"""
from skimage import transform

def preprocess_frame(frame):
    #grayscale already done in config file
    # x = np.mean(frame, -1)

    #Crop the screen (remove the roof because it doesn't contain
    # useful information)
    cropped_frame = frame[80:,:]
    normalized_frame = cropped_frame / 255.0

    preprocessed_frame = transform.resize(normalized_frame, [84,84])

    return preprocessed_frame

