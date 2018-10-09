from skimage import transform

def preprocess_frame(frame):
   # Crop the screen (remove part that contains no information)
    # [Up: Down, Left: right]
    cropped_frame = frame[15:-5,20:-20]
    
    # Normalize Pixel Values
    normalized_frame = cropped_frame/255.0
    
    # Resize
    preprocessed_frame = transform.resize(cropped_frame, [100,120])
    
    return preprocessed_frame # 100x120x1 frame

