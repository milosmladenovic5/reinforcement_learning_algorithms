import frame_preprocessing as fp
from collections import deque
import numpy as np

def stack_frames (stacked_frames, state, is_new_episode, stack_size):
    frame = fp.preprocess_frame(state)

    if is_new_episode:
        #clear stacked frames
        stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4) 

        for i in (0, stack_size):
            stacked_frames.append(frame)
    else:
        #append the frame to deque, automatically removes the oldest one
        stacked_frames.append(frame)

        #build the stacked state

    stacked_state = np.stack(stacked_frames, axis = 2)
    return stacked_state, stacked_frames