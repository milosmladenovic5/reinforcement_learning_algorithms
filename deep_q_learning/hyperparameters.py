state_size = [84, 84, 4] # input is a stack of 4 frames, with h x w  = 84 x 84
action_size = game.get_available_buttons_size() # 3 possible actions
learning_rate = 0.
