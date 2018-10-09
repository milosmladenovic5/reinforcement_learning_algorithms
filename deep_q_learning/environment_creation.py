from vizdoom import *       # Doom Environment
import random                # Handling random number generation
from collections import deque# Ordered collection with ends
import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

def create_environment():
    game = DoomGame()

    game.load_config("basic.cfg")

    game.set_doom_scenario_path("basic.wad")

    game.init()

    #now we list all possible actions, one hot encoded
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]

    return game, possible_actions
