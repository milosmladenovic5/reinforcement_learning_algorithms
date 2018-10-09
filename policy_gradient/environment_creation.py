from vizdoom import *
import random
import numpy as np

def create_environment():
    game = DoomGame()

    #Load correct configuration
    game.load_config("health_gathering.cfg") #type of game to play

    #load the correct scenario (defend the center scenario in this case)
    game.set_doom_scenario_path("health_gathering.wad")

    game.init()

    #hardcoded possible actions
    possible_actions = np.identity(3, dtype = int).tolist()

    return game, possible_actions
