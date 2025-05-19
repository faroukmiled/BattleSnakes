# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#
# This file can be a nice home for your Battlesnake logic and helper functions.
#
# To get you started we've included code to prevent your Battlesnake from moving backwards.
# For more info see docs.battlesnake.com

import random
import typing
import json
import os
import torch
from create_env import DQN
import numpy as np

model = DQN(4, 5)
model.load_state_dict(torch.load("policy_net.pth"))


# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "Group 7",  # TODO: Your Battlesnake Username
        "color": "#FF0000",  # TODO: Choose color
        "head": "default",  # TODO: Choose head
        "tail": "default",  # TODO: Choose tail
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")


def game_state_to_env_state(game_state):
    height = game_state['board']["height"]
    width = game_state['board']["width"]
    state = np.zeros((11, 11, 5)).astype(np.int8)
    for food in game_state['board']['food']:
        state[(height - 1 - food['y']), food['x'], 0] = 1  # Mark food with 1

    # Mark snake positions
    for i in range(len(game_state["board"]["snakes"])):
        for segment in game_state["board"]["snakes"][i]["body"]:
            if segment == game_state["board"]["snakes"][i]["body"][0]:
                state[(height - 1 - segment['y']), segment['x'], i + 1] = 5
            else:
                state[(height - 1 - segment['y']), segment['x'],
                      i + 1] = 1  # Mark snake with 1
    return state


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:
    action_dict = {0: "up", 1: "down", 2: "left", 3: "right"}

    state = game_state_to_env_state(game_state)
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action = model(state).argmax().item()
    next_move = action_dict[action]
    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
