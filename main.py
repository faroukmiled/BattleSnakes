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
import numpy as np
from torch import nn
from torch.distributions import MultivariateNormal,Categorical
# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init)

        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                             torch.nn.Conv2d(in_channels=state_dim,out_channels=2,kernel_size=3),
                             torch.nn.Conv2d(in_channels=2,out_channels=1,kernel_size=3),
                             torch.nn.Flatten(),
                             torch.nn.Linear(49, 24),
                             torch.nn.Linear(24, 24),
                             torch.nn.Linear(24, 4)
                        )
        else:
            self.actor = nn.Sequential(
                            torch.nn.Conv2d(in_channels=state_dim,out_channels=2,kernel_size=3),
                            torch.nn.Tanh(),
                            torch.nn.Conv2d(in_channels=2,out_channels=1,kernel_size=3),
                            torch.nn.Flatten(),
                            torch.nn.Tanh(),
                            torch.nn.Linear(49, 24),
                            torch.nn.Tanh(),
                            torch.nn.Linear(24, 24),
                            torch.nn.Tanh(),
                            torch.nn.Linear(24, action_dim),
                            torch.nn.Softmax(dim=-1)
                        )

        
        # critic
        self.critic = nn.Sequential(
                        torch.nn.Conv2d(in_channels=5,out_channels=2,kernel_size=3),
                        torch.nn.Tanh(),
                        torch.nn.Conv2d(in_channels=2,out_channels=1,kernel_size=3),
                        torch.nn.Flatten(),
                        torch.nn.Tanh(),
                        torch.nn.Linear(49, 24),
                        torch.nn.Tanh(),
                        torch.nn.Linear(24, 24),
                        torch.nn.Tanh(),
                        torch.nn.Linear(24, 1),
                    )
        
    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def forward(self):
        raise NotImplementedError
    

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # for single action continuous environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy
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
env_name =  "BattleSnakeEnv"
directory = "PPO_preTrained" + '/' + env_name + '/'
checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, 0, 0)
model = ActorCritic(5,4, False, 0)
model.load_state_dict(torch.load(checkpoint_path))

# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def game_state_to_env_state(game_state):
        height = game_state['board']["height"]
        width = game_state['board']["width"]
        state = np.zeros((11,11,5)).astype(np.int8)
        for food in game_state['board']['food']:
            state[(height - 1 - food['y']), food['x'],0] = 1  # Mark food with 1

        # Mark snake positions
        for i in range(len(game_state["board"]["snakes"])):
                for segment in game_state["board"]["snakes"][i]["body"]:
                    if segment == game_state["board"]["snakes"][i]["body"][0]:
                        state[(height - 1 - segment['y']), segment['x'],i+1] = 5
                    else:
                        state[(height - 1  - segment['y']), segment['x'],i+1] = 1  # Mark snake with 1
        return state
def move(game_state: typing.Dict) -> typing.Dict:
    state = game_state_to_env_state(game_state)
    state = torch.FloatTensor(np.transpose(state,(2,0,1)))
    action, action_logprob, state_val = model.act(state)
    action_dict = {0:"up",1:"down",2:"left",3:"right"}
    next_move = action_dict[action]

    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move}

def get_next_position(pos, direction):
    if direction == "up":
        return {"x": pos["x"], "y": pos["y"] + 1}
    if direction == "down":
        return {"x": pos["x"], "y": pos["y"] - 1}
    if direction == "left":
        return {"x": pos["x"] - 1, "y": pos["y"]}
    if direction == "right":
        return {"x": pos["x"] + 1, "y": pos["y"]}
    

def flood_fill(start, occupied, width, height, limit):
    stack = [start]
    visited = set()

    while stack and len(visited) <= limit:
        pos = stack.pop()
        x, y = pos["x"], pos["y"]

        if (x, y) in visited or (x, y) in occupied:
            continue

        if x < 0 or x >= width or y < 0 or y >= height:
            continue

        visited.add((x, y))

        neighbors = [
            {"x": x + 1, "y": y},
            {"x": x - 1, "y": y},
            {"x": x, "y": y + 1},
            {"x": x, "y": y - 1},
        ]
        stack.extend(neighbors)

    return len(visited)


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
