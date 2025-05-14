# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# or in the "license" file accompanying this file. This file is distributed 
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
# express or implied. See the License for the specific language governing 
# permissions and limitations under the License.

import numpy as np

class Rewards:
    '''
    Base class to set up rewards for the battlesnake gym
    '''
    def get_reward(self, name, snake_id, episode):
        raise NotImplemented()

class SimpleRewards(Rewards):
    '''
    Simple class to handle a fixed reward scheme
    '''
    def __init__(self):
        self.reward_dict = {
            "another_turn": 1,             # Small reward for staying alive
            "ate_food": 0.5,                 # Big reward for eating
            "won": 2,                     # Large reward for winning
            "died": -3.0,                    # Penalty for dying
            "ate_another_snake": 0.0,        # Huge reward if snake kills another
            "hit_wall": 0.0,                # Wall collisions are bad
            "hit_other_snake": 0.0,         # Hitting another snake is also bad
            "hit_self": -3.0,                # Running into own body
            "was_eaten": 0.0,               # Died because of another snake
            "other_snake_hit_body": 0.0,     # Reward for being solid — other snake hit us
            "forbidden_move": -3.0,          # Illegal move penalty
            "starved": -2.0                 # Ran out of health
        }

    def get_reward(self, name, snake_id, episode):
        return self.reward_dict[name]