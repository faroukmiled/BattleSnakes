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
import gymnasium as gym
from gymnasium import spaces
import math

def is_coord_in(coord, array):
    for a in array:
        if a[0] == coord[0] and a[1] == coord[1]:
            return True
    return False

def get_random_coordinates(map_size, n, excluding=[]):
    '''
    Helper function to get n number of random coordinates based on the map
    Parameters:
    ----------
    map_size, (int, int)
        Size of the map with possible coordinates
    
    n, int
        number of coordinates to get

    excluding: [(int, int)]
        A list of coordinates to not include in the randomly generated coordinates
    '''
    coordinates_indexes = []
    coordinates = []
    count = 0
    for i in range(map_size[0]):
        for j in range(map_size[1]):
            if is_coord_in(coord=(i, j), array=excluding):
                continue
            coordinates.append((i, j))
            coordinates_indexes.append(count)
            count += 1

    indexes = np.random.choice(coordinates_indexes, n, replace=False)
    random_coordinates = np.array(coordinates)[indexes]
    return random_coordinates

def generate_coordinate_list_from_binary_map(map_image):
    '''
    Helper function to convert binary maps into a list of coordinates
    '''
    coordinate_list = []
    for i in range(map_image.shape[0]):
        for j in range(map_image.shape[1]):
            if map_image[i][j] > 0:
                coordinate_list.append((i, j))
    return coordinate_list


class MultiAgentActionSpace(spaces.Space):
    def __new__(cls, agents_action_spaces):
        if len(agents_action_spaces) == 1:
            # Single agent case → just return the Discrete space itself
            return agents_action_spaces[0]
        return super().__new__(cls)

    def __init__(self, agents_action_spaces):
        if len(agents_action_spaces) == 1:
            # __new__ returned Discrete(4), __init__ won't run
            return
        self.agents_action_spaces = agents_action_spaces
        self.n = len(agents_action_spaces)
        super().__init__(shape=(self.n,), dtype=np.int64)

    def sample(self):
        return [space.sample() for space in self.agents_action_spaces]

    def contains(self, x):
        return isinstance(x, (list, tuple)) and all(
            space.contains(a) for space, a in zip(self.agents_action_spaces, x)
        )

    def __repr__(self):
        return f"MultiAgentActionSpace({self.agents_action_spaces})"

def get_distance(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)