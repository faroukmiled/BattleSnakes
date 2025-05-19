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


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:

    is_move_safe = {"up": True, "down": True, "left": True, "right": True}
    if not os.path.exists("game_state.json"):
        with open("game_state.json","w") as f:
            json.dump(game_state,f)

    # We've included code to prevent your Battlesnake from moving backwards
    my_head = game_state["you"]["body"][0]  # Coordinates of your head
    my_neck = game_state["you"]["body"][1]  # Coordinates of your "neck"

    if my_neck["x"] < my_head["x"]:  # Neck is left of head, don't move left
        is_move_safe["left"] = False

    elif my_neck["x"] > my_head["x"]:  # Neck is right of head, don't move right
        is_move_safe["right"] = False

    elif my_neck["y"] < my_head["y"]:  # Neck is below head, don't move down
        is_move_safe["down"] = False

    elif my_neck["y"] > my_head["y"]:  # Neck is above head, don't move up
        is_move_safe["up"] = False

    # TODO: Step 1 - Prevent your Battlesnake from moving out of bounds
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']

    if my_head["x"] == 0:
        is_move_safe["left"] = False
    if my_head["x"] == board_width - 1:
        is_move_safe["right"] = False
    if my_head["y"] == 0:
        is_move_safe["down"] = False
    if my_head["y"] == board_height - 1:
        is_move_safe["up"] = False

    # TODO: Step 2 - Prevent your Battlesnake from colliding with itself
    my_body = game_state["you"]["body"]
    body_coords = [(segment["x"], segment["y"]) for segment in my_body]
    for direction in is_move_safe.keys():
        if is_move_safe[direction]:
            next_pos = get_next_position(my_head, direction)
            if (next_pos["x"], next_pos["y"]) in body_coords:
                is_move_safe[direction] = False

    # TODO: Step 3 - Prevent your Battlesnake from colliding with other Battlesnakes
    opponents = game_state['board']['snakes']
    opponent_coords = set()
    for snake in opponents:
        if snake["id"] != game_state["you"]["id"]:
            for segment in snake["body"]:
                opponent_coords.add((segment["x"], segment["y"]))

    for direction in list(is_move_safe):
        if not is_move_safe[direction]:
            continue
        next_pos = get_next_position(my_head, direction)
        if (next_pos["x"], next_pos["y"]) in opponent_coords:
            is_move_safe[direction] = False
            
    # TODO: Step 3.5: Predict opponent head moves and avoid dangerous head-on tiles
    danger_zones = set()
    my_length = len(my_body)

    for snake in opponents:
        if snake["id"] == game_state["you"]["id"]:
            continue  # skip self

        their_head = snake["body"][0]
        their_length = len(snake["body"])

        if their_length >= my_length:
            # They can kill us in a head-on collision
            for direction in ["up", "down", "left", "right"]:
                target = get_next_position(their_head, direction)
                danger_zones.add((target["x"], target["y"]))

    # Now block moves that lead us into a head-on danger zone
    for direction in list(is_move_safe):
        if not is_move_safe[direction]:
            continue
        next_pos = get_next_position(my_head, direction)
        if (next_pos["x"], next_pos["y"]) in danger_zones:
            is_move_safe[direction] = False
            
    # Step 5: Avoid trapping self
    occupied = set(body_coords).union(opponent_coords)
    my_length = len(my_body)
    for direction in list(is_move_safe):
        if not is_move_safe[direction]:
            continue
        next_pos = get_next_position(my_head, direction)
        reachable_space = flood_fill(
            next_pos,
            occupied,
            board_width,
            board_height,
            limit=my_length + 2  # give it a bit more than length buffer
        )
        if reachable_space < my_length:
            is_move_safe[direction] = False

    # Are there any safe moves left?
    safe_moves = []
    for move, isSafe in is_move_safe.items():
        if isSafe:
            safe_moves.append(move)

    if len(safe_moves) == 0:
        print(
            f"MOVE {game_state['turn']}: No safe moves detected! Moving down")
        return {"move": "down"}

    # Choose a random move from the safe ones
    next_move = random.choice(safe_moves)

    # TODO: Step 4 - Move towards food instead of random, to regain health and survive longer
    safe_moves = [move for move, safe in is_move_safe.items() if safe]
    food = game_state["board"]["food"]
    my_health = game_state["you"]["health"]

    if safe_moves and food and my_health < 40:
        # Try to move toward nearest food
        food.sort(key=lambda f: abs(f["x"] - my_head["x"]) + abs(f["y"] - my_head["y"]))
        target = food[0]
        dx = target["x"] - my_head["x"]
        dy = target["y"] - my_head["y"]

        food_moves = []
        if dx < 0 and is_move_safe["left"]:
            food_moves.append("left")
        if dx > 0 and is_move_safe["right"]:
            food_moves.append("right")
        if dy < 0 and is_move_safe["down"]:
            food_moves.append("down")
        if dy > 0 and is_move_safe["up"]:
            food_moves.append("up")

        if food_moves:
            next_move = random.choice(food_moves)
            print(f"MOVE {game_state['turn']}: Low health, chasing food -> {next_move}")
            return {"move": next_move}

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
