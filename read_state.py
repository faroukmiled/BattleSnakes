import json 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
with open("game_state.json","r") as f:
    game_state = json.load(f)

height = game_state["board"]["height"]
width =  game_state["board"]["width"]
board = np.ones((11,11,3)) * 255

def get_next_pos_head(action,head_pos):
    if action ==0 : 
        next_pos_head = {"x": head_pos["x"],"y":head_pos["y"] +1}
    elif action ==1 : 
        next_pos_head = {"x": head_pos["x"],"y":head_pos["y"] -1}
    elif action== 2 :
        next_pos_head = {"x": head_pos["x"] -1,"y":head_pos["y"]}
    else :
        next_pos_head = {"x": head_pos["x"] +1,"y":head_pos["y"]}
    if next_pos_head["y"]==11 or next_pos_head["y"]==-1 or next_pos_head["x"]==-1 or next_pos_head["x"]==11 :
        return -1
    else:
        return next_pos_head

def act(action,game_state):
    foods = game_state["board"]["food"]
    head_pos = game_state["you"]["body"][0]
    neck_pos = game_state["you"]["body"][1]
    tail_pos = game_state["you"]["body"][-1]
    if action ==0 : 
        next_pos_head = {"x": head_pos["x"],"y":head_pos["y"] +1}
    elif action ==1 : 
        next_pos_head = {"x": head_pos["x"],"y":head_pos["y"] -1}
    elif action== 2 :
        next_pos_head = {"x": head_pos["x"] -1,"y":head_pos["y"]}
    else :
        next_pos_head = {"x": head_pos["x"] +1,"y":head_pos["y"]}
    if next_pos_head["y"]==11 or next_pos_head["y"]==-1 or next_pos_head["x"]==-1 or next_pos_head["x"]==11 :
        return "lost"
    else :
        test = False
        food_index = -1
        for index,food in enumerate(foods):
            if next_pos_head["x"]==food["x"] and next_pos_head["y"]==food["y"]:
                test=True
                food_index = index
                break
        board[height-1-next_pos_head["y"],next_pos_head["x"]] = [255,0,0]
        test1 = False
        for index,segment in enumerate(game_state["you"]["body"]):
            if segment ==next_pos_head:
                test1 = True
        if (test1):
            return "snake ate itself"
        else:
            game_state["you"]["body"] = [{"x":next_pos_head["x"],"y":next_pos_head["y"]}] + game_state["you"]["body"]
            if (test):
                game_state["board"]["food"].pop(food_index)
                tail_pos = game_state["you"]["body"][-1]
                if tail_pos==game_state["you"]["body"][1]:
                    game_state["you"]["body"] = [{"x":next_pos_head["x"],"y":next_pos_head["y"]},tail_pos,tail_pos]
                elif len(game_state["you"]["body"])>=4:
                    if game_state["you"]["body"][-1]==game_state["you"]["body"][-2]:
                        game_state["you"]["body"].pop(-1)
            


            else:
                tail_pos  = game_state["you"]["body"][-1]
                board[height-1-tail_pos["y"],tail_pos["x"]] = [255,255,255]
                index_tail = len(game_state["you"]["body"]) -1
                for index_seg, segment in enumerate(game_state["you"]["body"]):
                    if segment==tail_pos:
                        index_tail = index_seg
                        break
                game_state["you"]["body"] =  game_state["you"]["body"][:index_tail]
                while(len(game_state["you"]["body"])<3):
                    game_state["you"]["body"].append(game_state["you"]["body"][-1])
    return game_state

def get_board(game_state):
    for food in game_state["board"]["food"] : 
        board[height - 1 - food["y"],food["x"]] = [0,255,0]
    for segment in game_state["you"]["body"]:
        board[height - 1 - segment["y"],segment["x"]] = [255,0,0]
    return board

def generate_new_food(game_state):
    while True:
         x = np.random.randint(0,11)
         y = np.random.randint(0,11)
         test1 = False
         for segment in game_state["you"]["body"]:
             if segment == {"x":x,"y":y}:
                 test1=True
         if (test1==False):
             game_state["board"]["food"].append({"x":x,"y":y})
             break
    return game_state
def game_state_to_env_state(game_state):
    state = np.zeros((11,11,2))
    for food in game_state['board']['food']:
        state[(height - 1 - food['y']), food['x'],0] = 1  # Mark food with 1

    # Mark snake positions
    for segment in game_state["you"]["body"]:
            if segment == game_state["you"]['body'][0]:
                state[(height - 1 - segment['y']), segment['x'],1] = 5
            else:
                state[(height - 1  - segment['y']), segment['x'],1] = 1  # Mark snake with 1
    return state
             
"""
episode_length = 100
iter = 0
while iter<episode_length:
    game_state = generate_new_food(game_state)
    action = np.random.randint(0,4)
    while(get_next_pos_head(action,game_state["you"]["body"][0])==-1):
        action = np.random.randint(0,4)

    iter+=1
    game_state = act(action,game_state)
    if game_state=="lost" or game_state=="snake ate itself":
        print(game_state)
        break
    state = game_state_to_env_state(game_state)
    print(state[:,:,1])
    board = get_board(game_state)
    plt.imshow(board)
    plt.axis("off")
    plt.show(block=False)  # Do not block the execution
    plt.pause(0.5)          # Pause for 0.5 seconds before the next iteration
    plt.clf()   

plt.close()"""
