import gymnasium as gym
import torch
import numpy as np
from typing import Optional
import os
import torch
import json
import matplotlib.pyplot as plt
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(torch.nn.Module):
    def __init__(self,output_size,in_channels):
        super(DQN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels,out_channels=2,kernel_size=3)
        self.conv2 = torch.nn.Conv2d(in_channels=2,out_channels=1,kernel_size=3)
        self.flatten1 = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(49, 24)
        self.fc2 = torch.nn.Linear(24, 24)
        self.fc3 = torch.nn.Linear(24, output_size)

    def forward(self, x):
        x = torch.nn.LeakyReLU(0.2)(self.conv1(x))
        x = torch.nn.LeakyReLU(0.2)(self.flatten1(self.conv2(x)))
        x = torch.nn.LeakyReLU(0.2)((self.fc1(x)))
        x = torch.nn.LeakyReLU(0.2)(self.fc2(x))
        return torch.nn.Softmax()(self.fc3(x))
class BattleSnakeEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "ascii"],
        "render_fps" : 30,
        "size": 11
        }
    def __init__(self,game_state,size = 11,render_mode = "rgb_array"):
         
        # The size of the square grid
        self.render_mode = render_mode
        self.size = size
        self.metadata["size"] = size
        self.turn_count = 0
        self.game_state = game_state
        self.health = 100
        self.snake_length = 0
        self.number_snakes  = 4
        self.previous_pos_heads = []

        # Define the agent and target location; randomly chosen in `reset` and updated in `step`

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2
        self.observation_space = gym.spaces.Box(low=0, high=5,
                                           shape=(11,
                                                  11,
                                                  self.number_snakes+1),
                                           dtype=np.int8)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(4) 
    def generate_new_food(self):
        while True:
            x = np.random.randint(0,11)
            y = np.random.randint(0,11)
            test1 = False
            for i in range(len(self.game_state["board"]["snakes"])):
                for segment in self.game_state["board"]["snakes"][i]["body"]:
                    if segment == {"x":x,"y":y}:
                        test1=True
            if (test1==False):
                self.game_state["board"]["food"].append({"x":x,"y":y})
                break
        return self.game_state
    def get_observation_space(self):
        return self.observation_space
    def _get_info(self):
        return {"len_snake":len(self.game_state["you"]["body"]),"food_number":len(self.game_state["board"]["food"])}
    
    def get_next_pos_head(self,action,head_pos):
        if action ==0 : 
            next_pos_head = {"x": head_pos["x"],"y":head_pos["y"] +1}
        elif action ==1 : 
            next_pos_head = {"x": head_pos["x"],"y":head_pos["y"] -1}
        elif action== 2 :
            next_pos_head = {"x": head_pos["x"] -1,"y":head_pos["y"]}
        else :
            next_pos_head = {"x": head_pos["x"] +1,"y":head_pos["y"]}
        return next_pos_head
    def take_action(self,action,snake_idx):
        if len(self.game_state["board"]["snakes"][snake_idx]["body"])==0:
            return 
        foods = self.game_state["board"]["food"]
        head_pos = self.game_state["board"]["snakes"][snake_idx]["body"][0]
        neck_pos = self.game_state["board"]["snakes"][snake_idx]["body"][1]
        tail_pos = self.game_state["board"]["snakes"][snake_idx]["body"][-1]
        height = self.game_state["board"]["height"]
        width = self.game_state["board"]["width"]
        next_pos_head = self.get_next_pos_head(action,head_pos)
        if next_pos_head["y"]==11 or next_pos_head["y"]==-1 or next_pos_head["x"]==-1 or next_pos_head["x"]==11 :
            self.game_state["board"]["snakes"][snake_idx]["body"] = []
            self.game_state["board"]["snakes"][snake_idx]["length"] = 0
            self.game_state["board"]["snakes"][snake_idx]["health"] = 0
            #print(f"snake number {snake_idx} : hit wall")
        else :
            test = False
            food_index = -1
            for index,food in enumerate(foods):
                if next_pos_head["x"]==food["x"] and next_pos_head["y"]==food["y"]:
                    test=True
                    food_index = index
                    break
            test1 = False
            for index,segment in enumerate(self.game_state["board"]["snakes"][snake_idx]["body"]):
                if segment ==next_pos_head:
                    test1 = True
            if (test1):
                self.game_state["board"]["snakes"][snake_idx]["body"] = []
                self.game_state["board"]["snakes"][snake_idx]["length"] = 0
                self.game_state["board"]["snakes"][snake_idx]["health"] = 0
                #print(f"snake  {snake_idx} : ate itself")
            else:
                self.game_state["board"]["snakes"][snake_idx]["body"] = [{"x":next_pos_head["x"],"y":next_pos_head["y"]}] + self.game_state["board"]["snakes"][snake_idx]["body"]
                if (test):
                    self.game_state["board"]["food"].pop(food_index)
                    tail_pos =self.game_state["board"]["snakes"][snake_idx]["body"][-1]
                    if tail_pos== self.game_state["board"]["snakes"][snake_idx]["body"][1]:
                        self.game_state["board"]["snakes"][snake_idx]["body"] = [{"x":next_pos_head["x"],"y":next_pos_head["y"]},tail_pos,tail_pos]
                    elif len(self.game_state["board"]["snakes"][snake_idx]["body"])>=4:
                        if self.game_state["board"]["snakes"][snake_idx]["body"][-1]==self.game_state["board"]["snakes"][snake_idx]["body"][-2]:
                            self.game_state["board"]["snakes"][snake_idx]["body"].pop(-1)
                    self.game_state["board"]["snakes"][snake_idx]["length"] +=1
                


                else:
                    tail_pos  = self.game_state["board"]["snakes"][snake_idx]["body"][-1]
                    index_tail = len(self.game_state["board"]["snakes"][snake_idx]["body"]) -1
                    for index_seg, segment in enumerate(self.game_state["board"]["snakes"][snake_idx]["body"]):
                        if segment==tail_pos:
                            index_tail = index_seg
                            break
                    self.game_state["board"]["snakes"][snake_idx]["body"] =  self.game_state["board"]["snakes"][snake_idx]["body"][:index_tail]
                    while(len(self.game_state["board"]["snakes"][snake_idx]["body"])<3):
                        self.game_state["board"]["snakes"][snake_idx]["body"].append(self.game_state["board"]["snakes"][snake_idx]["body"][-1])
                    self.game_state["board"]["snakes"][snake_idx]["health"] -=1

    def check_hit_snake_head_to_head(self,snake_id,other_snake_id):
        if self.previous_pos_heads[snake_id] == self.game_state["board"]["snakes"][other_snake_id]["body"][0] and self.previous_pos_heads[other_snake_id]==self.game_state["board"]["snakes"][snake_id]["body"][0]:
            return True
        return self.game_state["board"]["snakes"][snake_id]["body"][0] == self.game_state["board"]["snakes"][other_snake_id]["body"][0]
    
    def check_hit_snake_body(self,snake_id,other_snake_id):
        head_pos = self.game_state["board"]["snakes"][snake_id]["body"][0]
        for _,segment in self.game_state["board"]["snakes"][other_snake_id]["body"]:
            if segment==head_pos:
                return True
        return False
    def snake_survives(self,snake_id):
        if len(self.game_state["board"]["snakes"][snake_id]["body"])==0 or self.game_state["board"]["snakes"][snake_id]["health"] ==0:
            return False
        for other_snake_id in range(self.number_snakes):
            if len(self.game_state["board"]["snakes"][other_snake_id]["body"])==0 or other_snake_id==snake_id or self.game_state["board"]["snakes"][other_snake_id]["health"] ==0:
                continue
            test_hit_other_snake_head_to_head = self.check_hit_snake_head_to_head(snake_id,other_snake_id)
            #if (test_hit_other_snake_head_to_head):
                #print(f"snake {snake_id} : hit another snake head to head")
            length_snake = self.game_state["board"]["snakes"][snake_id]["length"]
            length_other_snake = self.game_state["board"]["snakes"][other_snake_id]["length"]
            if test_hit_other_snake_head_to_head:
                if length_snake>=length_other_snake:
                    self.game_state["board"]["snakes"][other_snake_id]["body"] = []
                    self.game_state["board"]["snakes"][other_snake_id]["length"] = 0
                    self.game_state["board"]["snakes"][other_snake_id]["health"] = 0
                    if length_snake==length_other_snake:
                        return False
                elif length_snake<length_other_snake:
                    return False
            else:
                test_hit_other_snake_body = self.check_hit_snake_body(snake_id,other_snake_id)
                if (test_hit_other_snake_body):
                    #print(f"snake {snake_id} : hit another snake body")
                    return  False
        return True
    def step(self,action):
        self.turn_count +=1
        self.take_action(action,0)
        action_snake1 = np.random.randint(4)
        action_snake2 = np.random.randint(4)
        action_snake3 = np.random.randint(4)
        #print(f" snake 0 : action : {action}")
        self.take_action(action_snake1,1)
        #print(f" snake 1 : action : {action_snake1}")
        self.take_action(action_snake2,2)
        #print(f" snake 2 : action : {action_snake2}")
        self.take_action(action_snake3,3)
        #print(f" snake 2 : action : {action_snake3}")
        main_snake_survives = self.snake_survives(0)
        if not main_snake_survives:
            terminated = True
            reward =   -1
            info = self._get_info()
            return self.game_state_to_env_state(),reward,terminated,False,info
        else:
            snake1_survives = self.snake_survives(1)
            #print(f"snake  one  survives : {snake1_survives}")
            snake2_survives = self.snake_survives(2)
            #print(f"snake two survives : {snake2_survives}")
            snake3_survives = self.snake_survives(3)
            #print(f"snake three survives : {snake3_survives}")
            self.previous_pos_heads[0] = self.game_state["board"]["snakes"][0]["body"][0]
            if (snake1_survives):
                self.previous_pos_heads[1] = self.game_state["board"]["snakes"][1]["body"][0]
            if (snake2_survives):
                self.previous_pos_heads[2] = self.game_state["board"]["snakes"][2]["body"][0]
            if (snake3_survives):
                self.previous_pos_heads[3] = self.game_state["board"]["snakes"][3]["body"][0]
            if not (snake1_survives or snake2_survives or snake3_survives):
                terminated = True
                #print("snake wins")
                reward = 1
                info = self._get_info()
                return "Win",reward,terminated,False,info
        terminated = False
        reward =   0.002
        info = self._get_info()
        self.game_state  = self.generate_new_food()
        #print(f'length of snake 1 : {self.game_state["board"]["snakes"][1]["length"]}')
        #print(f'length of snake 2 : {self.game_state["board"]["snakes"][2]["length"]}')
        #print(f'length of snake 3 : {self.game_state["board"]["snakes"][3]["length"]}')
        return self.game_state_to_env_state(),reward,terminated,False,info
        
    def game_state_to_env_state(self):
        height = self.game_state['board']["height"]
        width = self.game_state['board']["width"]
        state = np.zeros((11,11,self.number_snakes+1)).astype(np.int8)
        for food in self.game_state['board']['food']:
            state[(height - 1 - food['y']), food['x'],0] = 1  # Mark food with 1

        # Mark snake positions
        for i in range(len(self.game_state["board"]["snakes"])):
                for segment in self.game_state["board"]["snakes"][i]["body"]:
                    if segment == self.game_state["board"]["snakes"][i]["body"][0]:
                        state[(height - 1 - segment['y']), segment['x'],i+1] = 5
                    else:
                        state[(height - 1  - segment['y']), segment['x'],i+1] = 1  # Mark snake with 1
        return state

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.turn_count = 0
        game_state = {}
        game_state["you"] = {}
        d = {"x":np.random.randint(0,11),"y":np.random.randint(0,11)}
        game_state["you"]["body"] = [d,d,d]
        game_state["board"] = {}
        game_state["board"]["food"] = []
        game_state["board"]["height"] = 11
        game_state["board"]["width"] = 11
        game_state["board"]["snakes"] = []
        game_state["board"]["snakes"].append({"body":game_state["you"]["body"],"health":100,"length":3})
        snakes_pos = [d]
        self.previous_pos_heads = [d]
        index = 1
        while index<self.number_snakes:
            d1 = {"x":np.random.randint(0,11),"y":np.random.randint(0,11)}
            if snakes_pos.count(d1)==0:
                game_state["board"]["snakes"].append({"body":[d1,d1,d1],"health":100,"length":3})
                index+=1
                snakes_pos.append(d1)
                self.previous_pos_heads.append(d1)

        self.game_state = game_state
        self.game_state  = self.generate_new_food()

        # We will sample the target's location randomly until it does not coincide with the agent's location

        observation = self.game_state_to_env_state()
        info = self._get_info()

        return observation,info
    def render(self, mode="rgb_array"):
        '''
        Inherited function from openAI gym to visualise the progression of the gym
        
        Parameter:
        ---------
        mode: str, options=["human", "rgb_array"]
            mode == human will present the gym in a separate window
            mode == rgb_array will return the gym in np.arrays
        '''
        state = self.game_state_to_env_state()
        board = np.ones((11,11,3)).astype(np.uint8)*255
        board[state[:,:,0]==1] = [0,255,0]
        board[state[:,:,1]==5] = [255,0,0]
        board[state[:,:,1]==1] = [0,0,255]
        if mode == "rgb_array":
            return board
        elif mode == "ascii":
            ascii = self._get_ascii()
            print(ascii)
            # for _ in range(ascii.count('\n')):
            #     print("\033[A")
            return ascii
        elif mode == "human":
            plt.imshow(board)
            plt.axis('off')  # Hide axes
            plt.show()
            return 
    def get_board(self):
        state = self.game_state_to_env_state()
        board = np.ones((11,11,3))*255
        board[state[:,:,0]==1] = [0,255,0]
        board[state[:,:,1]==5] = [255,0,0]
        board[state[:,:,1]==1] = [0,0,255]
        return board


os.environ['SDL_AUDIODRIVER'] = 'directx' 
gym.register(
    id="BattleSnakeEnv",
    entry_point=BattleSnakeEnv
)
"""
pygame.init()
DISPLAYSURF = pygame.display.set_mode((512,512),0,32)
clock = pygame.time.Clock()
pygame.display.flip()

training_period = 250  # record the agent's episode every 250
num_training_episodes = 1  # total number of training episodes
with open("game_state.json","r") as f:
    game_state = json.load(f)"""
env = gym.make("BattleSnakeEnv",game_state ={},size = 11, render_mode = "rgb_array")  # replace with your environment
"""
def get_board(obs):
    board = np.ones((11,11,3))*255
    board[obs[:,:,0]==1] = [0,255,0] ### green for food
    board[obs[:,:,1]==5] = [255,0,0]  ### red head for main snake
    board[obs[:,:,1]==1] = [128,0,0] ### 
    board[obs[:,:,2]==5] = [0,0,255] ### blue head for snake 2
    board[obs[:,:,2]==1] = [0,0,128]
    board[obs[:,:,3]==5] = [0,255,255]
    board[obs[:,:,3]==1] = [0,128,128]
    board[obs[:,:,4]==5] = [255,255,0]
    board[obs[:,:,4]==1] = [128,128,0]
    
    return board

obs,info = env.reset()
output_size = env.action_space.n
input_size = env.observation_space.shape[0]  # 4 inputs: position, velocity, angle, angular velocity
output_size = env.action_space.n  # 4 outputs: left or right
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    if not done:
        board = get_board(obs)
        plt.imshow(board)
        plt.axis("off")
        plt.show(block=False)  # Do not block the execution
        plt.pause(0.5)          # Pause for 0.5 seconds before the next iteration
        plt.clf()  



policy_net = DQN(output_size, 5)
target_net = DQN(output_size, 5)
target_net.load_state_dict(policy_net.state_dict())
observation = np.transpose(obs,(2,0,1))
target_net.eval()

# Optimizer and loss function
optimizer = optim.Adam(policy_net.parameters())
loss_fn = torch.nn.CrossEntropyLoss()
print(policy_net(torch.from_numpy(observation.astype(np.float32))))

# Replay buffer
replay_buffer = deque(maxlen=5000)

# Function to store experience in replay buffer
def store_experience(state, action, reward, next_state, done):
    replay_buffer.append((state, action, reward, next_state, done))

# Sample a batch of experiences from the buffer
def sample_experiences(batch_size):
    return random.sample(replay_buffer, batch_size)
# Training parameters
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
target_update_frequency = 100
episodes = 50000
x = np.arange(episodes//100)
reward_list = np.zeros_like(x)
def test_dqn_agent(episodes=1):
    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        steps = 0

        while not done:
            board = get_board(state)
            plt.imshow(board)
            plt.axis("off")
            plt.show(block=False)  # Do not block the execution
            plt.pause(0.5)          # Pause for 0.5 seconds before the next iteration
            plt.clf()   
            action = np.argmax(policy_net(torch.FloatTensor(np.transpose(state,(2,0,1)))).detach().numpy())
            state, _, done, _, _ = env.step(action)
            steps += 1
        
        print(f"Test Episode {episode + 1}: Balanced for {steps} steps.")
    env.close()


for episode in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0
    iter = 0
    
    while not done:
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(policy_net(torch.FloatTensor(np.transpose(state,(2,0,1)))).detach().numpy())  # Exploit

        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        if not done:

            store_experience(state, action, reward, next_state, done)
            state = next_state

            # Train the network if the buffer has enough experiences
            if len(replay_buffer) > batch_size:
                experiences = sample_experiences(batch_size)
                states, actions, rewards, next_states, dones = zip(*experiences)
                
                # Convert to PyTorch tensors
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)
                
                # Calculate Q-values
                current_q_values = policy_net(states.permute((0,3,1,2))).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_net(next_states.permute((0,3,1,2))).max(1)[0]
                target_q_values = rewards + gamma * next_q_values * (1 - dones)

                loss = loss_fn(current_q_values, target_q_values.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
        iter+=1

    if episode % target_update_frequency == 0:
            target_net.load_state_dict(policy_net.state_dict())
    if episode % 100 == 0:
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")
            print(f"Update step  : {iter}")
            reward_list[episode//100] = total_reward
    if (episode%1000)  == 0:
        torch.save(policy_net.state_dict(), 'policy_net.pth')
        test_dqn_agent()

env.close()


# Test the agent
#test_dqn_agent()
plt.plot(x,reward_list)
plt.show()
"""