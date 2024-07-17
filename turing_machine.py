import gym
from gym import spaces
import numpy as np

class TuringMachineEnv(gym.Env):
    def __init__(self, array_length=6):
        super(TuringMachineEnv, self).__init__()
        
        self.array_length = array_length
        self.max_steps = array_length * 3  # Adjust as needed
        
        # Action space: 0 - Left, 1 - Right, 2 - Swap, 3 - Quit
        self.action_space = spaces.Discrete(4)
        
        # Observation space: array_length integers (0-9) + head position
        self.observation_space = spaces.Box(low=0, high=9, 
                                            shape=(array_length + 1,), 
                                            dtype=np.int32)
        
        self.reset()
    
    def reset(self):
        self.tape = np.random.choice(10, self.array_length, replace=False)
        self.head_position = 0
        self.steps = 0
        return self._get_obs()
    
    def step(self, action):
        self.steps += 1
        done = False
        reward = 0
        
        if action == 0:  # Left
            if self.head_position > 0:
                self.head_position -= 1
        elif action == 1:  # Right
            if self.head_position < len(self.tape) - 1:
                self.head_position += 1
        elif action == 2:  # Swap
            if self.head_position < len(self.tape) - 1:
                self.tape[self.head_position], self.tape[self.head_position + 1] = \
                    self.tape[self.head_position + 1], self.tape[self.head_position]
        elif action == 3:  # Quit
            done = True
        
        # Calculate reward
        inversions = self._count_inversions()
        reward = self.previous_inversions - inversions
        self.previous_inversions = inversions
        
        if np.array_equal(self.tape, sorted(self.tape)):
            reward += 10  # Bonus for sorting the array
            done = True
        
        if self.steps >= self.max_steps:
            done = True
        
        return self._get_obs(), reward, done, {}
    
    def _get_obs(self):
        return np.append(self.tape, self.head_position)
    
    def _count_inversions(self):
        inv_count = 0
        for i in range(len(self.tape)):
            for j in range(i + 1, len(self.tape)):
                if self.tape[i] > self.tape[j]:
                    inv_count += 1
        return inv_count
    
    def render(self, mode='human'):
        print(f"Tape: {self.tape}, Head at: {self.head_position}")
