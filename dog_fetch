import numpy as np
import matplotlib.pyplot as plt
import random
from random import sample

# Define the environment class
class Ground:
    def __init__(self):  # Constructor to initialize attributes
        self.action_space = [0, 1, 2, 3]  # Actions: left, right, up, down
        self.state_space: list[int] = list(range(25))  # State space: 0 to 24 (5x5 grid)
        self.state = 0  # Initial state
        self.steps_left = 200  # Steps limit

    def step(self, action):  # Update state based on action, return new state, reward, and termination flag
        grid_size = 5
        x = self.state % grid_size  # Get current x-coordinate
        y = self.state // grid_size  # Get current y-coordinate

        # Update coordinates based on action
        if action == 0:  # Up
            y = max(0, y - 1)
        elif action == 1:  # Down
            y = min(grid_size - 1, y + 1)
        elif action == 2:  # Left
            x = max(0, x - 1)
        elif action == 3:  # Right
            x = min(grid_size - 1, x + 1)

        new_state = y * grid_size + x  # Convert (x, y) back to state
        self.state = new_state  # Update current state
        self.steps_left -= 1  # Decrement steps left

        # Calculate reward: +1 for reaching the goal, -1 otherwise
        if self.state == 24:
            reward = 1
        else:
            reward = -1  # Use -0.1 to encourage quicker goal achievement (optional)

        # Check for termination condition
        terminate = self.steps_left <= 0
        return self.state, reward, terminate

    def reset(self):  # Reset the environment to the initial state
        self.state = 0
        self.steps_left = 200
        return self.state

    def render(self):  # Visualization method (to be implemented later)
        pass

def run(episodes, is_training=True, render=False):
    env = Ground()  # Create an instance of the environment

    # Initialize Q-table with zeros
    q = np.zeros((len(env.state_space), len(env.action_space)))  # 25 states x 4 actions

    # Hyperparameters for Q-learning
    learning_rate_a = 0.9  # Alpha: learning rate
    discount_factor_g = 0.9  # Gamma: discount rate
    epsilon = 1.0  # Initial exploration rate
    epsilon_decay = 0.99  # Decay factor for exploration rate
    min_epsilon = 0.1  # Minimum exploration rate
    rewards = []  # List to store rewards for plotting

    for episode in range(1, episodes + 1):  # Loop through episodes
        state = env.reset()  # Reset environment for new episode
        terminate = False  # Termination flag
        score = 0  # Initialize score for the episode

        while not terminate:  # Loop until episode terminates
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action = random.choice(env.action_space)  # Explore: choose random action
            else:
                action = np.argmax(q[state, :])  # Exploit: choose action with max Q-value

            # Take action and get new state, reward, and termination status
            n_state, reward, terminate = env.step(action)

            # Update Q-value using the Q-learning formula
            q[state, action] += learning_rate_a * (
                reward + discount_factor_g * np.max(q[n_state, :]) - q[state, action]
            )

            state = n_state  # Move to the new state
            score += reward  # Update score for the episode

        rewards.append(score)  # Store the total score for this episode
        #print(f'episode: {episode}, score: {score}')  # Print episode score

        # Decay epsilon to reduce exploration over time
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

    # Plotting rewards over episodes
    plt.plot(range(1, episodes + 1), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Rewards over Episodes')
    plt.show()

# Run the simulation n times   episodes
run(500)
#after 250 it stops exploration and finds a path
