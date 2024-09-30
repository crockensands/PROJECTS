import numpy as np
import tkinter as tk
import random
import time  # Import time module for slowing down

GRID_SIZE = 5
CELL_SIZE = 100

class Ground:
    def __init__(self, canvas):
        self.canvas = canvas
        self.action_space = [0, 1, 2, 3]  # Up, Down, Left, Right
        self.state_space = list(range(GRID_SIZE * GRID_SIZE))
        self.state = 0
        self.steps_left = 200
        self.agent = canvas.create_rectangle(0, 0, CELL_SIZE, CELL_SIZE, fill="blue")
        self.goal = canvas.create_rectangle(CELL_SIZE * 4, CELL_SIZE * 4, CELL_SIZE * 5, CELL_SIZE * 5, fill="red")

    def step(self, action):
        x = self.state % GRID_SIZE
        y = self.state // GRID_SIZE

        if action == 0: y = max(0, y - 1)  # Up
        elif action == 1: y = min(GRID_SIZE - 1, y + 1)  # Down
        elif action == 2: x = max(0, x - 1)  # Left
        elif action == 3: x = min(GRID_SIZE - 1, x + 1)  # Right

        self.state = y * GRID_SIZE + x
        self.steps_left -= 1

        reward = 1 if self.state == 24 else -1
        done = self.steps_left <= 0 or self.state == 24

        # Update agent position
        self.canvas.coords(self.agent, x * CELL_SIZE, y * CELL_SIZE, (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE)
        return self.state, reward, done

    def reset(self):
        self.state = 0
        self.steps_left = 200
        self.canvas.coords(self.agent, 0, 0, CELL_SIZE, CELL_SIZE)
        return self.state

def run_simulation(episodes, canvas, window, episode_text):
    env = Ground(canvas)
    q_table = np.zeros((GRID_SIZE * GRID_SIZE, 4))

    learning_rate = 0.9
    discount_factor = 0.9
    epsilon = 1.0
    epsilon_decay = 0.99
    min_epsilon = 0.1

    for episode in range(episodes):
        state = env.reset()
        done = False

        # Update the badge to show the current episode
        canvas.itemconfig(episode_text, text=f"Episode: {episode + 1}")
        
        while not done:
            window.update()
            time.sleep(0.01)  # Add a 0.1 second delay (adjust to control speed)

            if random.uniform(0, 1) < epsilon:
                action = random.choice(env.action_space)
            else:
                action = np.argmax(q_table[state, :])

            new_state, reward, done = env.step(action)

            q_table[state, action] += learning_rate * (
                reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action]
            )

            state = new_state

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

# Set up the Tkinter window
window = tk.Tk()
canvas = tk.Canvas(window, width=GRID_SIZE * CELL_SIZE, height=GRID_SIZE * CELL_SIZE)
canvas.pack()

# Create a text element for showing the episode number (badge)
episode_text = canvas.create_text(10, 10, anchor="nw", font=("Arial", 16), text="Episode: 0", fill="green")

# Run the simulation
run_simulation(500, canvas, window, episode_text)
window.mainloop()
