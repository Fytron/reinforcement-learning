import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
from IPython import display
import math
import os
from snake import SnakeGame

# Neural Network for Deep Q Learning
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=100000)
        self.model = DQN(11, 256, 3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Initialize plotting
        plt.ion()
        self.figure, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 8))
        self.scores = []
        self.mean_scores = []
        self.total_score = 0
        self.record = 0

    def update_plot(self):
        try:
            self.ax1.clear()
            self.ax2.clear()
            
            # Plot individual scores
            self.ax1.set_title('Training Progress')
            self.ax1.set_xlabel('Number of Games')
            self.ax1.set_ylabel('Score')
            self.ax1.plot(self.scores, label='Score', color='blue')
            self.ax1.plot(self.mean_scores, label='Average Score', color='red')
            if len(self.scores) > 0:
                self.ax1.text(len(self.scores)-1, self.scores[-1], str(self.scores[-1]))
            if len(self.mean_scores) > 0:
                self.ax1.text(len(self.mean_scores)-1, self.mean_scores[-1], 
                             str(round(self.mean_scores[-1], 2)))
            self.ax1.legend()
            
            # Plot score distribution
            self.ax2.set_title('Score Distribution')
            self.ax2.set_xlabel('Score')
            self.ax2.set_ylabel('Frequency')
            if len(self.scores) > 0:
                self.ax2.hist(self.scores, bins=min(20, len(set(self.scores))), 
                            color='skyblue', edgecolor='black')
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)
        except Exception as e:
            print(f"Plot update failed: {e}")

    def get_state(self, game):
        return game._get_state()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > 1000:
            mini_sample = random.sample(self.memory, 1000)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, dones)

    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(np.array(states), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float)
        dones = torch.tensor(np.array(dones), dtype=torch.bool)

        # Get Q values for current states
        current_q_values = self.model(states)
        
        # Get Q values for next states
        with torch.no_grad():
            next_q_values = self.model(next_states)
            max_next_q_values = next_q_values.max(1)[0]

        # Calculate target Q values
        target_q_values = current_q_values.clone()
        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]:
                Q_new = rewards[idx] + self.gamma * max_next_q_values[idx]
                
            # Use actions tensor directly without converting to tensor again
            action_idx = actions[idx].max(0)[1]
            target_q_values[idx, action_idx] = Q_new

        # Compute loss and update model
        self.optimizer.zero_grad()
        loss = F.mse_loss(target_q_values, current_q_values)
        loss.backward()
        self.optimizer.step()

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float, device='cpu').unsqueeze(0)
            with torch.no_grad():
                prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train(num_episodes=1000):
    agent = Agent()
    game = SnakeGame()
    
    try:
        for episode in range(num_episodes):
            state_old = agent.get_state(game)
            
            while True:
                final_move = agent.get_action(state_old)
                reward, done, score = game.play_step(final_move)
                state_new = agent.get_state(game)

                # Train short memory
                agent.train_step([state_old], [final_move], [reward], [state_new], [done])

                # Remember
                agent.remember(state_old, final_move, reward, state_new, done)
                
                state_old = state_new

                if done:
                    game.reset()
                    agent.n_games += 1
                    agent.train_long_memory()

                    if score > agent.record:
                        agent.record = score
                        agent.model.save()

                    print(f'Game {agent.n_games}/{num_episodes}, Score: {score}, Record: {agent.record}')

                    # Update plotting data
                    agent.scores.append(score)
                    agent.total_score += score
                    mean_score = agent.total_score / agent.n_games
                    agent.mean_scores.append(mean_score)
                    agent.update_plot()
                    break
            
            if agent.n_games >= num_episodes:
                print(f"Training completed after {num_episodes} episodes!")
                break
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        # Save final plot
        plt.savefig('training_results.png')
        plt.close()

if __name__ == '__main__':
    train(num_episodes=200)