import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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

def play():
    # Initialize game and agent
    game = SnakeGame()
    
    # Load the trained model
    model = DQN(11, 256, 3)
    model_folder = './model'
    model_path = os.path.join(model_folder, 'model.pth')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("Loaded trained model")
    else:
        print(f"No model found at {model_path}")
        return

    # Game loop
    while True:
        # Get old state
        state_old = game._get_state()
        
        # Get move
        state_old_tensor = torch.tensor(state_old, dtype=torch.float)
        with torch.no_grad():
            prediction = model(state_old_tensor)
        
        final_move = [0, 0, 0]
        move = torch.argmax(prediction).item()
        final_move[move] = 1
        
        # Perform move and get new state
        _, game_over, score = game.play_step(final_move)
        
        if game_over:
            game.reset()
            print(f'Game Over! Score: {score}')

if __name__ == '__main__':
    play()