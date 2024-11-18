import numpy as np

class Grid:
    ACTIONS = ['U', 'D', 'L', 'R']
    
    def __init__(self, size, rewards, actions):
        self.height, self.width = size
        self.rewards = rewards
        self.actions = actions
        self.num_states = np.prod(size)
        self.num_actions = len(Grid.ACTIONS)
    
    def in_grid(self, state):
        r, c = state
        r = max(0, min(r, self.height - 1))
        c = max(0, min(c, self.width - 1))
        return (r, c)
    
    def ns_reward(self, action, state):
        r, c = state
        if action == 'U':
            r, c = r - 1, c
        elif action == 'D':
            r, c = r + 1, c
        elif action == 'R':
            r, c = r, c + 1
        elif action == 'L':
            r, c = r, c - 1
        ns = self.in_grid((r, c))
        return ns, self.rewards.get(ns, 0)
    
    def transition(self, action, state, choose=False):
        def move(possible_actions, prob):
            if not choose:
                result = []
                for i, a in enumerate(possible_actions):
                    coord, reward = self.ns_reward(a, state)
                    result.append((prob[i], coord, reward))
                return result
            else:
                a = np.random.choice(possible_actions, 1, p=prob)[0]
                coord, reward = self.ns_reward(a, state)
                return coord, reward

        if action == 'U':
            return move(['U', 'R', 'L'], [0.8, 0.1, 0.1])
        elif action == 'D':
            return move(['D', 'R', 'L'], [0.8, 0.1, 0.1])
        elif action == 'L':
            return move(['L', 'U', 'D'], [0.8, 0.1, 0.1])
        elif action == 'R':
            return move(['R', 'U', 'D'], [0.8, 0.1, 0.1])


def show_grid(env, content_dict):
    grid = np.full((env.height, env.width), '', dtype=object)
    for coord, content in content_dict.items():
        grid[coord[0], coord[1]] = content
    print(grid)


def play_game(env, start, end, policy):
    steps = []
    state = start
    while state not in end:
        state_idx = state[0] * env.width + state[1]
        action = policy[state_idx]
        ns, reward = env.transition(action, state, choose=True)
        steps.append([action, list(state), reward])
        state = ns
    steps.append(['G', list(end), 0])
    return steps


def epsilon_greedy(action, epsilon):
    p = np.random.random()
    if p < (1 - epsilon):
        return action
    else:
        return np.random.choice(Grid.ACTIONS, 1)[0]


def qlearning(env, num_episodes, epsilon, alpha, stop, discount_factor):
    all_states = set(env.rewards.keys())
    Q = {state: {action: 0 for action in Grid.ACTIONS} for state in all_states}
    
    for episode in range(num_episodes):
        row = np.random.randint(0, env.height)
        col = np.random.randint(0, env.width)
        start = (row, col)
        state = start
        
        while state not in stop:
            action = max(Q[state], key=Q[state].get)
            action = epsilon_greedy(action, epsilon)
            ns, reward = env.transition(action, state, choose=True)
            best_next_action = max(Q[ns], key=Q[ns].get)
            td_target = reward + discount_factor * Q[ns][best_next_action]
            Q[state][action] += alpha * (td_target - Q[state][action])
            state = ns
    
    policy = {state: max(Q[state], key=Q[state].get) if state not in stop else 'G' for state in all_states}
    q_values = {state: max(Q[state].values()) for state in all_states}
    return policy, q_values


if __name__ == '__main__':
    rewards = {
        (0, 0): 0, (0, 1): -80, (0, 2): 100,
        (1, 0): 0, (1, 1): 0, (1, 2): 0,
        (2, 0): 25, (2, 1): -100, (2, 2): 80,
    }
    actions = {
        (0, 0): ['D', 'R'],
        (0, 1): [],
        (0, 2): [],
        (1, 0): ['R', 'U', 'D'],
        (1, 1): ['R', 'L', 'U', 'D'],
        (1, 2): ['L', 'U', 'D'],
        (2, 0): [],
        (2, 1): [],
        (2, 2): []
    }
    stop = {(0, 2), (2, 2)}
    
    env = Grid((3, 3), rewards, actions)
    policy, q_values = qlearning(env, num_episodes=1000, epsilon=0.1, alpha=0.1, stop=stop, discount_factor=0.9)
    print("Policy:", policy)
    print("Q-values:", q_values)
